"""Network monitoring utilities for tracking layer properties and gradient statistics."""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class EmptyMonitor:
    """A no-op network monitor."""

    def should_log(self) -> bool:
        return False

    def monitor_network(
        self,
        network: torch.nn.Module,
        prefix: str = "",
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        return {}

    def log_network_stats(
        self,
        agent,
        network: torch.nn.Module,
        prefix: str,
    ) -> None:
        pass

    @property
    def networks_to_monitor(self) -> List[str]:
        return []

    def reset(self) -> None:
        pass

class NetworkMonitor:
    """Monitors neural network layer properties and gradient statistics."""

    def __init__(
        self,
        track_weights: bool = True,
        track_gradients: bool = True, # TODO: check how e.g. target net without grads is handled
        compute_eigenvalues: bool = True,
        compute_stable_rank: bool = True,
        compute_lipschitz: bool = True,
        track_dormant_neurons: bool = True,
        dormant_threshold: float = 0.025,
        log_frequency: int = 1,
    ):
        """
        Args:
            track_weights: Log weight matrix properties
            track_gradients: Log gradient statistics
            compute_eigenvalues: Compute top eigenvalue (expensive for large layers)
            compute_stable_rank: Compute stable rank
            track_dormant_neurons: Track % of dormant neurons per layer
            dormant_threshold: Threshold τ for dormancy (neurons with score ≤ τ are dormant)
            log_frequency: Log every N calls to avoid overhead
        """
        self.track_weights = track_weights
        self.track_gradients = track_gradients
        self.compute_eigenvalues = compute_eigenvalues
        self.compute_stable_rank = compute_stable_rank
        self.compute_lipschitz = compute_lipschitz
        self.track_dormant_neurons = track_dormant_neurons
        self.dormant_threshold = dormant_threshold
        self.log_frequency = log_frequency

        self._call_count = 0
        self._prev_grads: Dict[str, torch.Tensor] = {}
        self._grad_history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._activation_hooks = []
        self._activations: Dict[str, List[torch.Tensor]] = {}

    def should_log(self) -> bool:
        """Check if we should log this iteration."""
        self._call_count += 1
        return (self._call_count % self.log_frequency) == 0

    def monitor_network(
        self,
        network: torch.nn.Module,
        prefix: str = "",
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute statistics for all layers in a network.
        
        Args:
            network: PyTorch module to monitor
            prefix: Prefix for metric names (e.g., "critic", "actor")
            step: Current training step (for gradient direction tracking)
            
        Returns:
            Dictionary of metric_name -> value
        """
        stats = {}
        lipschitz_log_sum = 0.0
        lipschitz_layer_count = 0
        
        for name, module in network.named_modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
                
            layer_name = f"{prefix}/{name}" if name else f"{prefix}/root"
            
            # Weight statistics
            if self.track_weights and hasattr(module, 'weight') and module.weight is not None:
                weight_stats = self._compute_weight_stats(
                    module.weight, layer_name
                )
                stats.update(weight_stats)
                if self.compute_lipschitz:
                    spec_key = f"{layer_name}/weight_spectral_norm"
                    if spec_key in weight_stats:
                        spec_val = max(float(weight_stats[spec_key]), 1e-12)
                        lipschitz_log_sum += np.log(spec_val)
                        lipschitz_layer_count += 1
            
            # Gradient statistics
            if self.track_gradients and hasattr(module, 'weight') and module.weight is not None:
                if module.weight.grad is not None:
                    grad_stats = self._compute_gradient_stats(
                        module.weight.grad, layer_name, step
                    )
                    stats.update(grad_stats)

        if self.compute_lipschitz and lipschitz_layer_count > 0:
            stats[f"{prefix}/lipschitz_estimate"] = float(np.exp(lipschitz_log_sum))

        return stats

    def _compute_weight_stats(
        self, weight: torch.Tensor, layer_name: str
    ) -> Dict[str, float]:
        """Compute weight matrix properties."""
        stats = {}
        
        with torch.no_grad():
            # Reshape to 2D if conv layer
            if weight.dim() > 2:
                w = weight.view(weight.size(0), -1)
            else:
                w = weight
            
            # Frobenius norm
            stats[f"{layer_name}/weight_norm"] = torch.norm(w, p='fro').item()
            
            # Spectral norm (largest singular value) - efficient approximation
            stats[f"{layer_name}/weight_spectral_norm"] = self._spectral_norm(w)
            
            # Mean and std
            stats[f"{layer_name}/weight_mean"] = w.mean().item()
            stats[f"{layer_name}/weight_std"] = w.std().item()
            
            # Stable rank (ratio of frobenius norm to spectral norm)
            if self.compute_stable_rank:
                fro_norm = stats[f"{layer_name}/weight_norm"]
                spec_norm = stats[f"{layer_name}/weight_spectral_norm"]
                if spec_norm > 1e-8:
                    stable_rank = (fro_norm / spec_norm) ** 2
                    stats[f"{layer_name}/stable_rank"] = stable_rank
            
            # Top eigenvalue of W^T W (approximate for efficiency)
            if self.compute_eigenvalues and min(w.shape) <= 512:  # Only for smaller layers
                try:
                    # Use power iteration for top eigenvalue (faster than full SVD)
                    top_eig = self._top_eigenvalue_power_iteration(w)
                    stats[f"{layer_name}/top_eigenvalue"] = top_eig
                except Exception:
                    pass

        return stats

    def _compute_gradient_stats(
        self, grad: torch.Tensor, layer_name: str, step: Optional[int]
    ) -> Dict[str, float]:
        """Compute gradient statistics."""
        stats = {}
        
        with torch.no_grad():
            # Reshape to 1D for consistent metrics
            g = grad.reshape(-1)
            
            # Magnitude
            grad_norm = torch.norm(g).item()
            stats[f"{layer_name}/grad_norm"] = grad_norm
            
            # Mean and std
            stats[f"{layer_name}/grad_mean"] = g.mean().item()
            stats[f"{layer_name}/grad_std"] = g.std().item()
            
            # Max absolute gradient (for detecting exploding gradients)
            stats[f"{layer_name}/grad_max_abs"] = g.abs().max().item()
            
            # Gradient direction change (cosine similarity with previous gradient)
            prev_key = f"{layer_name}_grad"
            if prev_key in self._prev_grads:
                prev_g = self._prev_grads[prev_key]
                if prev_g.shape == g.shape and grad_norm > 1e-8:
                    # Cosine similarity
                    cos_sim = torch.dot(g, prev_g) / (
                        torch.norm(g) * torch.norm(prev_g) + 1e-8
                    )
                    stats[f"{layer_name}/grad_cos_sim"] = cos_sim.item()
                    
                    # Angle in degrees
                    angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / np.pi
                    stats[f"{layer_name}/grad_angle_deg"] = angle.item()
            
            # Store current gradient for next comparison
            self._prev_grads[prev_key] = g.clone()

        return stats

    def _spectral_norm(self, weight: torch.Tensor, num_iters: int = 1) -> float:
        """Compute spectral norm (largest singular value) via power iteration."""
        with torch.no_grad():
            if weight.numel() == 0:
                return 0.0
            
            # For efficiency, use a single power iteration
            u = torch.randn(weight.size(0), device=weight.device)
            u = u / (torch.norm(u) + 1e-8)
            
            for _ in range(num_iters):
                v = weight.t() @ u
                v = v / (torch.norm(v) + 1e-8)
                u = weight @ v
                u = u / (torch.norm(u) + 1e-8)
            
            # Rayleigh quotient
            sigma = (u @ weight @ (weight.t() @ u)) / (u @ u + 1e-8)
            return torch.sqrt(torch.abs(sigma)).item()

    def _top_eigenvalue_power_iteration(
        self, weight: torch.Tensor, num_iters: int = 20
    ) -> float:
        """Compute top eigenvalue of W^T W using power iteration."""
        with torch.no_grad():
            wtw = weight.t() @ weight
            v = torch.randn(wtw.size(0), device=weight.device)
            v = v / (torch.norm(v) + 1e-8)
            
            for _ in range(num_iters):
                v_new = wtw @ v
                v_new = v_new / (torch.norm(v_new) + 1e-8)
                v = v_new
            
            eigenvalue = (v @ wtw @ v) / (v @ v + 1e-8)
            return eigenvalue.item()

    def log_network_stats(
        self,
        agent,
        network: torch.nn.Module,
        prefix: str,
    ) -> None:
        """
        Monitor network and log stats to agent's loggers.
        
        Args:
            agent: Agent with log_history method and learn_env_steps
            network: Network to monitor
            prefix: Prefix for metric names
        """
        stats = self.monitor_network(network, prefix, agent.learn_env_steps)
        for metric_name, value in stats.items():
            agent.log_history(metric_name, value, agent.learn_env_steps)

        # Log evaluation stats on fixed random samples if available
        samples = getattr(agent, "monitor_samples", None)
        if samples is not None:
            sample_stats = self._evaluate_on_samples(network, samples, prefix)
            for metric_name, value in sample_stats.items():
                agent.log_history(metric_name, value, agent.learn_env_steps)

            # Compute dormant neuron statistics on the same samples
            if self.track_dormant_neurons:
                dormant_stats = self._compute_dormant_neurons(network, samples, prefix)
                for metric_name, value in dormant_stats.items():
                    agent.log_history(metric_name, value, agent.learn_env_steps)

    def _compute_dormant_neurons(
        self, network: torch.nn.Module, samples: dict, prefix: str
    ) -> Dict[str, float]:
        """
        Compute percentage of dormant neurons per layer according to Definition 3.1.
        
        A neuron i in layer ℓ has score s_i^ℓ = E[|h_i^ℓ(x)|] / (1/H_ℓ * Σ_k E[|h_k^ℓ(x)|])
        A neuron is τ-dormant if s_i^ℓ ≤ τ.
        
        Args:
            network: Network to analyze
            samples: Dict with "states" and optionally "actions" tensors
            prefix: Prefix for metric names
            
        Returns:
            Dict mapping layer_name/pct_dormant -> percentage [0-100]
        """
        stats = {}
        states = samples.get("states")
        if states is None:
            return stats

        # Register forward hooks to capture activations
        self._activations.clear()
        self._activation_hooks.clear()

        def make_hook(name):
            def hook(module, input, output):
                # Store the output activations
                if isinstance(output, tuple):
                    output = output[0]
                self._activations[name] = output.detach()
            return hook

        # Register hooks for all linear/conv layers
        for name, module in network.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                layer_name = f"{prefix}/{name}" if name else f"{prefix}/root"
                handle = module.register_forward_hook(make_hook(layer_name))
                self._activation_hooks.append(handle)

        # Forward pass to collect activations
        with torch.no_grad():
            try:
                actions = samples.get("actions")
                _ = network(states, actions) if actions is not None else network(states)
            except Exception:
                try:
                    _ = network(states)
                except Exception:
                    # Clean up hooks
                    for handle in self._activation_hooks:
                        handle.remove()
                    return stats

        # Compute dormant neuron percentage for each layer
        for layer_name, activations in self._activations.items():
            try:
                # Flatten spatial dimensions for conv layers: [batch, channels, ...] -> [batch, channels]
                if activations.dim() > 2:
                    # Average over spatial dimensions
                    acts = activations.reshape(activations.size(0), activations.size(1), -1)
                    acts = acts.mean(dim=2)  # [batch, channels]
                else:
                    acts = activations  # [batch, neurons]

                # Compute mean absolute activation per neuron across batch
                mean_abs_activations = acts.abs().mean(dim=0)  # [num_neurons]

                # Normalize: divide by mean of all neuron activations
                # Score s_i = E[|h_i(x)|] / (1/H * Σ_k E[|h_k(x)|])
                sum_activations = mean_abs_activations.sum()
                num_neurons = mean_abs_activations.numel()

                if sum_activations > 1e-12 and num_neurons > 0:
                    # Normalized scores (sum to num_neurons, so average score is 1.0)
                    neuron_scores = mean_abs_activations / (sum_activations / num_neurons)

                    # Count dormant neurons
                    dormant_mask = neuron_scores <= self.dormant_threshold
                    pct_dormant = 100.0 * dormant_mask.float().mean().item()

                    stats[f"{layer_name}/pct_dormant"] = pct_dormant
                    stats[f"{layer_name}/num_dormant"] = float(dormant_mask.sum().item())
                    stats[f"{layer_name}/num_neurons"] = float(num_neurons)

            except Exception as e:
                # Skip layers that error
                pass

        # Remove hooks
        for handle in self._activation_hooks:
            handle.remove()
        self._activation_hooks.clear()

        return stats

    def _evaluate_on_samples(self, network: torch.nn.Module, samples: dict, prefix: str) -> Dict[str, float]:
        """Evaluate network outputs on fixed random samples and return summary stats."""
        stats = {}
        states = samples.get("states")
        actions = samples.get("actions")
        if states is None:
            return stats

        with torch.no_grad():
            try:
                outputs = network(states, actions) if actions is not None else network(states)
            except Exception:
                try:
                    outputs = network(states)
                except Exception:
                    return stats

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            out = outputs.reshape(-1).detach().cpu()

        if out.numel() == 0:
            return stats

        stats[f"{prefix}/sampled_output_min"] = float(out.min().item())
        stats[f"{prefix}/sampled_output_max"] = float(out.max().item())
        stats[f"{prefix}/sampled_output_mean"] = float(out.mean().item())
        stats[f"{prefix}/sampled_output_std"] = float(out.std().item())
        stats[f"{prefix}/sampled_output_median"] = float(out.median().item())
        return stats

    def reset(self) -> None:
        """Reset internal state (e.g., gradient history)."""
        self._prev_grads.clear()
        self._grad_history.clear()
        self._call_count = 0
        
        # Clean up any lingering hooks
        for handle in self._activation_hooks:
            handle.remove()
        self._activation_hooks.clear()
        self._activations.clear()


class NetworkMonitorCallback:
    """Callback for monitoring networks during training."""

    def __init__(
        self,
        monitor: NetworkMonitor,
        networks_to_monitor,
    ):
        """
        Args:
            monitor: NetworkMonitor instance
            networks_to_monitor: list of attribute names (e.g., ["critic", "actor"]) or
                                 dict mapping attr name to prefix
        """
        self.monitor = monitor
        self.networks_to_monitor = networks_to_monitor

    def __call__(self, agent) -> None:
        """Called by agent after gradient step."""
        if not self.monitor.should_log():
            return
        if isinstance(self.networks_to_monitor, dict):
            items = self.networks_to_monitor.items()
        else:
            items = ((name, name) for name in self.networks_to_monitor)

        for attr_name, prefix in items:
            network = getattr(agent, attr_name, None)
            if network is not None and isinstance(network, torch.nn.Module):
                self.monitor.log_network_stats(agent, network, prefix)


def create_monitor_for_agent(
    named_networks: List[str],
    log_frequency: int = 100,
    track_eigenvalues: bool = False,  # Expensive, disabled by default
    compute_stable_rank: bool = True,
    compute_lipschitz: bool = True,
    track_dormant_neurons: bool = True,
    dormant_threshold: float = 0.025,
) -> Tuple[NetworkMonitor, List[str]]:
    """
    Create a NetworkMonitor configured with explicit network attribute names.
    
    Args:
        named_networks: List of agent attribute names to monitor (e.g., ["critic", "actor"])
        log_frequency: How often to compute/log metrics
        track_eigenvalues: Whether to compute eigenvalues (expensive)
        track_dormant_neurons: Whether to track dormant neuron percentage
        dormant_threshold: Threshold τ for dormancy (default 0.025 = 2.5%)
        
    Returns:
        (monitor, named_networks)
    """
    monitor = NetworkMonitor(
        track_weights=True,
        track_gradients=True,
        compute_eigenvalues=track_eigenvalues,
        compute_stable_rank=compute_stable_rank,
        compute_lipschitz=compute_lipschitz,
        track_dormant_neurons=track_dormant_neurons,
        dormant_threshold=dormant_threshold,
        log_frequency=log_frequency,
    )
    
    return monitor, list(named_networks)
