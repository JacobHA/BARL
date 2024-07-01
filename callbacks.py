from BaseAgent import BaseAgent

class EvalCallback:
    def __init__(self, agent: 'BaseAgent'):
        self.agent = agent


class AUCCallback(EvalCallback):
    def __init__(self, agent: 'BaseAgent'):
        super().__init__(agent)
        self.auc = 0
    def __call__(self, state=None, action=None, reward=None, done=None, end=False):
        if end:
            self.agent.log_history('eval/auc', self.auc, self.agent.learn_env_steps)
            return
        self.auc += reward
