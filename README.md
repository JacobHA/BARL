# 🛢️
BARL: Base Agents for Reinforcement Learning

This codebase provides implementations of many RL algorithms with the goal of being flexible for new algorithm ideas and fast experimentation.




## TODO:
- [ ] Architecture kwargs (rename to model?)
- [ ] reduce branching in preprocess obs, maybe cache which preprocess for each env (map of env to function)
- [ ] add more tests
- [ ] dueling architecture
- [ ] hyperparameters
- [ ] random seeds
- [ ] double check Polyak averaging tau, timing
- [ ] Clean up eval thread


### Algorithms:
- [x] DQN
- [x] SQL (discrete actions)
- [ ] SAC
- [ ] Rainbow DQN
- [ ] TD3
- [ ] PPO

### Architectures:
- [x] MLP
- [x] CNN
- [ ] LSTM
- [ ] Dueling architectures
      
