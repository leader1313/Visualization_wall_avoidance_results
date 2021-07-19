# Visualization_wall_avoidance_results
A wall-avoidance task is implemented for writing journal of BDI. This project describe visualization wall-avoidance experiment results.
## To do:
- Expert's demonstrations
  - without disturbance injection
  - with uniform disturbance injection
  - with state-dependent disturbance injection
- Visualization of disturbacne's variance
  - data
  - graph
  - video
- env

## How to record a gym envs

```python
import gym
from gym.wrappers import Monitor
env = Monitor(gym.make('CartPole-v0'), './video', force=True)
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
env.close()
```

## Demonstration with injection disturbances

### get disturbances' variance dataset
`disturbanced_demo.py`: comment in bellow code
- get successful demo graph (state-dependent disturbance)
```python
S, A, R = self.env.sample(
    1, policy=self.sup, render=True, cov=self.disturbance_generator)
injected_d_x = self.env.results['Trajectories']['Success'][0][2][:, 0]
injected_d_y = self.env.results['Trajectories']['Success'][0][2][:, 1]
```
- get failure demo graph (state-independent disturbance)
```python
S, A, R = self.env.sample(
    1, policy=self.sup, render=True, cov=torch.tensor([0.4, 0.1]))
injected_d_x = self.env.results['Trajectories']['Fail'][0][2][:, 0]
injected_d_y = self.env.results['Trajectories']['Fail'][0][2][:, 1]
```

dataset will be save as `Data/Disturbance/*.pickle`

### get demo video of gym env
- gym video automatically recorded, when operating `disturbanced_demo.py`. 
- Video file are saved at `/Videos/gym_video/*.mp4`.


## How to get video of disturbances' variance graph
- Operating `VideoMaker.py`: change bellow part
```python
# Import data set
    dataset = torch.load('Data/Disturbances/<dataset_name>.pickle')
```
