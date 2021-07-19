import gym
from gym.wrappers import Monitor
env = Monitor(gym.make('CartPole-v0'), './video', force=True)
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
env.close()
