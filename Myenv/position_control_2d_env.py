import numpy as np
import gym
from gym.utils import seeding
from gym import error, spaces, utils

# state
# upper side: positive
# lower side: negative
# right side: positive
# left  side: negative


class PositionControl2DEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        self.viewer = None

        # field
        self.max_position_x = 1.0
        self.max_position_y = 1.0

        # action / state dimension
        self.statesize = 2
        self.actionsize = 2

        # action
        self.max_action = 0.3
        self.action_clip = self.max_action

        # goal
        self.goal_offset = 0.10
        self.subgoal = [self.max_position_x, 0.0]

        self.state = np.array([0.0, 0.0])

    # --------------------------------------------------------

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, a):
        # action = np.clip(a, -self.max_action, self.max_action)
        action = a
        self.state_transition(action)
        death = self.judge_death()
        reward = self.cal_reward()
        done = self.judge_goal(reward)
        return self.state, reward, done, death

    def state_transition(self, action):
        # transition
        self.state[0] = self.state[0] + action[0]
        self.state[1] = self.state[1] + action[1]

    def cal_reward(self):
        diff = [0, 0]
        diff[0] = self.subgoal[0] - self.state[0]
        diff[1] = self.subgoal[1] - self.state[1]
        L = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        return L

    def judge_goal(self, reward):
        if reward < self.goal_offset:
            return True
        else:
            return False

    def judge_death(self):
        if (
            self.state[0] > self.max_position_x
            or abs(self.state[1]) > self.max_position_y
        ):
            return True
        else:
            return False

    def _reset(self):
        print("reset env")
        self.state[0] = 0.0
        self.state[1] = 0.0
        return self.state

    # -----------------------------------------------------------------
    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # window size
        screen_width = 1000
        screen_height = 500

        # circle size
        r = 30

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # center line
            xs = np.linspace(0, 16, 10)
            ys = np.zeros(10)
            xys = list(zip(xs * screen_width, ys))
            self.state_line = rendering.make_polyline(xys)
            self.state_line.add_attr(
                rendering.Transform(translation=[0, screen_height / 2])
            )
            self.viewer.add_geom(self.state_line)

            # start circle
            self.start = rendering.make_circle(r)
            self.start.set_color(0, 0, 255)
            self.start.add_attr(
                rendering.Transform(
                    translation=[
                        0,
                        screen_height / 2,
                    ]
                )
            )
            self.viewer.add_geom(self.start)

            # goal circle
            self.goal = rendering.make_circle(r)
            self.goal.set_color(255, 0, 0)
            self.goal.add_attr(
                rendering.Transform(
                    translation=[
                        screen_width,
                        screen_height / 2,
                    ]
                )
            )
            self.viewer.add_geom(self.goal)

            # agent circle
            self.agent = rendering.Transform()
            agent_circle = rendering.make_circle(r)
            agent_circle.set_color(0, 255, 0)
            agent_circle.add_attr(self.agent)
            self.viewer.add_geom(agent_circle)

        # calculate state
        pos = [
            self.state[0] / self.max_position_x * (screen_width),
            self.state[1] / self.max_position_y * (screen_height / 2)
            + (screen_height / 2),
        ]

        # move agent circle
        self.agent.set_translation(pos[0], pos[1])

        # render
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
