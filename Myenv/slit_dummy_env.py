import torch
import gym
from gym.utils import seeding


# from gym import error, spaces, utils

# state
# upper side: positive
# lower side: negative
# right side: positive
# left  side: negative


class SlitDummyEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Observation:
        Type: Box(2)
        Num     Observation               Min                     Max
        0       Cart X-Position           -10.0                   10.0
        1       Cart X-Velocity           -inf                    inf
        2       Cart Y-Position           -10.0                   10.0
        3       Cart Y-Velocity           -inf                    inf
    Actions:
        Type: Continuous(1)
        Num   Action                      Min                     Max
        0     push cart x                 -1                      1
        1     push cart y                 -1                      1
        Note:
    Reward:
        Reward is 1 for cart reaching to goal area.
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Cart collide to base.
        Center of the cart reaches the edge of the screen.
        Episode length is greater than 200.
    """

    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        self.STEP_LIMIT = 3000
        self.ENV_NOISE = 0.01
        # dynamics
        self.total_mass = self.masscart = 1.0
        self.force_scale = 30.0
        # tau -------------------------------------------
        # [collecting] slow : 0.015, fast: 0.02
        # [operating] 0.025
        self.tau = 0.025  # seconds between state updates
        self.kinematics_integrator = "euler"

        # field
        self.x_threshold = 10.0
        self.y_threshold = 10.0

        # action / state dimension
        self.statesize = 4
        self.actionsize = 2

        # goal
        self.goal_offset = 1.0
        # self.goal_pose = torch.tensor([0.0, -self.y_threshold])
        # self.goal_pose = torch.tensor([0.0, -self.y_threshold + 0.5])
        self.goal_pose = torch.tensor([-8.3, -self.y_threshold + 4.])

        # cart
        self.cart_init_pose = torch.tensor([0.0, 4.0])
        # self.cart_init_pose = torch.tensor([0.0, 8.0])
        self.cart_width = 1.0
        self.cart_height = 1.168

        # base
        self.wall_width = self.x_threshold / 12
        self.wall_height = self.y_threshold / 3
        self.obstacle_width = self.x_threshold * 1.5
        self.base_y_offset = -self.y_threshold / 3
        self.base_y_threshold = [
            self.base_y_offset - self.wall_height / 2 - self.cart_height / 2,
            self.base_y_offset + self.wall_height / 2 + self.cart_height / 2,
        ]
        self.obstacle_threshold = [
            -self.obstacle_width / 2 - self.cart_width / 2,
            self.obstacle_width / 2 + self.cart_width / 2,
        ]
        self.wall_threshold = [
            self.cart_width + self.wall_width / 2 - self.x_threshold,
            -self.cart_width - self.wall_width / 2 + self.x_threshold,
        ]

        self._seed()
        self.viewer = None
        self.state = None

    # --------------------------------------------------------

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.state_transition(action)
        death = self.judge_death()
        done = reward = self.compute_reward()
        return self.state, reward, done, death

    def state_transition(self, action):
        x, x_dot, y, y_dot = self.state

        d = action.shape[0]
        env_noise_power = self.ENV_NOISE
        action += torch.clamp(
            torch.normal(mean=torch.zeros(
                d), std=torch.ones(d) * env_noise_power),
            min=-1.0,
            max=1.0,
        )
        # friction = 0.1 * self.masscart
        friction = 0.15 * self.masscart

        force = action * self.force_scale

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force) / self.total_mass
        [xacc, yacc] = temp / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc - x_dot * friction
            y = y + self.tau * y_dot
            y_dot = y_dot + self.tau * yacc - y_dot * friction
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            y_dot = y_dot + self.tau * yacc
            y = y + self.tau * y_dot

        self.state = torch.tensor([x, x_dot, y, y_dot])

    def compute_reward(self):
        diff = [0, 0]
        diff[0] = self.goal_pose[0] - self.state[0]
        diff[1] = self.goal_pose[1] - self.state[2]
        L = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)
        reward = L < self.goal_offset
        return torch.tensor([reward]).float()

    def judge_goal(self, reward):
        return reward

    def judge_death(self):
        # out of world
        if (
            abs(self.state[0]) > self.x_threshold
            or abs(self.state[2]) > self.y_threshold
        ):
            return False
        # contact with base
        elif (
            self.state[2] > self.base_y_threshold[0]
            and self.state[2] < self.base_y_threshold[1]
        ):
            # contact-obstacle
            if (
                self.state[0] > self.obstacle_threshold[0]
                and self.state[0] < self.obstacle_threshold[1]
            ):
                return True
            # contact-wall
            elif (
                self.state[0] < self.wall_threshold[0]
                or self.state[0] > self.wall_threshold[1]
            ):
                return True
        else:
            return False

    def _reset(self):
        # print("\r reset env", end="")
        np_state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        init_state = (
            torch.cat(
                (
                    self.cart_init_pose[0][None],
                    torch.tensor([0.0]),
                    self.cart_init_pose[1][None],
                    torch.tensor([0.0]),
                )
            )
            + np_state
        )
        self.state = torch.tensor(init_state)
        return self.state

    # -----------------------------------------------------------------
    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # window size
        screen_width = 800
        screen_height = 800

        # world <-scale-> screen
        world_width = self.x_threshold * 2
        world_height = self.y_threshold * 2
        xscale = screen_width / world_width
        yscale = screen_height / world_height

        # circle size
        circle_r = self.goal_offset * xscale * 0.5

        # cart size
        cartwidth = self.cart_width * xscale
        cartheight = self.cart_height * yscale

        # wall size
        wallwidth = self.wall_width * xscale
        wallheight = self.wall_height * yscale

        # obstacle size
        obstaclewidth = self.obstacle_width * xscale
        obstacleheight = wallheight

        base_y_offset = self.base_y_offset * yscale

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform(
                translation=[
                    self.cart_init_pose[0] * xscale + (screen_width) / 2.0,
                    self.cart_init_pose[1] * yscale + (screen_height) / 2.0,
                ]
            )
            cart.add_attr(self.carttrans)
            cart.set_color(0, 0, 255)
            self.viewer.add_geom(cart)

            # wall_left
            l, r, t, b = -wallwidth / 2, wallwidth / 2, wallheight / 2, -wallheight / 2
            wall_l = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.walltrans_l = rendering.Transform(
                translation=[
                    wallwidth / 2,
                    screen_height / 2 + base_y_offset,
                ]
            )
            wall_l.add_attr(self.walltrans_l)
            wall_l.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(wall_l)

            # wall_right
            l, r, t, b = -wallwidth / 2, wallwidth / 2, wallheight / 2, -wallheight / 2
            wall_r = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.walltrans_r = rendering.Transform(
                translation=[
                    screen_width - wallwidth / 2,
                    screen_height / 2 + base_y_offset,
                ]
            )
            wall_r.add_attr(self.walltrans_r)
            wall_r.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(wall_r)

            # Obstacle
            l, r, t, b = (
                -obstaclewidth / 2,
                obstaclewidth / 2,
                obstacleheight / 2,
                -obstacleheight / 2,
            )
            obstacle = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.obstacletrans = rendering.Transform(
                translation=[
                    screen_width / 2,
                    screen_height / 2 + base_y_offset,
                ]
            )
            obstacle.add_attr(self.obstacletrans)
            obstacle.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(obstacle)

            # goal circle
            self.goal = rendering.make_circle(circle_r)
            self.goal.set_color(255, 0, 0)
            goal_pose = self.goal_pose.mul(torch.tensor([xscale, yscale]))
            self.goal_trans = rendering.Transform(
                translation=goal_pose
                + torch.tensor([screen_width / 2, screen_height / 2])
            )
            self.goal.add_attr(self.goal_trans)
            self.viewer.add_geom(self.goal)

        # calculate state
        pos = [
            # self.state[0] / self.x_threshold * (screen_width),
            self.state[0] * xscale + (screen_width) / 2.0,
            self.state[2] * yscale + (screen_height) / 2.0,
        ]

        # move cart
        self.carttrans.set_translation(pos[0], pos[1])

        # render
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
