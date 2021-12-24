from gym.envs.registration import register

register(
    id="position_control_2d-v0",
    entry_point="Myenv.position_control_2d_env:PositionControl2DEnv",
)

register(id="slit-v0", entry_point="Myenv.slit_env:SlitEnv")
register(id="slit-v1", entry_point="Myenv.slit_video_env:SlitVideoEnv")
register(id="slit-dummy-v0", entry_point="Myenv.slit_dummy_env:SlitDummyEnv")
