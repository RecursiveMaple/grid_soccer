from gymnasium.envs.registration import register

register(
    id="GridSoccer-6x4",
    entry_point="grid_soccer.soccer:GridSoccerEnv",
    kwargs={"grid_size": (6, 4), "max_steps": 50},
)

register(
    id="GridSoccer-8x6",
    entry_point="grid_soccer.soccer:GridSoccerEnv",
    kwargs={"grid_size": (8, 6), "max_steps": 100},
)
