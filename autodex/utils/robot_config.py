import numpy as np

XARM_INIT = np.array([
    -0.21991149, -0.20245819, -1.13620934, 2.33175988, 0.31939525, 2.36492114
])

ALLEGRO_INIT = np.array([
    0.0, 1.5707, 0.0, 0.0,
    0.0, 1.5707, 0.0, 0.0,
    0.0, 1.5707, 0.0, 0.0,
    1.24565697, 0.05513508, 0.23153956, -0.02217758
])

INIT_STATE = np.concatenate([XARM_INIT, ALLEGRO_INIT])

LINK6_TO_WRIST = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0.1552],
    [0, 0, 0, 1]
])
