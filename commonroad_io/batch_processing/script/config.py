from abc import ABC


class Config(ABC):
    pass


class Default(Config):
    SAVE_FIG = False
    PLOTTING_PARAMS = {
        0: ('gold', 'solid', 2.0),
        1: ('gold', '--', 2.0),
        2: ('red', 'solid', 2.0),
        3: ('grey', 'solid', 2.0),
        4: ('green', 'solid', 3.0)

    }

    JUPYTER_NOTEBOOK = True
    PLOT_LEGEND = True
    PLOT_COLLISION_STEPS = True


class StudentScript(Config):
    SAVE_FIG = False
    PLOTTING_PARAMS = {
        0: ('gold', 'solid', 2.0),
        1: ('gold', '--', 2.0),
        2: ('red', 'solid', 2.0),
        3: ('grey', 'solid', 2.0),
        4: ('green', 'solid', 3.0)

    }
    JUPYTER_NOTEBOOK = False
    PLOT_LEGEND = True
    PLOT_COLLISION_STEPS = True




