from .env.monopoly_go import MonopolyGoEnv

class env(MonopolyGoEnv):

    def __init__(self, render_mode):
        super().__init__()
