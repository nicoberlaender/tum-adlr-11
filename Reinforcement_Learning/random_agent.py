class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_action(self):
        return self.action_space.sample()