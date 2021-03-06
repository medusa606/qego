from abc import ABC

import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding


class Agent(ABC):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)

        self.index = index

    def reset(self):
        raise NotImplementedError

    def choose_action(self, state, action_space, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_state, action, state, reward, done):
        raise NotImplementedError


class NoopAgent(Agent):
    def __init__(self, noop_action, **kwargs):
        super().__init__(**kwargs)

        self.noop_action = noop_action

    def reset(self):
        pass

    def choose_action(self, state, action_space, info=None):
        return self.noop_action

    def process_feedback(self, previous_state, action, state, reward, done):
        pass


class RandomAgent(NoopAgent):
    def __init__(self, epsilon, np_random=seeding.np_random(None)[0], **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.np_random = np_random

        self.action = self.noop_action

    def reset(self):
        self.action = self.noop_action

    def choose_action(self, state, action_space, info=None):
        if self.epsilon_valid():
            action_space_sample = action_space.sample()
            self.action = list(action_space_sample) if isinstance(action_space_sample, np.ndarray) else action_space_sample  # update and store action (so that it can be repeated)
        return self.action

    def process_feedback(self, previous_state, action, state, reward, done):
        pass

    def epsilon_valid(self):
        return self.np_random.uniform(0.0, 1.0) < self.epsilon
