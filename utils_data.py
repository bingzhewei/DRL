import functools
import random
import statistics
from collections import deque

import cv2
import gym
import numpy as np


class KPastStatesObservation(gym.Wrapper):
    def __init__(self, environment, agent_history_length):
        super().__init__(environment)
        self.environment = environment
        assert (agent_history_length >= 1)
        self.agent_history_length = agent_history_length

        self.k_past_states_queue = deque(maxlen=self.agent_history_length)
        for _ in range(self.agent_history_length):
            self.k_past_states_queue.append(np.zeros(self.environment.observation_space.shape, dtype=np.uint8))

        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(
                                                      self.agent_history_length,) + self.environment.observation_space.shape,
                                                dtype=np.uint8)

    def step(self, action):
        state, reward, done, info = self.environment.step(action)
        self.k_past_states_queue.append(state)
        return tuple(self.k_past_states_queue), reward, done, info

    def reset(self, **kwargs):
        for _ in range(self.agent_history_length):
            self.k_past_states_queue.append(np.zeros(self.environment.observation_space.shape, dtype=np.uint8))

        state = self.environment.reset(**kwargs)
        self.k_past_states_queue.append(state)

        return tuple(self.k_past_states_queue)


class ScaleAndGreyscaleObservation(gym.ObservationWrapper):
    def __init__(self, environment, environment_state_height, environment_state_width):
        super().__init__(environment)
        self.environment = environment
        assert (environment_state_height >= 1)
        assert (environment_state_width >= 1)
        self.environment_state_height = environment_state_height
        self.environment_state_width = environment_state_width

        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.environment_state_height, self.environment_state_width),
                                                dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (self.environment_state_width, self.environment_state_height))
        return frame


class NOOPInitializationOnReset(gym.Wrapper):
    def __init__(self, environment, environment_no_op_max):
        super().__init__(environment)
        self.environment = environment
        assert (environment_no_op_max >= 1)
        self.environment_no_op_max = environment_no_op_max
        self.environment_no_op_action_index = self.environment.unwrapped.get_action_meanings().index('NOOP')

    def step(self, action):
        return self.environment.step(action)

    def reset(self, **kwargs):
        state = self.environment.reset(**kwargs)

        for _ in range(random.randint(1, self.environment_no_op_max)):
            state, _, done, _ = self.step(self.environment_no_op_action_index)
            if done:
                state = self.environment.reset(**kwargs)

        return state


class EpisodeEndOnLifeLoss(gym.Wrapper):
    def __init__(self, environment):
        super().__init__(environment)
        self.environment = environment

        self.most_recent_lives = None

    def step(self, action):
        state, reward, done, info = self.environment.step(action)
        final_lives = self.environment.unwrapped.ale.lives()

        if 0 < final_lives < self.most_recent_lives:
            done = True

        self.most_recent_lives = final_lives
        return state, reward, done, info

    def reset(self, **kwargs):
        state = self.environment.reset(**kwargs)
        self.most_recent_lives = self.environment.unwrapped.ale.lives()
        return state


class MaxpoolPastKFrames(gym.Wrapper):
    def __init__(self, environment, environment_frames_to_maxpool):
        super().__init__(environment)
        self.environment = environment
        assert (environment_frames_to_maxpool >= 1)
        self.environment_frames_to_maxpool = environment_frames_to_maxpool

        self.k_past_states_queue = deque(maxlen=self.environment_frames_to_maxpool)
        for _ in range(self.environment_frames_to_maxpool):
            self.k_past_states_queue.append(
                np.zeros(self.environment.observation_space.shape, dtype=np.uint8))

    def step(self, action):
        state, reward, done, info = self.environment.step(action)
        self.k_past_states_queue.append(state)
        return functools.reduce(np.maximum, self.k_past_states_queue), reward, done, info

    def reset(self, **kwargs):
        for _ in range(self.environment_frames_to_maxpool):
            self.k_past_states_queue.append(
                np.zeros(self.environment.observation_space.shape, dtype=np.uint8))

        state = self.environment.reset(**kwargs)
        self.k_past_states_queue.append(state)

        return functools.reduce(np.maximum, self.k_past_states_queue)


class SkipKFrames(gym.Wrapper):
    def __init__(self, environment, agent_action_repeat):
        super().__init__(environment)
        self.environment = environment
        assert (agent_action_repeat >= 0)
        self.agent_action_repeat = agent_action_repeat

    def step(self, action):
        total_reward = 0

        for _ in range(self.agent_action_repeat):
            state, reward, done, info = self.environment.step(action)
            total_reward += reward
            if done:
                break

        return state, total_reward, done, info

    def reset(self, **kwargs):
        return self.environment.reset(**kwargs)


class RewardClipping(gym.RewardWrapper):
    def __init__(self, environment, environment_reward_clip_limit):
        super().__init__(environment)
        assert (environment_reward_clip_limit >= 0)
        self.environment_reward_clip_limit = environment_reward_clip_limit

    def reward(self, reward):
        if reward < -self.environment_reward_clip_limit:
            return -self.environment_reward_clip_limit
        elif reward > self.environment_reward_clip_limit:
            return self.environment_reward_clip_limit
        else:
            return reward


class LimitActionSpace(gym.Wrapper):
    def __init__(self, environment):
        super().__init__(environment)
        self.environment = environment

        if 'pong' in self.environment.game:
            self.environment.action_space.n = 4
            print(f'Pong: Limiting to first {self.environment.action_space.n} actions.')
        elif 'enduro' in self.environment.game:
            self.environment.action_space.n = 5
            print(f'Enduro: Limiting to first {self.environment.action_space.n} actions.')
        elif 'riverraid' in self.environment.game:
            self.environment.action_space.n = 6
            print(f'RiverRaid: Limiting to first {self.environment.action_space.n} actions.')
        elif 'seaquest' in self.environment.game:
            self.environment.action_space.n = 6
            print(f'SeaQuest: Limiting to first {self.environment.action_space.n} actions.')
        elif 'space_invaders' in self.environment.game:
            self.environment.action_space.n = 4
            print(f'SpaceInvaders: Limiting to first {self.environment.action_space.n} actions.')

        print(f'Action Space: {self.environment.unwrapped.get_action_meanings()}')

    def step(self, action):
        return self.environment.step(action)

    def reset(self, **kwargs):
        return self.environment.reset(**kwargs)


class ReplayBuffer:
    def __init__(self, maxlen):
        super(ReplayBuffer, self).__init__()
        self.buffer = []
        self.index = 0
        self.maxlen = maxlen

    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.maxlen:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.index % self.maxlen] = (state, action, reward, next_state, done)
        self.index += 1

    def sample(self, batch_size):
        return tuple(zip(*random.choices(self.buffer, k=batch_size)))

    def __len__(self):
        return len(self.buffer)


class MovingAverage:
    def __init__(self, num_to_average):
        super(MovingAverage, self).__init__()
        self.buffer = deque(maxlen=num_to_average)

    def append(self, value):
        self.buffer.append(value)

    def mean(self):
        return statistics.mean(self.buffer)

    def median(self):
        return statistics.median(self.buffer)

    def variance(self):
        if len(self.buffer) <= 1:
            return 0
        else:
            return statistics.variance(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def num_to_average(self):
        return self.buffer.maxlen
