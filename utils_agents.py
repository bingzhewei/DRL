import random

import numpy as np
import torch as th
import torch.nn as nn

from utils_data import ReplayBuffer
from utils_layers import AtariDQNNetwork


class AtariDQNAgent:
    def __init__(self, environment_action_space,
                 environment_state_height, environment_state_width,
                 agent_history_length, agent_action_repeat, agent_update_frequency,
                 agent_discount_factor, agent_epsilon_greedy_epsilon_initial, agent_epsilon_greedy_epsilon_final,
                 agent_epsilon_greedy_annealing, agent_epsilon_greedy_epsilon_evaluation, agent_replay_buffer_initial,
                 agent_replay_buffer_maximum_length,
                 optimizer_name, optimizer_minibatch_size, optimizer_q_network_parameter_update_interval_frames,
                 optimizer_learning_rate, optimizer_gradient_momentum, optimizer_squared_gradient_momentum,
                 optimizer_minimum_squared_gradient, device):
        super(AtariDQNAgent, self).__init__()
        self.environment_action_space = environment_action_space
        self.environment_state_height = environment_state_height
        self.environment_state_width = environment_state_width

        self.agent_history_length = agent_history_length
        self.agent_action_repeat = agent_action_repeat
        self.agent_update_frequency = agent_update_frequency
        self.agent_discount_factor = agent_discount_factor
        self.agent_epsilon_greedy_epsilon_initial = agent_epsilon_greedy_epsilon_initial
        self.agent_epsilon_greedy_epsilon_final = agent_epsilon_greedy_epsilon_final
        self.agent_epsilon_greedy_annealing = agent_epsilon_greedy_annealing
        self.agent_epsilon_greedy_epsilon_evaluation = agent_epsilon_greedy_epsilon_evaluation
        self.agent_replay_buffer_initial = agent_replay_buffer_initial
        self.agent_replay_buffer_maximum_length = agent_replay_buffer_maximum_length

        self.optimizer_name = optimizer_name
        self.optimizer_minibatch_size = optimizer_minibatch_size
        self.optimizer_q_network_parameter_update_interval_frames = optimizer_q_network_parameter_update_interval_frames
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_gradient_momentum = optimizer_gradient_momentum
        self.optimizer_squared_gradient_momentum = optimizer_squared_gradient_momentum
        self.optimizer_minimum_squared_gradient = optimizer_minimum_squared_gradient
        self.device = device

        self.agent_epsilon_greedy_current_epsilon = self.agent_epsilon_greedy_epsilon_initial

        self.replay_buffer = ReplayBuffer(self.agent_replay_buffer_maximum_length)

        self.current_q_network = AtariDQNNetwork(self.environment_action_space.n)
        self.old_q_network = AtariDQNNetwork(self.environment_action_space.n)

        self.old_q_network.load_state_dict(self.current_q_network.state_dict())
        self.current_q_network.train()
        self.old_q_network.train()
        self.current_q_network.to(device=self.device)
        self.old_q_network.to(device=self.device)

        if self.optimizer_name == 'rmsprop':
            self.optimizer = th.optim.RMSprop(params=self.current_q_network.parameters(),
                                              lr=self.optimizer_learning_rate,
                                              alpha=self.optimizer_squared_gradient_momentum,
                                              eps=self.optimizer_minimum_squared_gradient)
        elif self.optimizer_name == 'adam':
            self.optimizer = th.optim.Adam(params=self.current_q_network.parameters(), lr=self.optimizer_learning_rate)
        self.loss_function = nn.SmoothL1Loss().to(device=self.device)

    def train_agent(self, total_frames):
        if len(self.replay_buffer) < self.agent_replay_buffer_initial:
            return 0
        else:
            batched_samples = self.replay_buffer.sample(self.optimizer_minibatch_size)
            state_batch = th.as_tensor(np.array(batched_samples[0], dtype=np.float32)).pin_memory()
            state_batch = state_batch.to(device=self.device, non_blocking=True)
            action_batch = th.as_tensor(np.array(batched_samples[1], dtype=np.int64)).pin_memory()
            action_batch = action_batch.to(device=self.device, non_blocking=True)
            reward_batch = th.as_tensor(np.array(batched_samples[2], dtype=np.float32)).pin_memory()
            reward_batch = reward_batch.to(device=self.device, non_blocking=True)
            next_state_batch = th.as_tensor(np.array(batched_samples[3], dtype=np.float32)).pin_memory()
            next_state_batch = next_state_batch.to(device=self.device, non_blocking=True)
            done_batch = th.as_tensor(np.array(batched_samples[4], dtype=np.bool)).pin_memory()
            done_batch = done_batch.to(device=self.device, non_blocking=True)

            old_q_network_q_values = self.old_q_network(next_state_batch[~done_batch])
            old_q_network_best_action_q_values = th.max(old_q_network_q_values, dim=-1)[0].detach()

            target_q_values = th.zeros(self.optimizer_minibatch_size, device=self.device)
            target_q_values[~done_batch] = old_q_network_best_action_q_values
            target_q_values = reward_batch + self.agent_discount_factor * target_q_values

            current_q_network_q_values = self.current_q_network(state_batch)
            current_q_network_taken_action_q_values = th.gather(current_q_network_q_values, dim=-1,
                                                                index=action_batch.view(-1, 1))
            current_q_network_taken_action_q_values = current_q_network_taken_action_q_values.squeeze(dim=-1)
            loss = self.loss_function(current_q_network_taken_action_q_values, target_q_values)
            loss_value = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if total_frames % self.optimizer_q_network_parameter_update_interval_frames == 0:
                self.old_q_network.load_state_dict(self.current_q_network.state_dict())
            return loss_value

    def eval_agent(self, state):
        state = th.as_tensor(np.array(state), dtype=th.float).pin_memory()
        state = state.to(device=self.device, non_blocking=True)
        state = state.unsqueeze(dim=0)
        self.current_q_network.eval()
        with th.no_grad():
            all_q_values = self.current_q_network(state)

        all_q_values = all_q_values.squeeze(dim=0).cpu().numpy()
        self.current_q_network.train()
        return all_q_values

    def sample_action_train(self, state, total_frames):
        self.agent_epsilon_greedy_current_epsilon = max(self.agent_epsilon_greedy_epsilon_final,
                                                        self.agent_epsilon_greedy_epsilon_initial - ((
                                                                                                             self.agent_epsilon_greedy_epsilon_initial - self.agent_epsilon_greedy_epsilon_final) / self.agent_epsilon_greedy_annealing * total_frames))
        if len(
                self.replay_buffer) < self.agent_replay_buffer_initial or random.random() < self.agent_epsilon_greedy_current_epsilon:
            return random.randrange(self.environment_action_space.n)
        else:
            return self.best_action(state)

    def sample_action_evaluation(self, state):
        if random.random() < self.agent_epsilon_greedy_epsilon_evaluation:
            return random.randrange(self.environment_action_space.n)
        else:
            return self.best_action(state)

    def best_action(self, state):
        state = th.as_tensor(np.array(state), dtype=th.float).pin_memory()
        state = state.to(device=self.device, non_blocking=True)
        state = state.unsqueeze(dim=0)
        self.current_q_network.eval()
        with th.no_grad():
            best_action = th.argmax(self.current_q_network(state), dim=-1)

        best_action = best_action.squeeze(dim=0).item()
        self.current_q_network.train()
        return best_action

    def replay_buffer_append(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)
