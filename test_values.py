import argparse
import json
import os
import shutil
import time

import torch as th

from utils_agents import AtariDQNAgent
from utils_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment_name', type=str)
    parser.add_argument('--environment_evaluation_frames', type=int, default=135000)
    parser.add_argument('--environment_no_op_max', type=int, default=30)
    parser.add_argument('--environment_frames_to_maxpool', type=int, default=2)
    parser.add_argument('--environment_state_height', type=int, default=84)
    parser.add_argument('--environment_state_width', type=int, default=84)

    parser.add_argument('--agent_history_length', type=int, default=4)
    parser.add_argument('--agent_action_repeat', type=int, default=4)
    parser.add_argument('--agent_update_frequency', type=int, default=1)

    parser.add_argument('--agent_checkpoint_path', type=str)
    parser.add_argument('--agent_discount_factor', type=float, default=0.99)
    parser.add_argument('--agent_epsilon_greedy_epsilon_initial', type=float, default=1.0)
    parser.add_argument('--agent_epsilon_greedy_epsilon_final', type=float, default=0.01)
    parser.add_argument('--agent_epsilon_greedy_annealing', type=int, default=1000000)
    parser.add_argument('--agent_epsilon_greedy_epsilon_evaluation', type=float, default=0.05)
    parser.add_argument('--agent_replay_buffer_initial', type=int, default=100000)
    parser.add_argument('--agent_replay_buffer_maximum_length', type=int, default=1000000)

    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--optimizer_minibatch_size', type=int, default=32)
    parser.add_argument('--optimizer_q_network_parameter_update_interval_frames', type=int, default=1000)
    parser.add_argument('--optimizer_learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer_gradient_momentum', type=float, default=0.95)
    parser.add_argument('--optimizer_squared_gradient_momentum', type=float, default=0.95)
    parser.add_argument('--optimizer_minimum_squared_gradient', type=float, default=0.01)
    parser.add_argument('--optimizer_gradient_clip_value', type=float, default=1.0)

    parser.add_argument('--gpu_id', type=int)
    args = parser.parse_args()

    save_path = os.path.join(os.path.splitext(args.agent_checkpoint_path)[0], 'values')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    evaluation_environment = gym.make(args.environment_name)
    evaluation_environment = MaxpoolPastKFrames(environment=evaluation_environment,
                                                environment_frames_to_maxpool=args.environment_frames_to_maxpool)
    evaluation_environment = SkipKFrames(environment=evaluation_environment,
                                         agent_action_repeat=args.agent_action_repeat)
    evaluation_environment = LimitActionSpace(environment=evaluation_environment)
    evaluation_environment = NOOPInitializationOnReset(environment=evaluation_environment,
                                                       environment_no_op_max=args.environment_no_op_max)
    evaluation_environment = ScaleAndGreyscaleObservation(environment=evaluation_environment,
                                                          environment_state_height=args.environment_state_height,
                                                          environment_state_width=args.environment_state_width)
    evaluation_environment = KPastStatesObservation(environment=evaluation_environment,
                                                    agent_history_length=args.agent_history_length)

    th.cuda.set_device(args.gpu_id)
    device = th.device(f'cuda:{args.gpu_id}')
    agent = AtariDQNAgent(environment_action_space=evaluation_environment.action_space,
                          environment_state_height=args.environment_state_height,
                          environment_state_width=args.environment_state_width,
                          agent_history_length=args.agent_history_length,
                          agent_action_repeat=args.agent_action_repeat,
                          agent_update_frequency=args.agent_update_frequency,
                          agent_discount_factor=args.agent_discount_factor,
                          agent_epsilon_greedy_epsilon_initial=args.agent_epsilon_greedy_epsilon_initial,
                          agent_epsilon_greedy_epsilon_final=args.agent_epsilon_greedy_epsilon_final,
                          agent_epsilon_greedy_annealing=args.agent_epsilon_greedy_annealing,
                          agent_epsilon_greedy_epsilon_evaluation=args.agent_epsilon_greedy_epsilon_evaluation,
                          agent_replay_buffer_initial=args.agent_replay_buffer_initial,
                          agent_replay_buffer_maximum_length=args.agent_replay_buffer_maximum_length,
                          optimizer_name=args.optimizer_name,
                          optimizer_minibatch_size=args.optimizer_minibatch_size,
                          optimizer_q_network_parameter_update_interval_frames=args.optimizer_q_network_parameter_update_interval_frames,
                          optimizer_learning_rate=args.optimizer_learning_rate,
                          optimizer_gradient_momentum=args.optimizer_gradient_momentum,
                          optimizer_squared_gradient_momentum=args.optimizer_squared_gradient_momentum,
                          optimizer_minimum_squared_gradient=args.optimizer_minimum_squared_gradient,
                          device=device)

    total_start_time = time.time()
    total_frames = 0
    total_episodes = 0
    current_episode_reward = 0
    current_episode_frames = 0
    all_episode_rewards = []
    all_episode_frames = []
    all_frame_values = []

    state = evaluation_environment.reset()
    episode_start_time = time.time()
    agent.current_q_network.load_state_dict(th.load(args.agent_checkpoint_path))

    while total_frames < args.environment_evaluation_frames:
        action = agent.sample_action_evaluation(state)
        for _ in range(args.agent_update_frequency):
            all_q_values = agent.eval_agent(state)
            all_frame_values.append(np.mean(all_q_values))

            state, reward, done, _ = evaluation_environment.step(action)

            current_episode_reward += reward
            current_episode_frames += 1
            total_frames += 1

            if done:
                all_episode_rewards.append(current_episode_reward)
                all_episode_frames.append(current_episode_frames)
                print(
                    f'Total Frames {total_frames} | Episode {total_episodes} | '
                    f'Reward {current_episode_reward} | Frames {current_episode_frames} | '
                    f'Time {time.time() - episode_start_time:.4f} | '
                    f'Total Time {time.time() - total_start_time:.4f}')

                state = evaluation_environment.reset()
                episode_start_time = time.time()
                total_episodes += 1
                current_episode_reward = 0
                current_episode_frames = 0

    print(
        f'Total Frames {total_frames} | Episode {total_episodes} | '
        f'Reward {current_episode_reward} | Frames {current_episode_frames} | '
        f'Time {time.time() - episode_start_time:.4f} | '
        f'Total Time {time.time() - total_start_time:.4f}')
    output_dict = {'Value Mean Over Frames': statistics.mean(all_frame_values),
                   'Value Variance Over Frames': statistics.variance(all_frame_values),
                   'Score Mean Over Episodes': statistics.mean(all_episode_rewards),
                   'Score Variance Over Episodes': statistics.variance(all_episode_rewards),
                   'Length Mean Over Episodes': statistics.mean(all_episode_frames),
                   'Length Variance Over Episodes': statistics.variance(all_episode_frames)}
    json.dumps(os.path.join(save_path, output_dict))
    print(output_dict)
