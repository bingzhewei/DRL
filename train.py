import argparse
import datetime
import os
import shlex
import subprocess
import time

import tensorboardX.summary
import torch as th

from utils_agents import AtariDQNAgent
from utils_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment_name', type=str)
    parser.add_argument('--environment_train_frames_max', type=int, default=10000000)
    parser.add_argument('--environment_evaluation_episodes', type=int, default=10)
    parser.add_argument('--environment_evaluation_interval_episodes', type=int, default=500)
    parser.add_argument('--environment_checkpoint_interval_episodes', type=int, default=500)
    parser.add_argument('--environment_no_op_max', type=int, default=30)
    parser.add_argument('--environment_reward_clip_limit', type=float, default=1.0)
    parser.add_argument('--environment_frames_to_maxpool', type=int, default=2)
    parser.add_argument('--environment_state_height', type=int, default=84)
    parser.add_argument('--environment_state_width', type=int, default=84)

    parser.add_argument('--agent_history_length', type=int, default=4)
    parser.add_argument('--agent_action_repeat', type=int, default=4)
    parser.add_argument('--agent_update_frequency', type=int, default=1)

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
    parser.add_argument('--log_and_model_base_save_path', type=str, default='/mnt/data/runs')
    parser.add_argument('--log_moving_average_num_episodes', type=int, default=100)
    args = parser.parse_args()
    vars(args)['model'] = 'DQN'

    git_branch_name = subprocess.check_output(shlex.split('git rev-parse --abbrev-ref HEAD')).decode('ascii').strip()
    git_commit_hash = subprocess.check_output(shlex.split('git rev-parse --short HEAD')).decode('ascii').strip()
    log_and_model_save_path = os.path.join(args.log_and_model_base_save_path,
                                           f'{git_branch_name}_{git_commit_hash}',
                                           f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    writer = tensorboardX.SummaryWriter(logdir=log_and_model_save_path)
    exp, ssi, sei = tensorboardX.summary.hparams(hparam_dict=vars(args), metric_dict={})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    train_environment = gym.make(args.environment_name)
    train_environment = MaxpoolPastKFrames(environment=train_environment,
                                           environment_frames_to_maxpool=args.environment_frames_to_maxpool)
    train_environment = SkipKFrames(environment=train_environment, agent_action_repeat=args.agent_action_repeat)
    train_environment = LimitActionSpace(environment=train_environment)
    train_environment = NOOPInitializationOnReset(environment=train_environment,
                                                  environment_no_op_max=args.environment_no_op_max)
    train_environment = EpisodeEndOnLifeLoss(environment=train_environment)
    # train_environment = RewardClipping(environment=train_environment,
    #                                    environment_reward_clip_limit=args.environment_reward_clip_limit)
    train_environment = ScaleAndGreyscaleObservation(environment=train_environment,
                                                     environment_state_height=args.environment_state_height,
                                                     environment_state_width=args.environment_state_width)
    train_environment = KPastStatesObservation(environment=train_environment,
                                               agent_history_length=args.agent_history_length)

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
    agent = AtariDQNAgent(environment_action_space=train_environment.action_space,
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
    episode_reward_moving_average = MovingAverage(num_to_average=args.log_moving_average_num_episodes)
    episode_loss_moving_average = MovingAverage(num_to_average=args.log_moving_average_num_episodes)
    episode_frames_moving_average = MovingAverage(num_to_average=args.log_moving_average_num_episodes)
    total_frames = 0
    previous_evaluation_episodes = 0
    previous_checkpoint_episodes = 0
    total_episodes = 0
    current_episode_loss = 0
    current_episode_reward = 0
    current_episode_frames = 0

    state = train_environment.reset()
    episode_start_time = time.time()

    while total_frames < args.environment_train_frames_max:
        for _ in range(args.agent_update_frequency):
            action = agent.sample_action_train(state, total_frames)
            next_state, reward, done, _ = train_environment.step(action)
            agent.replay_buffer_append(state, action, reward, next_state, done)

            state = next_state
            current_episode_reward += reward
            current_episode_frames += 1
            total_frames += 1

            if done:
                print(
                    f'Total Frames {total_frames} | Episode {total_episodes} | '
                    f'Time {time.time() - episode_start_time:.4f} | '
                    f'Total Time {time.time() - total_start_time:.4f}')
                episode_reward_moving_average.append(current_episode_reward)
                episode_loss_moving_average.append(current_episode_loss)
                episode_frames_moving_average.append(current_episode_frames)
                writer.add_scalar('Train/Episode Loss', current_episode_loss, total_episodes)
                writer.add_scalar('Train/Episode Reward', current_episode_reward, total_episodes)
                writer.add_scalar('Train/Episode Frames', current_episode_frames, total_episodes)
                writer.add_scalar(f'Train/Episode Reward Last {args.log_moving_average_num_episodes} Episodes/Mean',
                                  episode_reward_moving_average.mean(), total_episodes)
                writer.add_scalar(f'Train/Episode Loss Last {args.log_moving_average_num_episodes} Episodes/Mean',
                                  episode_loss_moving_average.mean(),
                                  total_episodes)
                writer.add_scalar(f'Train/Episode Frames Last {args.log_moving_average_num_episodes} Episodes/Mean',
                                  episode_frames_moving_average.mean(),
                                  total_episodes)
                writer.add_scalar(f'Train/Episode Reward Last {args.log_moving_average_num_episodes} Episodes/Median',
                                  episode_reward_moving_average.median(), total_episodes)
                writer.add_scalar(f'Train/Episode Loss Last {args.log_moving_average_num_episodes} Episodes/Median',
                                  episode_loss_moving_average.median(),
                                  total_episodes)
                writer.add_scalar(f'Train/Episode Frames Last {args.log_moving_average_num_episodes} Episodes/Median',
                                  episode_frames_moving_average.median(),
                                  total_episodes)
                writer.add_scalar(f'Train/Episode Reward Last {args.log_moving_average_num_episodes} Episodes/Variance',
                                  episode_reward_moving_average.variance(), total_episodes)
                writer.add_scalar(f'Train/Episode Loss Last {args.log_moving_average_num_episodes} Episodes/Variance',
                                  episode_loss_moving_average.variance(),
                                  total_episodes)
                writer.add_scalar(f'Train/Episode Frames Last {args.log_moving_average_num_episodes} Episodes/Variance',
                                  episode_frames_moving_average.variance(),
                                  total_episodes)
                writer.add_histogram(
                    f'Train/Episode Reward Last {args.log_moving_average_num_episodes} Episodes/Histogram',
                    np.array(episode_reward_moving_average.buffer), total_episodes)
                writer.add_histogram(
                    f'Train/Episode Loss Last {args.log_moving_average_num_episodes} Episodes/Histogram',
                    np.array(episode_loss_moving_average.buffer),
                    total_episodes)
                writer.add_histogram(
                    f'Train/Episode Frames Last {args.log_moving_average_num_episodes} Episodes/Histogram',
                    np.array(episode_frames_moving_average.buffer),
                    total_episodes)
                writer.add_scalar('Train/Epsilon Value', agent.agent_epsilon_greedy_current_epsilon,
                                  total_episodes)
                writer.add_scalar('Train/Replay Buffer Length', len(agent.replay_buffer), total_episodes)
                writer.add_scalar('Train/Total Frames', total_frames, total_episodes)

                if total_episodes - previous_evaluation_episodes > args.environment_evaluation_interval_episodes:
                    evaluation_reward_average = MovingAverage(num_to_average=args.environment_evaluation_episodes)
                    evaluation_frames_average = MovingAverage(num_to_average=args.environment_evaluation_episodes)

                    for _ in range(args.environment_evaluation_episodes):
                        evaluation_episode_reward = 0
                        evaluation_episode_frames = 0
                        state = evaluation_environment.reset()

                        while True:
                            action = agent.sample_action_evaluation(state)
                            state, reward, done, _ = evaluation_environment.step(action)
                            evaluation_episode_reward += reward
                            evaluation_episode_frames += 1

                            if done:
                                break

                        evaluation_reward_average.append(evaluation_episode_reward)
                        evaluation_frames_average.append(evaluation_episode_frames)

                    previous_evaluation_episodes = total_episodes
                    writer.add_scalar(
                        f'Evaluation/Episode Reward of {args.environment_evaluation_episodes} Episodes/Mean',
                        evaluation_reward_average.mean(), total_episodes)
                    writer.add_scalar(
                        f'Evaluation/Episode Frames of {args.environment_evaluation_episodes} Episodes/Mean',
                        evaluation_frames_average.mean(), total_episodes)
                    writer.add_scalar(
                        f'Evaluation/Episode Reward of {args.environment_evaluation_episodes} Episodes/Median',
                        evaluation_reward_average.median(), total_episodes)
                    writer.add_scalar(
                        f'Evaluation/Episode Frames of {args.environment_evaluation_episodes} Episodes/Median',
                        evaluation_frames_average.median(), total_episodes)
                    writer.add_scalar(
                        f'Evaluation/Episode Reward of {args.environment_evaluation_episodes} Episodes/Variance',
                        evaluation_reward_average.variance(), total_episodes)
                    writer.add_scalar(
                        f'Evaluation/Episode Frames of {args.environment_evaluation_episodes} Episodes/Variance',
                        evaluation_frames_average.variance(), total_episodes)
                    writer.add_histogram(
                        f'Evaluation/Episode Reward of {args.environment_evaluation_episodes} Episodes/Histogram',
                        np.array(evaluation_reward_average.buffer), total_episodes)
                    writer.add_histogram(
                        f'Evaluation/Episode Frames of {args.environment_evaluation_episodes} Episodes/Histogram',
                        np.array(evaluation_frames_average.buffer), total_episodes)

                if total_episodes - previous_checkpoint_episodes > args.environment_checkpoint_interval_episodes:
                    current_q_network_checkpoint_save_path = os.path.join(log_and_model_save_path,
                                                                          f'checkpoint_current_q_network-total_frames-{total_frames}-total_episodes-{total_episodes}.params')
                    th.save(agent.current_q_network.state_dict(), current_q_network_checkpoint_save_path)
                    old_q_network_checkpoint_save_path = os.path.join(log_and_model_save_path,
                                                                      f'checkpoint_old_q_network-total_frames-{total_frames}-total_episodes-{total_episodes}.params')
                    th.save(agent.old_q_network.state_dict(), old_q_network_checkpoint_save_path)
                    previous_checkpoint_episodes = total_episodes

                total_episodes += 1
                current_episode_loss = 0
                current_episode_reward = 0
                current_episode_frames = 0

                state = train_environment.reset()
                episode_start_time = time.time()

        loss = agent.train_agent(total_frames)
        current_episode_loss += loss

# print(
#     f'Episode {episode} | Frame {frame} | Total Frames {frames} | '
#     f'Reward {reward:.4f} | '
#     f'Avg Elapsed Time(s) per Frame {(time.time() - start_time) / frames:.4f} | '
#     f'Total Elapsed Time(s) {time.time() - start_time:.4f}')
# writer.add_scalar('Frame Reward', reward, frames)
# writer.add_scalar('Action Taken', action, frames)
# writer.add_scalar('Current Epsilon Value', agent.agent_epsilon_greedy_current_epsilon,
#                   frames)
# writer.add_image('Raw State', environment.render(mode='rgb_array'),
#                  frames, dataformats='HWC')
# writer.add_images('State', np.expand_dims(np.array(state), axis=1), frames)
# writer.add_images('Next State', np.expand_dims(np.array(next_state), axis=1), frames)
