# Deep Reinforcement Learning

## Overview
Reinforcement learning (RL) is concerned with enabling an agent acting in an
environment to maximize the cumulative reward received by the agent from the aforementioned
environment.
To formalize RL, we utilize the framework of Markov Decision Processes (MDPs). Reinforcement
learning for MDPs is concerned with enabling an agent acting in an stochastic Markovian
environment to maximize the expected future reward received by the agent from the aforementioned environment.


In many applications of RL for MDPs, the transition probabilities and/or rewards of the MDP are
unknown. Alternatively, we may wish to consider model-free RL. To deal with these scenarios, we
consider directly approximating the Q-function from data by running a simulator to collect sample
transitions (s, a, r, s′) and then subsequently approximating the transition probabilities of the MDP
using these samples, then subsequently extracting the optimal policy from the learnt Q-function.
Experimental results demonstrate that our implementation results in agents with comparable performance to results reported in literature on Breakout and Pong. We also demonstrate that our implementation enables an agent
to also play Enduro, River Raid, and Space Invaders without game-specific hyperparameter tuning.


## Results
![](https://github.com/bingzhewei/DRL/blob/main/1.png)

In this evaluation episode, the agent achieved a score of 361 points. At around frame 259, the agent
almost fails to catch the ball and hence the estimated value drops abruptly.

![](https://github.com/bingzhewei/DRL/blob/main/7.png)

At around frame 440, the agent has successfully dug a tunnel through the bricks and bounced the
ball through the tunnel, so the estimated value rises dramatically.
![](https://github.com/bingzhewei/DRL/blob/main/8.png)

## Architecture

The overall code architecture is as shown below. We implemented DQN in
Python using OpenAI Gym and PyTorch. While we used PyTorch to implement training and
inference of the Q-function value approximation neural network, we utilized the Atari 2600 simulator
as implemented by OpenAI Gym as the environment simulator. Each frame preprocessing and
environment modification transformation was implemented using the gym.Wrapper API of OpenAI
Gym to enable modularity and ease of testing different combinations of frame preprocessing and
environment modification transformations. Using TensorboardX, we implemented a logging
system to enable visualization of training progress.

![](https://github.com/bingzhewei/DRL/blob/main/2.png)
![](https://github.com/bingzhewei/DRL/blob/main/3.png)
![](https://github.com/bingzhewei/DRL/blob/main/4.png)
