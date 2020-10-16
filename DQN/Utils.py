import random
import time

import gym
import numpy as np
import pandas as pd

from MazeEnv import Maze


class MemoryReplay(object):
    def __init__(self, replay_size=1000, e_greed=0.9, e_greed_decay=1):
        self.replay_size = replay_size
        self.total_replay_count = 0
        self.e_greed = e_greed
        self.e_greed_decay = e_greed_decay
        self.state_dim, self.actions_num = self.init_memory_replay()

    def init_memory_replay(self):
        env = Maze()
        # env = gym.make("CartPole-v0")
        observation = env.reset()
        state_dim, actions_num = observation.shape[0], env.action_space.n

        self.memory_replay = np.array([[0.] * (2*state_dim+2) for i in range(self.replay_size)])

        for i in range(self.replay_size):
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            
            self.memory_replay[i][:state_dim] = observation
            self.memory_replay[i][state_dim] = int(action)
            self.memory_replay[i][state_dim+1] = reward
            self.memory_replay[i][state_dim+2:] = next_observation
            
            observation = np.array(env.cur_position)
        
        self.env = env
        return state_dim, actions_num

    def explore_env(self, explore_num, model=None):
        for i in range(explore_num):
            observation = np.array(self.env.cur_position)
            index = self.total_replay_count % self.replay_size

            if model and np.random.random() >= self.e_greed:
                action = model.predict(self.memory_replay[-1][:self.state_dim].reshape(-1, 2)).argmax()
                action = int(action)
            else:
                action = self.env.action_space.sample()

            next_observation, reward, done, info = self.env.step(action)

            self.memory_replay[index][:self.state_dim] = observation
            self.memory_replay[index][self.state_dim] = int(action)
            self.memory_replay[index][self.state_dim+1] = reward
            self.memory_replay[index][self.state_dim+2:] = next_observation

            self.total_replay_count += 1

        self.e_greed *= self.e_greed_decay
        
    def sample(self, num):
        num = min(self.memory_replay.shape[0], num) 
        df = pd.DataFrame(self.memory_replay)
        return df.sample(num, replace=False).values

    def get_state_dim(self):
        return self.state_dim

    def get_actions_num(self):
        return self.actions_num


if __name__ == "__main__":
    agent = MemoryReplay(10)
    for v in agent.memory_replay:
        print(v)
    print()
    agent.explore_env(10)
    for v in agent.memory_replay:
        print(v)
    print()
    for v in agent.sample(10):
        print(v)
