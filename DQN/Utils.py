import random
import time

import gym
import gym_maze
import numpy as np
import pandas as pd


class MemoryReplay(object):
    def __init__(self, replay_size=1000, e_greed=0.6, e_greed_decay=0.9):
        self.replay_size = replay_size
        self.e_greed = e_greed
        self.e_greed_decay = e_greed_decay
        self.state_dim, self.actions_num = self.init_memory_replay()

    def init_memory_replay(self):
        self.memory_replay = [0] * self.replay_size

        env = gym.make('maze-sample-5x5-v0')
        observation = env.reset()
        state_dim, actions_num = observation.shape[0], env.action_space.n
        for i in range(self.replay_size):
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            reward = -1 if (next_observation == observation).all() else reward
            self.memory_replay[i] = (observation, action, reward, np.copy(next_observation))
            observation = np.copy(next_observation)
        
        self.env = env
        return state_dim, actions_num

    def explore_env(self, explore_num, model=None):
        self.memory_replay = self.memory_replay[explore_num:]
        for i in range(explore_num):
            observation = self.memory_replay[-1][0] if i == 0 else observation
            if model and np.random.random() >= self.e_greed:
                action = model.predict(self.memory_replay[-1][0].reshape(-1, 2)).argmax()
                action = int(action)
            else:
                action = self.env.action_space.sample()
            next_observation, reward, done, info = self.env.step(action)
            reward = -1 if (next_observation == observation).all() else reward
            self.memory_replay.append((observation, action, reward, np.copy(next_observation)))
            observation = np.copy(next_observation)
        self.e_greed *= self.e_greed_decay
        
    def sample(self, num, input_np=False):
        help_func = lambda i: np.array(list(map(lambda x: x[i], self.memory_replay)))
        init_np = help_func(0)
        for i in range(1, 4):
            next_np = help_func(i).reshape(-1, 1) if len(help_func(i).shape) == 1 else help_func(i)
            init_np = np.append(init_np, next_np, axis=1)
        init_np = np.append(init_np, input_np, axis=0) if not isinstance(input_np, bool) else init_np
        init_np = np.unique(init_np, axis=0)
        df = pd.DataFrame(init_np)
        return df.sample(num, replace=True).values

    def get_memory_replay(self):
        return self.memory_replay

    def get_state_dim(self):
        return self.state_dim

    def get_actions_num(self):
        return self.actions_num

def test_model(model, pro=0.7):
    # model = tf.keras.models.load_model("./RL_MODEL")
    env = gym.make('maze-sample-5x5-v0')
    observation = env.reset()
    for i in range(20):
        env.render()
        time.sleep(0.1)
        # print(model.predict(observation.reshape(-1, 2)))
        # action = ['N', 'E', 'S', 'W'][model.predict(observation.reshape(-1, 2)).argmax()] if input() == "m" else input()
        action = int(model.predict(observation.reshape(-1, 2)).argmax()) if np.random.random()<pro else env.action_space.sample()
        print(["N", "S", "E", "W"][action])
        next_observation, reward, done, info = env.step(action)
        observation = np.copy(next_observation)
    env.close()

if __name__ == "__main__":
    # test_model(0)
    a = MemoryReplay(1000)
    memory_replay = a.get_memory_replay()
    for i in memory_replay:
        print(i)
    print("("*30)
    a.explore_env(100)
    a.sample(1000)
