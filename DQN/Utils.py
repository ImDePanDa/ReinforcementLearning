import time

import gym
import gym_maze
import numpy as np
import tensorflow as tf

def create_experience_replay(n):
    env = gym.make('maze-sample-5x5-v0')
    observation = env.reset()
    for i in range(n):
        # env.render()
        # time.sleep(1)
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        ret_tuple = (observation, action, reward, np.copy(next_observation))
        observation = np.copy(next_observation)
        yield ret_tuple

def test_model(model):
    # model = tf.keras.models.load_model("./RL_MODEL")
    env = gym.make('maze-sample-5x5-v0')
    observation = env.reset()
    for i in range(10):
        env.render()
        time.sleep(0.5)
        # print(model.predict(observation.reshape(-1, 2)))
        # action = ['N', 'E', 'S', 'W'][model.predict(observation.reshape(-1, 2)).argmax()] if input() == "m" else input()
        action = ['N', 'E', 'S', 'W'][model.predict(observation.reshape(-1, 2)).argmax()]
        print(action)
        next_observation, reward, done, info = env.step(action)
        observation = np.copy(next_observation)
    env.close()

if __name__ == "__main__":
    test_model(0)