import time

import gym
import gym_maze
import numpy as np


# def create_experience_replay(experience_replay, num_next):
#     env = gym.make('maze-sample-5x5-v0')
#     observation = env.reset()
#     actions_num = env.action_space.n
#     end_count = 0 if not experience_replay else 10
#     iter_num = 10000 if num_next == 0 else num_next
#     for i in range(iter_num):
#         # env.render()
#         # time.sleep(1)
#         action = env.action_space.sample()
#         next_observation, reward, done, info = env.step(action)
#         end_count = end_count + 1 if reward != -0.004 else end_count
#         experience_replay.append((observation, action, reward, np.copy(next_observation)))
#         observation = np.copy(next_observation)
#     while end_count<100:
#         action = env.action_space.sample()
#         next_observation, reward, done, info = env.step(action)
#         end_count = end_count + 1 if reward != -0.004 else end_count
#         experience_replay.append((observation, action, reward, np.copy(next_observation)))
#         observation = np.copy(next_observation)
#     env.close()
#     return experience_replay, actions_num

def create_experience_replay(n):
    env = gym.make('maze-sample-5x5-v0')
    observation = env.reset()
    actions_num = env.action_space.n
    for i in range(n):
        # env.render()
        # time.sleep(1)
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        ret_tuple = (observation, action, reward, next_observation)
        observation = np.copy(next_observation)
        yield ret_tuple, actions_num

def test_model(model):
    env = gym.make('maze-sample-5x5-v0')
    observation = env.reset()
    for i in range(20):
        env.render()
        time.sleep(0.5)
        action = ['N', 'E', 'S', 'W'][model.predict(observation.reshape(-1, 2)).argmax()]
        print(action)
        next_observation, reward, done, info = env.step(action)
        observation = next_observation
    env.close()

if __name__ == "__main__":
    for i, v in enumerate(create_experience_replay(100)):
        print(v[0])
        if (i+1)%10: 
            break
    print(create_experience_replay(10).__next__()[0])