import gym
import gym_maze

def demo():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

def test():
    env = gym.make('maze-sample-5x5-v0')
    observation = env.reset()
    print(observation)
    print(env.observation_space.shape)
    action = env.action_space.sample()
    print(action)
    
    observation, reward, done, info = env.step(action)
    print(observation)
    print(reward)
    print(done)
    print(info)
    env.close()


if __name__ == "__main__":
    # demo()
    test()    