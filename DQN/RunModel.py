import gym
import numpy as np
import tensorflow as tf

from DeepQNetwork import PredictDeepQNet, TargetDeepQNet
from MazeEnv import Maze
from Utils import MemoryReplay


class Model(object):
    def __init__(self):
        self.memory_replay = MemoryReplay(replay_size=100, e_greed=0.9, e_greed_decay=1)
        self.explore_num = 16
        self.train_samples_num = 32

        self.state_dim = self.memory_replay.get_state_dim()
        self.actions_num = self.memory_replay.get_actions_num()
        self.init_model()

    def init_model(self):
        self.predict_model = PredictDeepQNet(input_dim=self.state_dim, output_dim=self.actions_num)
        self.target_model = TargetDeepQNet(input_dim=self.state_dim, output_dim=self.actions_num)
        self.predict_model.compile(optimizer="Adam", loss="MSE", metrics=["mse"])

    def get_train_samples(self, samples_num):
        return self.memory_replay.sample(samples_num)

    def train(self):
        train_samples = self.get_train_samples(self.train_samples_num)
        init_states_np = train_samples[:, :self.state_dim]
        actions_np = train_samples[:, self.state_dim]
        rewards_np = train_samples[:, self.state_dim+1]
        next_states_np = train_samples[:, self.state_dim+2:]

        labels = self.get_labels(init_states_np, actions_np, rewards_np, next_states_np)
        self.predict_model.fit(init_states_np, labels, batch_size=1, shuffle=True, verbose=0, epochs=1, validation_split=0, callbacks=[])

    def test(self):
        env = Maze()
        obversation = env.reset()
        for i in range(16):
            env.render()
            # print(obversation)
            # print(self.target_model.predict(obversation.reshape(-1, self.state_dim)))
            action = int(self.target_model.predict(obversation.reshape(-1, self.state_dim)).argmax())
            print(["Left", "Down", "Right", "Up"] [action])
            next_observation, reward, done, info = env.step(action)
            if (next_observation == obversation).all():
                break
            obversation = next_observation 
 
    def get_labels(self, init_states_np, actions_np, rewards_np, next_states_np):
        gama = 0.9
        predict_data = self.predict_model.predict(init_states_np, verbose=0)
        target_data = self.target_model.predict(next_states_np, verbose=0)
        
        target_data_max = np.max(target_data, axis=1)
        for i, row in enumerate(predict_data):
            row[int(actions_np[i])] = (target_data_max[i]*gama + rewards_np[i])
        return predict_data

    def run(self):
        for i in range(2000):
            print("EPOCHES: {}".format(i))
            if i==0:
                self.target_model.set_weights(self.predict_model.get_weights())
            self.train()
            if (i+1) % 3 == 0:
                self.memory_replay.explore_env(self.explore_num, self.target_model)
            if (i+1) % 20 == 0:
                self.target_model.set_weights(self.predict_model.get_weights())
                self.test()


if __name__ == "__main__":
    model = Model()
    model.run()
