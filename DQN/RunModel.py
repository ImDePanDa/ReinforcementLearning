import datetime
import os
from threading import Thread

import gym
import numpy as np
import tensorflow as tf

from DeepQNetwork import PredictDeepQNet, TargetDeepQNet
from MazeEnv import Maze
from Utils import MemoryReplay


class Model(object):
    def __init__(self):
        self.memory_replay = MemoryReplay(replay_size=1024, e_greed=0.95, e_greed_decay=0.99)
        self.explore_num = 8
        self.train_samples_num = 64
        self.train_iter_num = 0
        self.place_weight_num = 128

        self.state_dim = self.memory_replay.get_state_dim()
        self.actions_num = self.memory_replay.get_actions_num()
        self.init_model()

    def init_model(self):
        self.predict_model = PredictDeepQNet(input_dim=self.state_dim, output_dim=self.actions_num)
        self.target_model = TargetDeepQNet(input_dim=self.state_dim, output_dim=self.actions_num)
        optimizer =  tf.keras.optimizers.Adam(lr=0.001)
        self.predict_model.compile(optimizer=optimizer, loss="MSE", metrics=["mse"])

    def get_train_samples(self, samples_num):
        return self.memory_replay.sample(samples_num)

    def train(self):
        self.train_iter_num += 1
        if (self.train_iter_num) % (self.place_weight_num) == 0:
            self.target_model.set_weights(self.predict_model.get_weights())
            self.test()
            # t1 = Thread(target=self.test)
            # t1.setDaemon(True)
            # t1.start()

        train_samples = self.get_train_samples(self.train_samples_num)
        init_states_np = train_samples[:, :self.state_dim]
        actions_np = train_samples[:, self.state_dim]
        rewards_np = train_samples[:, self.state_dim+1]
        next_states_np = train_samples[:, self.state_dim+2:]

        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=1)
        labels = self.get_labels(init_states_np, actions_np, rewards_np, next_states_np)
        self.predict_model.fit(init_states_np, labels, batch_size=self.train_samples_num, shuffle=True, verbose=0, epochs=2, validation_split=0, callbacks=[earlystop_callback])

    def test(self):
        env = Maze()
        obversation = env.reset()

        for i in range(16):
            env.render()
            print("now position:", obversation)
            print("model output:", self.target_model.predict(obversation.reshape(-1, self.state_dim)))
            action = int(self.target_model.predict(obversation.reshape(-1, self.state_dim)).argmax())
            print(["Left", "Down", "Right", "Up"] [action])
            next_observation, reward, done, info = env.step(action)
            if (next_observation == obversation).all() or done:
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
        for i in range(3000):
            done = False
            print("Episode: {}".format(i))
            while not done:
                self.train()
                done = self.memory_replay.explore_env(self.explore_num, self.target_model)
                
if __name__ == "__main__":
    # log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))#在tensorboard可视化界面中会生成带时间标志的文件
    # tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    # tensorboard --logdir=E:\Algorithmn\ReinforcementLearning\DQN\logs
    # http://localhost:6006/#scalars

    model = Model()
    model.run()
