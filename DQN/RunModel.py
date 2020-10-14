from DeepQNetwork import PredictDeepQNet, TargetDeepQNet
from Utils import MemoryReplay, test_model
import numpy as np 
import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, predict_model, target_model):
        self.predict_model = predict_model
        self.target_model = target_model

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch+1) % 6 == 0:
            self.target_model.set_weights(self.predict_model.get_weights())
            # self.target_model.save("./RL_MODEL")
            test_model(self.target_model)

class Model(object):
    def __init__(self):
        self.init_model()
        self.lambda_func = lambda i: np.array(list(map(lambda x: x[i], self.experience_replay)))

    def init_model(self):
        self.predict_model = PredictDeepQNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        self.target_model = TargetDeepQNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        self.predict_model.compile(optimizer="Adam", loss="MSE", metrics=["mse"])

    def custom_loss(self, y_actual, y_pred):
        max_num = tf.reduce_max(y_actual, axis=1, keepdims=True)
        mask = tf.greater_equal(y_actual, max_num)
        y_actual = tf.where(mask, y_actual, y_pred)
        return tf.keras.losses.MSE(y_actual, y_pred)

    def train(self):
        init_states_np, next_states_np, actions = self.lambda_func(0), self.lambda_func(3), self.lambda_func(1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                    patience=0, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        custom_callback = CustomCallback(self.predict_model, self.target_model)
        labels_data = self.get_labels(init_states_np, next_states_np, actions)
        self.predict_model.fit(init_states_np, labels_data, batch_size=4, shuffle=True, verbose=0, epochs=1, validation_split=0, callbacks=[])
        
    def get_labels(self, init_states_np, next_states_np, actions):
        gama = 0.9
        target_data = self.lambda_func(2).reshape(-1, 1) + gama * self.target_model.predict(next_states_np, verbose=0)
        target_data_max = np.max(target_data, axis=1)
        predict_data = self.predict_model.predict(init_states_np, verbose=0)
        for i, action in enumerate(actions):
            predict_data[i][action] = target_data_max[i]
        return predict_data

    def run(self):
        memory_replay = MemoryReplay(3000, e_greed=0.6, e_greed_decay=0.9)
        self.experience_replay = memory_replay.sample(1000)

        for i in range(100):
            print("EPOCHES: {}".format(i))
            self.train()
            if (i+1) % 4 == 0:
                self.target_model.set_weights(self.predict_model.get_weights())
                test_model(self.target_model)

            memory_replay.explore_env(100, self.target_model)
            self.experience_replay = memory_replay.sample(1000)
            for i in self.experience_replay:
                print(i)

if __name__ == "__main__":
    INPUT_DIM, OUTPUT_DIM = 2, 4
    model = Model()
    model.run()