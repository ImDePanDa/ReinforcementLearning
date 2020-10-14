from DeepQNetwork import PredictDeepQNet, TargetDeepQNet
from Utils import create_experience_replay, test_model
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
        self.experience_replay = []
        self.lambda_func = lambda i: np.array(list(map(lambda x: x[i], self.experience_replay)))

    def init_model(self):
        self.predict_model = PredictDeepQNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        self.target_model = TargetDeepQNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        self.predict_model.compile(optimizer="Adam", loss=self.custom_loss, metrics=["mse"])

    def custom_loss(self, y_actual, y_pred):
        max_num = tf.reduce_max(y_actual, axis=1, keepdims=True)
        mask = tf.greater_equal(y_actual, max_num)
        y_actual = tf.where(mask, y_actual, y_pred)
        return tf.keras.losses.MSE(y_actual, y_pred)

    def train(self, init_states_np, next_states_np):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                    patience=0, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        custom_callback = CustomCallback(self.predict_model, self.target_model)
        # print(init_states_np)
        # print(next_states_np)
        target_data = self.get_labels(next_states_np)
        # print(target_data)
        self.predict_model.fit(init_states_np, target_data, batch_size=4, shuffle=True, verbose=0, epochs=10, validation_split=0, callbacks = [custom_callback])
        
    def get_labels(self, predict_data):
        gama = 0.9
        return self.lambda_func(2).reshape(-1, 1) + gama * self.target_model.predict(predict_data, verbose=0)

    def run(self):
        creater_iteration = create_experience_replay(1000*100)
        for i in range(100):
            print("EPOCHES: {}".format(i))
            for index, observation_tuple in enumerate(creater_iteration):
                self.experience_replay.append(observation_tuple)
                if len(self.experience_replay) >= 1000: break
            self.train(self.lambda_func(0), self.lambda_func(3))
            self.experience_replay = self.experience_replay[100:]

if __name__ == "__main__":
    INPUT_DIM, OUTPUT_DIM = 2, 4
    model = Model()
    model.run()