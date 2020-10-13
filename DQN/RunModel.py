from DeepQNetwork import PredictDeepQNet, TargetDeepQNet
from Utils import create_experience_replay, test_model
import numpy as np 
import tensorflow as tf

class Model(object):
    def __init__(self):
        self.init_data()
        self.init_model()

    def init_data(self):
        self.experience_replay, self.actions_num = create_experience_replay(experience_replay=[], num_next=0)
        self.lambda_func = lambda i: np.array(list(map(lambda x: x[i], self.experience_replay)))

    def prepare_data(self):
        self.experience_replay, self.actions_num = create_experience_replay([], 0)

    def init_model(self):
        self.predict_model = PredictDeepQNet(input_dim=self.lambda_func(0).shape[1], output_dim=self.actions_num)
        self.target_model = TargetDeepQNet(input_dim=self.lambda_func(0).shape[1], output_dim=self.actions_num)
        self.predict_model.compile(optimizer="SGD", loss=self.custom_loss, metrics=["mse"])

    def custom_loss(self, y_actual, y_pred):
        max_num = tf.reduce_max(y_actual, axis=1, keepdims=True)
        mask = tf.greater_equal(y_actual, max_num)
        y_actual = tf.where(mask, y_actual, y_pred)
        return tf.keras.losses.MSE(y_actual, y_pred)

    def train(self, init_states_np, next_states_np):
        early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                    patience=0, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        target_data = self.get_labels(next_states_np)
        self.predict_model.fit(init_states_np, target_data, batch_size=256, shuffle=True, verbose=0, epochs=80, validation_split=0.2, callbacks = [early_stopping])
        
    def get_labels(self, predict_data):
        gama = 0.9
        return self.lambda_func(2).reshape(-1, 1) + gama * self.target_model.predict(predict_data, verbose=0)

    def run(self):
        for i in range(100):
            print("EPOCHES: {}".format(i))
            self.train(self.lambda_func(0), self.lambda_func(3))
            if (i+1)%2 == 0:
                self.target_model.set_weights(self.predict_model.get_weights())
                self.prepare_data()
            if (i+1)%10 == 0:
                self.target_model.save("./RL_MODEL")
                test_model(self.target_model)
        # model = tf.keras.models.load_model(modelpath)

if __name__ == "__main__":
    model = Model()
    model.run()


