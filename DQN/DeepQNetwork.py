import tensorflow as tf
from tensorflow.keras import layers



class DeepQNet(object):
    def __init__(self):
        pass


class TimeEncoder(tf.keras.Model):
    def __init__(self):
        super(TimeEncoder, self).__init__()
        self.gauss = layers.GaussianNoise(1)
        self.dense1 = layers.Dense(32, activation=tf.nn.tanh, input_dim=88)
        self.dense2 = layers.Dense(88)

    def call(self, inputs, training=False):
        inputs = self.gauss(inputs) if training else inputs
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output