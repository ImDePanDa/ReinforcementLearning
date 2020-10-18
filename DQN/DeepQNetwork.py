import tensorflow as tf
from tensorflow.keras import layers


class PredictDeepQNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PredictDeepQNet, self).__init__()
        self.dense1 = layers.Dense(8, activation=tf.nn.elu, input_dim=input_dim, kernel_initializer=tf.constant_initializer(0.1))
        self.dense2 = layers.Dense(output_dim, activation=tf.nn.elu, kernel_initializer=tf.constant_initializer(0.1))

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

class TargetDeepQNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(TargetDeepQNet, self).__init__()
        self.dense1 = layers.Dense(8, activation=tf.nn.elu, input_dim=input_dim, kernel_initializer=tf.constant_initializer(0.1))
        self.dense2 = layers.Dense(output_dim, activation=tf.nn.elu, kernel_initializer=tf.constant_initializer(0.1))

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

if __name__ == "__main__":
    # INPUT_DIM 表示输入状态的维度
    # OUTPUT_DIM 可选动作的种类数
    pass
