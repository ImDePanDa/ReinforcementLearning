import numpy as np 
import random

class QLearning(object):
    """根据QLearning图片，自动寻找最优路径
    """

    def __init__(self, R, lr):
        """初始化方法
        """
        self.R = R
        self.Q = np.zeros((R.shape[0], R.shape[1]))
        self.lr = lr
        self.END_STATE = 5
    
    def Q_update(self, state):
        """Q矩阵更新方法
        """
        actions = np.where(self.R[state, :] != -1)
        random_index = random.randint(0, len(actions[0])-1)
        action = actions[0][random_index]
        next_state = action
        next_actions = np.where(self.R[next_state, :] != -1)
        # 引入公式, 进行状态转换 Q_(s, a) = Q_(s, a) + lr * ( R_(s, a) + gama * max(Q_(s', a')) -  Q_(s, a) )
        self.Q[state, action] += self.lr * (self.R[state, action] + 0.8 * max(map(lambda x: self.Q[next_state, x], next_actions[0])) - self.Q[state, action])
        if not self.end_condition(next_state):
            self.Q_update(next_state)

    def end_condition(self, state):
        """一次训练终止条件
        """
        return state == self.END_STATE

    def train(self, epoches):
        """大脑训练
        """
        print("In Train:")
        for i in range(epoches):
            init_state = random.randint(0, 5)
            self.Q_update(init_state)
            print("Train Epoch: {}，Q: \n{}".format(i, self.Q))

    def test(self, init_state):
        """测试大脑
        """
        print("{} ->".format(init_state), end=' ')
        actions = self.R[init_state, :]
        next_state = np.argmax(actions)
        if not self.end_condition(next_state):
            self.test(next_state)
        else:
            print(next_state)


if __name__ == "__main__":
    np.set_printoptions(precision=3, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None)
    R = np.array([
                [-1, -1, -1, -1, 0, -1],
                [-1, -1, -1, 0, -1, 100],
                [-1, -1, -1, 0, -1, -1],
                [-1, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, -1, 100],
                [-1, 0, -1, -1, 0, 100]
                ])
    q_learning = QLearning(R=R, lr=0.8)
    q_learning.train(100)
    q_learning.test(2)