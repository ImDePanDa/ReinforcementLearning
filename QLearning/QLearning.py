import numpy as np 
import random

"""
根据QLearning图片，自动寻找最优路径
"""

# Stage 1:
# 定义reward矩阵, 横纵坐标分别表示各个状态，譬如
# 坐标（0，0）为-1，表示不可做此状态转换
# 坐标（0，4）为0，表示从0状态到4状态，reward为0
R = np.array([
            [-1, -1, -1, -1, 0, -1],
            [-1, -1, -1, 0, -1, 100],
            [-1, -1, -1, 0, -1, -1],
            [-1, 0, 0, -1, 0, -1],
            [0, -1, -1, 0, -1, 100],
            [-1, 0, -1, -1, 0, 100]
            ])

# Stage 2:
# 定义Q矩阵，维度与R矩阵一致，初始化全为0，代表任何状态转换此时都没有reward
Q = np.zeros((R.shape[0], R.shape[1]))

# Satge 3:
# 定义Q矩阵更新方法，根据公式进行状态转换 Q_(s, a) = R_(s, a) + lambda * max(Q_(s', a'))
# 公式简单理解为 Q矩阵由s状态和a动作进行更新
def Q_Update(state, R, Q):
    """Q矩阵更新方法，输入：
        state   : 更新前agent状态（位置）
        R       : reward矩阵
        Q       : Q矩阵
    """
    # 学习率
    RATE = 0.8 

    actions = np.where(R[state, :] != -1)
    random_index = random.randint(0, len(actions[0])-1)
    action = actions[0][random_index]
    next_state = action
    next_actions = np.where(R[next_state, :] != -1)
    # 引入公式
    Q[state, action] = R[state, action] + RATE * max(map(lambda x: Q[next_state, x], next_actions[0]))
    if not end_condition(next_state):
        Q_Update(next_state, R, Q)

def end_condition(state):
    """一次训练终止条件判断
    """
    return state == 5

def train(epoches):
    """训练大脑，输入
        epoches: 训练次数
    """
    for i in range(epoches):
        init_state = random.randint(0, 5)
        Q_Update(init_state, R, Q)
        print("经过第{}次训练，Q矩阵为:".format(i))
        print(Q)

def test(init_state):
    """测试大脑，输入
        init_state: 初始位置 
    """
    print("{} ->".format(init_state), end=' ')
    actions = R[init_state, :]
    next_state = np.argmax(actions)
    if not end_condition(next_state):
        test(next_state)
    print(5)

if __name__ == "__main__":
    epoches = 10
    train(epoches)

    for i in range(50):
        init_state = random.randint(0, 5)
        test(init_state)