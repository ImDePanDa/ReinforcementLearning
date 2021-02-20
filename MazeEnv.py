# coding=utf-8
import numpy as np
import pandas as pd


class ActionSpace(object):
    """动作空间类
    """
    def __init__(self, actions_num):
        self.n = actions_num

    def sample(self):
        return np.random.randint(0, self.n)

class Maze(object):
    """迷宫类
    """
    def __init__(self):
        self.maze_map = self.build_maze()
        self.action_space = ActionSpace(4)
        self.cur_position = (0, 0)
        self.plot = False

    def build_maze(self):
        """迷宫构建方法，在初始化中调用"""
        maze_map = np.zeros([10, 10])

        treasure_position = [(7, 8)]
        for position in treasure_position:
            maze_map[position] = 1

        bomb_position = [(0, 6), (3, 9), (6, 7)]
        for position in bomb_position:
            maze_map[position] = -1

        return maze_map

    def reset(self):
        """当前位置初始化"""
        self.cur_position = (0, 0)
        return np.array(self.cur_position)

    def step(self, action):
        """进行探索
        输入: 动作
        """
        if action == 0:
            next_position = (self.cur_position[0], max(self.cur_position[1]-1, 0))
        elif action == 1:
            next_position = (min(self.cur_position[0]+1, self.maze_map.shape[1]-1), self.cur_position[1])
        elif action == 2:
            next_position = (self.cur_position[0], min(self.cur_position[1]+1, self.maze_map.shape[0]-1))
        elif action == 3:
            next_position = (max(self.cur_position[0]-1, 0), self.cur_position[1])
        else:
            print("ERROR: ACTION '{}' is not allowed!".format(action))
            exit(0)

        done, info = False, None
        reward = 0
        if self.maze_map[next_position] == -1:
            reward = -1
            done = True
        elif self.maze_map[next_position] == 1:
            reward = 1
            done = True
        elif self.cur_position == next_position:
            reward = -0.5
            done = True

        self.cur_position = next_position

        if self.plot: 
            self.plot_maze()

        return self.cur_position, reward, done, info

    def render(self):
        """开启迷宫输出"""
        self.plot = True

    def plot_maze(self):
        """迷宫绘制"""
        df = pd.DataFrame(self.maze_map)
        if df.iloc[self.cur_position] == -1:
            print("GET BOMB")
        elif df.iloc[self.cur_position] == 1:
            print("GET TREASURE")
        df.iloc[self.cur_position] = 'X'
        print(df)
        print("-"*60)

def run():
    """迷宫运行测试"""
    env = Maze()
    obversation = env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        print(["Left", "Down", "Right", "Up"][action])
        env.step(action)

def train():
    """进行策略学习"""
    
    # 初始化训练参数
    actions = range(4)
    env = Maze()
    s_a_values_matrix = np.ones_like(env.maze_map, dtype=np.float32)
    s_actions_matrix = np.zeros_like(env.maze_map)
    gama = 0.9
    threshold = 1e-5
    epoches = 100

    for epoch in range(epoches):
        # 保存与原始状态值
        s_a_values_matrix_copy = np.copy(s_a_values_matrix)
        # 遍历所有状态
        for i in range(s_a_values_matrix.shape[0]):
            for j in range(s_a_values_matrix.shape[1]):
                # 遍历所有动作
                action_values = np.array([0]*4, dtype=np.float32)
                for action in actions:
                    env.cur_position = (i, j)
                    cur_position, reward, done, info = env.step(action)
                    action_values[action] = reward + gama*s_a_values_matrix[cur_position]

                # 逐个状态更新价值
                s_a_values_matrix[i, j] = action_values.max()
                s_actions_matrix[i, j] = action_values.argmax()
        
        diff = np.power((s_a_values_matrix_copy - s_a_values_matrix), 2).mean()

        visual_s_actions_matrix = pd.DataFrame(s_actions_matrix)
        mask = visual_s_actions_matrix.isin([0])
        visual_s_actions_matrix = visual_s_actions_matrix.where(~mask, other="L")
        mask = visual_s_actions_matrix.isin([1])
        visual_s_actions_matrix = visual_s_actions_matrix.where(~mask, other="D")
        mask = visual_s_actions_matrix.isin([2])
        visual_s_actions_matrix = visual_s_actions_matrix.where(~mask, other="R")
        mask = visual_s_actions_matrix.isin([3])
        visual_s_actions_matrix = visual_s_actions_matrix.where(~mask, other="U")

        print("Epoch: {} | Diff: {:.6f} \nmaze: \n{} \ns_actions_matrix: \n{} \ns_value_matrix: \n{}.".format(\
            epoch, diff, "env.maze_map", visual_s_actions_matrix, "s_a_values_matrix"))
        if diff<=threshold: break
        

if __name__ == "__main__":
    # run()
    train()




