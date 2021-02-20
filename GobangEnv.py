# coding=utf-8
import numpy as np
import pandas as pd

class ActionSpace(object):
    """动作空间类
    """
    def __init__(self, op_cord):
        self.op_cord = op_cord

    def sample(self):
        choose_cord_index = np.random.randint(0, len(self.op_cord))
        choose_cord = self.op_cord.pop(choose_cord_index)
        return choose_cord

class Gobang(object):
    """五子棋环境
    """
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.epoch_num = 0
        self.build_checkerboard()
        self.op_cord = [(x, y) for x in range(nrows) for y in range(ncols)]
        self.action_space = ActionSpace(self.op_cord)

    def build_checkerboard(self):
        """棋盘构建方法，在初始化中调用"""
        self.checkerboard = np.zeros([self.nrows, self.ncols])    

    def reset(self):
        """初始化棋盘"""
        self.epoch_num = 0
        self.build_checkerboard()
        self.op_cord = [(x, y) for x in range(self.nrows) for y in range(self.ncols)]

    def step(self, action):
        """对弈"""
        self.epoch_num += 1
        reward, done, info = 0, False, None
        print("Epoch: {}".format(self.epoch_num))
        # 我方回合
        self.checkerboard[action] = 1
        if self.is_win(action, 1):
            reward = 1
            done = True
            print("选手一获胜！")
        print("选手一落子: {}.".format(action))
        self.plot_checkerboard()
        
        # 对方回合
        self.turn_end_flag = True
        self.code_end_flag = False
        while self.turn_end_flag:
            try:
                action = self.opponent_turn()
            except:
                print("坐标输入有误，注意格式")
            if self.code_end_flag: 
                print("程序结束，正在退出")
                exit(0)
        if self.is_win(action, 2):
            reward = -1
            done = True
            print("选手二获胜！")
        self.plot_checkerboard()

        return self.checkerboard, reward, done, info

    def render(self):
        """开启迷宫输出"""
        self.plot = True

    def is_win(self, action, player):
        """判断是否赢棋"""
        # 水平判断
        count = 1
        i, j = action
        while i-1>=0 and self.checkerboard[i-1, j]==player:
            count += 1
            i -= 1
        i, j = action
        while i+1<self.ncols and self.checkerboard[i+1, j]==player:
            count += 1
            i += 1
        if count>=5 : 
            return True
        # 垂直判断
        count = 1
        i, j = action
        while j-1>=0 and self.checkerboard[i, j-1]==player:
            count += 1
            j -= 1
        i, j = action
        while j+1<self.ncols and self.checkerboard[i, j+1]==player:
            count += 1
            j += 1
        if count>=5 : 
            return True
        # 左斜判断
        count = 1
        i, j = action
        while j-1>=0 and i-1>=0 and self.checkerboard[i-1, j-1]==player:
            count += 1
            i -= 1
            j -= 1
        i, j = action
        while j+1<self.ncols and i+1<self.nrows and self.checkerboard[i+1, j+1]==player:
            count += 1
            i += 1
            j += 1
        if count>=5 : 
            return True        
        # 右斜判断
        count = 1
        i, j = action
        while j-1>=0 and i+1<self.nrows and self.checkerboard[i+1, j-1]==player:
            count += 1
            i += 1
            j -= 1
        i, j = action
        while j+1<self.ncols and i-1>=0 and self.checkerboard[i-1, j+1]==player:
            count += 1
            i -= 1
            j += 1
        if count>=5 : 
            return True 
        return False

    def opponent_turn(self):
        """对方回合"""
        cord = input("你的回合, 请输入落子坐标(直接以空格分隔):\n").split(" ")
        if cord[0] == "q": self.code_end_flag=True
        action = tuple(map(int, cord))
        while self.checkerboard[action] != 0:
            action = (map(int, input("落子重复, 请输入落子坐标(直接以空格分隔):\n").split(" ")))
        self.turn_end_flag = False
        self.checkerboard[action] = 2
        self.op_cord.remove(action)
        print("选手二落子: {}.".format(action))
        return action

    def plot_checkerboard(self):
        """棋盘绘制"""
        df = pd.DataFrame(self.checkerboard)

        mask = df.isin([1])
        df = df.where(~mask, other="+")
        mask = df.isin([2])
        df = df.where(~mask, other="x")
        
        print(df)
        print("-"*60)

if __name__ == "__main__":
    env = Gobang(10, 10)
    for i in range(1000):
        checkerboard, reward, done, info = env.step(env.action_space.sample())
        if done:
            print("Game End!")
            env.reset()