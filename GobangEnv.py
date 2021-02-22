# coding=utf-8
import numpy as np
import pandas as pd
import json

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
    def __init__(self, nrows, ncols, player1, player2):
        self.nrows = nrows
        self.ncols = ncols
        self.epoch_num = 0
        self.build_checkerboard()
        self.op_cord = [(x, y) for x in range(nrows) for y in range(ncols)]
        self.action_space = ActionSpace(self.op_cord)
        self.player1 = player1
        self.player2 = player2

    def build_checkerboard(self):
        """棋盘构建方法，在初始化中调用"""
        self.checkerboard = np.zeros([self.nrows, self.ncols])    

    def reset(self):
        """初始化棋盘"""
        self.epoch_num = 0
        self.build_checkerboard()
        self.op_cord = [(x, y) for x in range(self.nrows) for y in range(self.ncols)]
        self.action_space = ActionSpace(self.op_cord)

    def step(self):
        """进入回合"""
        self.epoch_num += 1
        reward, done, info = 0, False, None
        print("Epoch: {}".format(self.epoch_num))
        
        # 选手一回合
        action = self.get_action(self.player1)
        ret_action, ret_checkerboard = action, np.copy(self.checkerboard)
        flag = self.play_chess(action, 1)
        print("选手一落子: {}.".format(action))
        self.plot_checkerboard()
        if flag == "win":
            reward = 1
            done = True
            return ret_checkerboard, reward, done, ret_action
        elif flag == "end":
            done = True
            return ret_checkerboard, reward, done, ret_action

        # 选手二回合
        action = self.get_action(self.player2)
        flag = self.play_chess(action, 2)
        print("选手二落子: {}.".format(action))
        self.plot_checkerboard()
        if flag == "win":
            reward = 1
            done = True
            return ret_checkerboard, reward, done, ret_action
        elif flag == "end":
            done = True
            return ret_checkerboard, reward, done, ret_action

        return ret_checkerboard, reward, done, ret_action

    def play_chess(self, action, player):
        """对弈"""
        self.checkerboard[action] = player
        if self.is_win(action, player):
            return "win"
        elif len(self.op_cord) == 0:
            return "end"
        else:
            return "continue"

    def get_action(self, flag):
        """得到落子位置"""
        action = (0, 0)
        if flag == "auto":
            action = self.action_space.sample()
        elif flag == "manual":
            is_reasonable = False
            while not is_reasonable:
                cord = input("你的回合, 请输入落子坐标(直接以空格分隔):\n")
                if cord == "q": exit("游戏退出！")
                try:
                    action = tuple(map(int, cord.split(" ")))
                    while self.checkerboard[action] != 0:
                        action = (map(int, input("落子重复, 请输入落子坐标(直接以空格分隔):\n").split(" ")))
                    self.op_cord.remove(action)
                    is_reasonable = True
                except:
                    print("输入错误, 请重新输入落子坐标！")
        else:
            exit("输入值错误: 'auto' or 'manual'")
        return action

    def render(self):
        """开启迷宫输出"""
        self.plot = True

    def is_win(self, action, player):
        """判断是否赢棋"""
        threshold = 3
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
        if count>=threshold: 
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
        if count>=threshold: 
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
        if count>=threshold: 
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
        if count>=threshold: 
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

        mask = df.isin([0])
        df = df.where(~mask, other="-")
        mask = df.isin([1])
        df = df.where(~mask, other="o")
        mask = df.isin([2])
        df = df.where(~mask, other="x")
        
        print(df)
        print("-"*60)

def offline_run():
    """模型离线运行"""
    env = Gobang(5, 5, player1="auto", player2="auto")
    for episode in range(3):
        for i in range(20): 
            checkerboard, reward, done, action = env.step()
            episode_checkerboard_memory = checkerboard if i==0 else np.append(episode_checkerboard_memory, checkerboard, axis=0)
            episode_action_memory = list(action) if i==0 else episode_action_memory + list(action)
            if done:
                print("Game End!")
                env.reset()
                break
        # 最后一位是reward
        episode_action_memory += [reward]
        np.save("./memory/episode_checkerboard_memory_{}.npy".format(episode), episode_checkerboard_memory)
        np.save("./memory/episode_action_memory_{}.npy".format(episode), episode_action_memory)

def online_run():
    """模型在线运行"""
    env = Gobang(5, 5, player1="auto", player2="auto")
    state_dict, state_num = {}, 0
    for episode in range(3):
        for i in range(20): 
            checkerboard, reward, done, action = env.step()
            # 保存状态
            if i==0:
                state_dict[state_num] = checkerboard.tolist()
                state_num += 1
            else:
                add_flag = True
                for k, v in state_dict.items():
                    if v==checkerboard.tolist():
                        add_flag = False
                if add_flag:
                    state_dict[state_num] = checkerboard.tolist()
                    state_num += 1
            if done:
                print("Game End!")
                env.reset()
                break
    json_str = json.dumps(state_dict)
    with open('./test.json', 'w') as f:
        json.dump(json_str, f)
    # with open('./test.json', 'r') as f:
    #     json_str = json.load(f)
    # data = json.loads(json_str)
    # print(data["0"])

def check_memory():
    episode_checkerboard_memory = np.load("./memory/episode_checkerboard_memory_1.npy")
    episode_action_memory = np.load("./memory/episode_action_memory_1.npy")
    print(episode_checkerboard_memory)
    print(episode_action_memory)

if __name__ == "__main__":
    online_run()
    # check_memory()
