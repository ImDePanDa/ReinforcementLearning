# coding=utf-8
import numpy as np
import pandas as pd
import json
import os 

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

    def step(self, action):
        """进入回合"""
        self.epoch_num += 1
        reward, done, info = 0, False, None
        print("Epoch: {}".format(self.epoch_num))
        
        # 选手一回合
        # action = self.get_action(self.player1)
        ret_action = action
        flag = self.play_chess(action, 1)
        ret_checkerboard = np.copy(self.checkerboard)
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
            reward = -1
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

    def plot_checkerboard(self):
        """棋盘绘制"""
        # return
        df = pd.DataFrame(self.checkerboard)

        mask = df.isin([0])
        df = df.where(~mask, other="-")
        mask = df.isin([1])
        df = df.where(~mask, other="o")
        mask = df.isin([2])
        df = df.where(~mask, other="x")
        
        print(df)
        print("-"*60)

def online_run(nepisode):
    """模型在线运行"""
    nrows, ncols = 5, 5
    env = Gobang(nrows, ncols, player1="auto", player2="manual")
    state_dict = read_from_json() if read_from_json() else {}
    state_val =  read_from_json("state_val") if os.path.exists("./state_val.json") else [0.]*len(state_dict)
    state_num = len(state_dict)
    change_flag = False
    gama = 0.9
    for episode in range(nepisode):
        print("Episode: {}".format(episode))
        episode_state_memory = []
        episode_action_memory = []
        for i in range(20): 
            # 人工博弈
            max_actions, max_index = -float("inf"), 0
            for index, action in enumerate(env.op_cord):
                flag = env.play_chess(action, 1)
                reward = 1 if flag == "win" else 0
                checkerboard = np.copy(env.checkerboard)
                env.checkerboard[action] = 0.
                for k, v in state_dict.items():
                    if v==checkerboard.tolist():
                        r = gama*state_val[int(k)]+reward
                        if r>max_actions:
                            max_actions = r
                            max_index = index
                        break
            action = env.op_cord[max_index]
            # 数据采集
            # action = env.action_space.sample()
            checkerboard, reward, done, action = env.step(action)
            checkerboard, action = checkerboard.tolist(), action[0]*nrows+action[1]
            # 保存状态
            add_flag = True
            for k, v in state_dict.items():
                if v==checkerboard:
                    episode_state_memory.append(int(k))
                    add_flag = False
                    break
            if add_flag:
                change_flag = True
                state_dict[state_num] = checkerboard
                episode_state_memory.append(state_num)
                state_num += 1
            episode_action_memory.append(action)
            if done:
                print("Game End!")
                env.reset()
                break
        # 加上最后reward
        episode_action_memory.append(reward)
        file_num = len(os.listdir("./memory/"))//2
        np.save("./memory/episode_checkerboard_memory_{}.npy".format(file_num), episode_state_memory)
        np.save("./memory/episode_action_memory_{}.npy".format(file_num), episode_action_memory)
    if change_flag: write_into_json(state_dict)

def write_into_json(state_dict, name="state_dict"):
    json_str = json.dumps(state_dict)
    with open('./{}.json'.format(name), 'w') as f:
        json.dump(json_str, f)

def read_from_json(name="state_dict"):
    if not os.path.exists('./{}.json'.format(name)):
        return None
    with open('./{}.json'.format(name), 'r') as f:
        json_str = json.load(f)
    state_dict = json.loads(json_str)
    return state_dict

def check_memory(num):
    episode_checkerboard_memory = np.load("./memory/episode_checkerboard_memory_{}.npy".format(num))
    episode_action_memory = np.load("./memory/episode_action_memory_{}.npy".format(num))
    # print(episode_checkerboard_memory)
    # print(episode_action_memory)
    return episode_checkerboard_memory, episode_action_memory

def train():
    state_dict = read_from_json()
    state_val =  read_from_json("state_val") if os.path.exists("./state_val.json") else [0.]*len(state_dict)
    if len(state_dict)>len(state_val):
        state_val += [0.]*(len(state_dict)-len(state_val))
    gama = 0.9
    diff, thre = 1, 1e-6
    while diff>thre:
        state_val_old = np.copy(np.array(state_val))
        nepisodes = len(os.listdir("./memory/"))//2
        for i in range(nepisodes):
            episode_state_memory, episode_action_memory = check_memory(i)
            for j in range(len(episode_state_memory)):
                if j == len(episode_state_memory)-1:
                    state_val[episode_state_memory[j]] = float(episode_action_memory[-1]) + 0
                else:
                    state_val[episode_state_memory[j]] =  gama*state_val[episode_state_memory[j+1]]
        diff = np.power((state_val_old-np.array(state_val)), 2).mean()
    print(state_val)
    write_into_json(state_val, "state_val")

if __name__ == "__main__":
    online_run(1)
    train()