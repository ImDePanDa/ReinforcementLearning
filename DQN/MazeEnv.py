import numpy as np
import pandas as pd


class ActionSpace(object):
    def __init__(self, actions_num):
        self.n = actions_num

    def sample(self):
        return np.random.randint(0, self.n)

class Maze(object):
    def __init__(self):
        self.maze_map = self.build_maze()
        self.action_space = ActionSpace(4)
        self.cur_position = (0, 0)
        self.plot = False

    def build_maze(self):
        maze_map = np.zeros([5, 5])

        treasure_position = [(2, 3)]
        for position in treasure_position:
            maze_map[position] = 1

        bomb_position = [(3, 2)]
        for position in bomb_position:
            maze_map[position] = -1

        return maze_map

    def reset(self):
        self.cur_position = (0, 0)
        return np.array(self.cur_position)

    def step(self, action):
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
        reward = -1/(self.maze_map.shape[0]*self.maze_map.shape[1])
        if self.maze_map[next_position] == -1:
            reward = -1
            done = True
        elif self.maze_map[next_position] == 1:
            reward = 1
            done = True

        self.cur_position = next_position

        if self.plot: 
            self.plot_maze()

        return np.array(self.cur_position), reward, done, info

    def render(self):
        self.plot = True

    def plot_maze(self):
        df = pd.DataFrame(self.maze_map)
        if df.iloc[self.cur_position] == -1:
            print("GET BOMB")
        elif df.iloc[self.cur_position] == 1:
            print("GET TREASURE")
        df.iloc[self.cur_position] = 'X'
        print(df)
        print("-"*60)


if __name__ == "__main__":
    env = Maze()
    obversation = env.reset()
    for i in range(10):
        env.render()
        action = env.action_space.sample()
        print(["Left", "Down", "Right", "Up"] [action])
        env.step(action)

