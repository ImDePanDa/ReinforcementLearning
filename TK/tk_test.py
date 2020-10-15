from tkinter import *
import numpy as np
import tkinter as tk

class MazePloter(object):
    def __init__(self, input_maze_np):
        self.maze_np = input_maze_np
        self.unit = 40
        self.col_margin = self.maze_np.shape[0]*self.unit*0.1
        self.maze_height = self.maze_np.shape[0]*self.unit
        self.cv_height =  self.maze_height+ 2*self.col_margin
        self.row_margin = self.maze_np.shape[1]*self.unit*0.1
        self.maze_width = self.maze_np.shape[1]*self.unit
        self.cv_width =  self.maze_width + 2*self.row_margin

    def run(self, cur_position):
        windows = Tk()
        windows.title("maze")
        cv = Canvas(windows, bg = 'white', height=self.cv_height, width=self.cv_width)

        left_top_point = (self.row_margin, self.col_margin)
        left_bottom_point = (self.row_margin, self.col_margin+self.maze_height)
        right_top_point = (self.row_margin+self.maze_width, self.col_margin)
        right_bottom_point = (self.row_margin+self.maze_width, self.col_margin+self.maze_height)

        cv.create_rectangle(left_top_point[0], left_top_point[1], right_bottom_point[0], right_bottom_point[1])

        for i in range(self.maze_np.shape[0]):
            cv.create_line(left_top_point[0], left_top_point[1]+i*self.unit, right_top_point[0], right_top_point[1]+i*self.unit)
        for i in range(self.maze_np.shape[1]):
            cv.create_line(left_top_point[0]+i*self.unit, left_top_point[1], left_bottom_point[0]+i*self.unit, left_bottom_point[1])

        bomb_pos_list, treasure_pos_list = [], []
        for i, row in enumerate(self.maze_np):
            for j, val in enumerate(row):
                if val==-1:
                    bomb_pos_list.append((i, j))
                elif val==1:
                    treasure_pos_list.append((i, j))

        for position in bomb_pos_list:
            cv.create_rectangle(self.row_margin+position[1]*self.unit, self.col_margin+position[0]*self.unit, \
                                self.row_margin+(position[1]+1)*self.unit, self.col_margin+(position[0]+1)*self.unit, \
                                fill='black')

        for position in treasure_pos_list:
            cv.create_rectangle(self.row_margin+position[1]*self.unit, self.col_margin+position[0]*self.unit, \
                                self.row_margin+(position[1]+1)*self.unit, self.col_margin+(position[0]+1)*self.unit, \
                                fill='blue')

        cv.create_oval(self.row_margin+cur_position[1]*self.unit, self.col_margin+cur_position[0]*self.unit, \
                            self.row_margin+(cur_position[1]+1)*self.unit, self.col_margin+(cur_position[0]+1)*self.unit, \
                            fill='red')

        cv.pack()
        windows.mainloop()

if __name__ == "__main__":
    maze_map = np.zeros([5, 6])

    treasure_position = [(2, 3)]
    for position in treasure_position:
        maze_map[position] = 1

    bomb_position = [(1, 1), (3, 2)]
    for position in bomb_position:
        maze_map[position] = -1
    print(maze_map)

    windows = MazePloter(maze_map)
    for i in range(3):
        windows.run((i, i))