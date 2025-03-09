import matplotlib.pyplot as plt
import math

# start and goal position
sx = -5.0  # [m]
sy = -5.0  # [m]
gx = 50.0  # [m]
gy = 50.0  # [m]

# set init obstacle positions
ox, oy = [], []
for i in range(-10, 60):
    ox.append(float(i))
    oy.append(-10.0)
for i in range(-10, 60):
    ox.append(60.0)
    oy.append(float(i))
for i in range(-10, 61):
    ox.append(float(i))
    oy.append(60.0)
for i in range(-10, 61):
    ox.append(-10.0)
    oy.append(float(i))
for i in range(-10, 40):
    ox.append(20.0)
    oy.append(float(i))
for i in range(0, 40):
    ox.append(40.0)
    oy.append(60.0 - i)
start = (sx, sy)
goal = (gx, gy)
obs = (ox, oy)

class Map:
    def __init__(self, obstacle_list:tuple[list,list],
                        start:tuple[float,float],
                        goal:tuple[float,float]):

        self.obstacle_list = obstacle_list
        self.start = start
        self.goal = goal
        self.tarj = []

    def showmap(self, with_traj=False)->None:
        plt.plot(self.obstacle_list[0], self.obstacle_list[1], ".k")
        plt.plot(self.start[0], self.start[1], "og")
        plt.plot(self.goal[0], self.goal[1], "xb")

        plt.grid(True)
        plt.axis("equal")
        plt.show()

    def isObs(self, x:int, y:int)->bool:
        if (x, y) in zip(self.obstacle_list[0], self.obstacle_list[1]):
            return True
        else:
            return False


class Node:
    def __init__(self, x:int, y:int, cost:float, parent_idx:str):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_idx = parent_idx

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(
            self.cost) + "," + str(self.parent_idx)




# test the class
if __name__ == "__main__":
    map = Map(obs, start, goal)
    map.showmap()
    print(map.isObs(0, 0)) # expect True