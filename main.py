import numpy as np
import matplotlib.pyplot as plt
import random as rng
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import convolve2d

class Gaussian_point():
    def __init__(self, amp, x, xs, y, ys, rot):
        self.amp = amp
        self.x = x
        self.xs = xs
        self.y = y
        self.ys = ys
        self.rot = rot

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def diffuse(map, scale_fac):
    kernel=np.array([[0, scale_fac, 0],
                  [scale_fac, 1-4*scale_fac, scale_fac],
                  [0, scale_fac, 0]], dtype='float64')
    diffused=convolve2d(map,kernel,mode='same', boundary='symm')
    return diffused

#test = np.zeros((4, 4))
#test[1, 1] = 1
#test[1, 0] = 1
#test[0, 0] = 2

#print(test)

#ret = diffuse(test, 0.05)
#print(ret)

pi = math.pi
x_size = 100
y_size = 100
point_list = []

add_min = 0
add_max = 10
add_times = 100000

point_list.append(Gaussian_point(0.20, 0.45, 0.35, 0.45, 0.30, 3*pi/16))
point_list.append(Gaussian_point(0.15, 0.45, 0.35, 0.55, 0.40, 0*pi/16))
point_list.append(Gaussian_point(0.10, 0.45, 0.10, 0.45, 0.25, pi/8))
point_list.append(Gaussian_point(0.08, 0.70, 0.20, 0.35, 0.15, 7*pi/16))
point_list.append(Gaussian_point(0.05, 0.80, 0.20, 0.80, 0.25, 5*pi/16))
point_list.append(Gaussian_point(0.05, 0.20, 0.25, 0.25, 0.15, 1*pi/32))
point_list.append(Gaussian_point(0.05, 0.70, 0.15, 0.80, 0.10, 5*pi/32))
point_list.append(Gaussian_point(0.02, 0.80, 0.10, 0.40, 0.05, 0))

amp_tot = 0
for point in point_list:
    amp_tot += point.amp

rock = np.zeros((x_size, y_size))

for i in range(add_times):
    select = rng.uniform(0, amp_tot)
    cum_amp = 0
    for point in point_list:
        cum_amp += point.amp
        #print(str(cum_amp) + ' ' + str(select))
        if select < cum_amp:
            break
    x = -1
    y = -1
    while (x < 0 or x >= x_size or y < 0 or y >= x_size):
        x = rng.gauss(point.x, point.xs)
        y = rng.gauss(point.y, point.ys)
        x, y = rotate((point.x, point.y), (x, y), point.rot)
        x = round(x * x_size)
        y = round(y * y_size)
    rock[x, y] += rng.random() * (add_max - add_min)

X, Y = np.meshgrid(range(x_size), range(y_size))

plt.imshow(rock)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, rock, cmap='jet', edgecolor="none")
plt.show()

for i in range(10):
    rock = diffuse(rock, 0.10)
plt.imshow(rock)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, rock, cmap='jet', edgecolor="none")
plt.show()
