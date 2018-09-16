import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
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
    diffused=convolve2d(map,kernel,mode='same', boundary='wrap')
    return diffused

def flow_to_dir(map, direction, scale_fac):
    if direction == 'up':
        kernel=np.array([[0, 0, 0],
                      [0, scale_fac, 0],
                      [0, -scale_fac, 0]], dtype='float64')
    elif direction == 'down':
        kernel=np.array([[0, -scale_fac, 0],
                      [0, scale_fac, 0],
                      [0, 0, 0]], dtype='float64')
    elif direction == 'left':
        kernel=np.array([[0, 0, 0],
                      [0, scale_fac, -scale_fac],
                      [0, 0, 0]], dtype='float64')
    elif direction == 'right':
        kernel=np.array([[0, 0, 0],
                      [-scale_fac, scale_fac, 0],
                      [0, 0, 0]], dtype='float64')
    else:
        print('invalid direction')
        exit()
    flow=convolve2d(map,kernel,mode='same', boundary='wrap')
    return flow.clip(min = 0)

def flow_water(rock_map, water_map, scale_fac):
    sum = rock_map + water_map
    u = flow_to_dir(sum, 'up', scale_fac)
    d = flow_to_dir(sum, 'down', scale_fac)
    l = flow_to_dir(sum, 'left', scale_fac)
    r = flow_to_dir(sum, 'right', scale_fac)
    flow_sum = u + d + l + r
    fact = np.divide(water_map, flow_sum, out=np.zeros_like(water_map), where=flow_sum != 0)
    fact = fact.clip(max = 1)
    u = u * fact
    d = d * fact
    l = l * fact
    r = r * fact
    rem_water = u + d + l + r
    add_water = np.roll(u, -1, axis=0) + np.roll(d, 1, axis=0) + np.roll(l, -1, axis=1) + np.roll(r, 1, axis=1)
    water_map = water_map - rem_water + add_water
    water_map = water_map.clip(min = 0)
    return rock_map, water_map

def evap(rock_map, water_map, scale_fac):
    # sum_water = sum(sum(water_map))
    evap = water_map * scale_fac
    evap_sum = sum(sum(evap))
    x, y = water_map.shape
    rain_per = evap_sum / (x * y)
    water_map = water_map - evap + rain_per
    return rock_map, water_map

def plot_4(rock, water, X, Y, minc, maxc, i):
    fig = plt.figure(i)
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, rock, cmap='jet', edgecolor="none")
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, water, cmap='jet', edgecolor="none")
    ax = fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, rock + water, cmap='jet', edgecolor="none")
    ax = fig.add_subplot(224, projection='3d')
    norm = matplotlib.colors.Normalize(minc, maxc)
    m = plt.cm.ScalarMappable(norm=norm, cmap='terrain')
    m.set_array([])
    cmap = m.to_rgba(-water)
    ax.plot_surface(X, Y, rock, facecolors=cmap, vmin=minc, vmax=maxc, edgecolor="none")
    ax.plot_surface(X, Y, rock + water, facecolors=cmap, vmin=minc, vmax=maxc, edgecolor="none")
    ax.set_zlim(zmin = 0, zmax = 200)
    plt.show()

def plot_1(rock, water, X, Y, minc, maxc, i):
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    norm = matplotlib.colors.Normalize(minc, maxc)
    m = plt.cm.ScalarMappable(norm=norm, cmap='terrain')
    m.set_array([])
    cmap = m.to_rgba(-water)
    ax.plot_surface(X, Y, rock, facecolors=cmap, vmin=minc, vmax=maxc, edgecolor="none")
    ax.plot_surface(X, Y, rock + water, facecolors=cmap, vmin=minc, vmax=maxc, edgecolor="none")
    ax.set_zlim(zmin = 0, zmax = 200)
    plt.show()

def test_func():
    test = np.zeros((4, 4))
    test_rock = np.zeros((4, 4))
    test_water = np.zeros((4, 4))
    test_rock[1, 1] = 1
    test_rock[1, 2] = 1
    test_water[1, 1] = 1
    test_water[3, 3] = 1
    test = test_rock + test_water

    print('test:')
    print(test)

    up = flow_to_dir(test, 'up', 0.25)
    print('up:')
    print(up)
    down = flow_to_dir(test, 'down', 0.25)
    print('down:')
    print(down)
    left = flow_to_dir(test, 'left', 0.25)
    print('left:')
    print(left)
    right = flow_to_dir(test, 'right', 0.25)
    print('right:')
    print(right)

    print()
    print()
    print()

    # test_rock, test_water = flow_water(test_rock, test_water, 0.25)
    test_rock, test_water = evap(test_rock, test_water, 0.10)

    print('test_rock')
    print(test_rock)
    print('test_water')
    print(test_water)

    exit()

#test_func()

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
point_list.append(Gaussian_point(0.02, 0.80, 0.10, 0.60, 0.05, 5*pi/8))

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

for i in range(15):
    rock = diffuse(rock, 0.10)
plt.imshow(rock)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, rock, cmap='jet', edgecolor="none")
plt.show()

mean_rock = np.mean(rock)

water = np.zeros((x_size, y_size))
water += mean_rock/5
# water[0,0] += x_size*y_size*mean_rock/5

plot_spash = 1
minc = -5
maxc = 12

for i in range(10):
    if plot_spash:
        plot_4(rock, water, X, Y, minc, maxc, i)

    for j in range(500):
        rock, water = flow_water(rock, water, 0.10)

for i in range(10):
    plot_1(rock, water, X, Y, minc, maxc, i)
    for j in range(500):
        rock , water = evap(rock, water, 0.0003)
        rock, water = flow_water(rock, water, 0.10)
