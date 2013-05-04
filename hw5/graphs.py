import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

plt.clf()
xs = [0.8, 0.85, 0.9, 0.95, 1]
ys1 = [27.1, 20.14, 21.33, 17.75, 18.44]
ys2 = [23.99, 24.57, 24.15, 24.6, 23.0]
p1, = plt.plot(xs, ys1, color='b')
p2, = plt.plot(xs, ys2, color='r')
plt.title('Q Learning Algo Performance')
plt.xlabel('Learning rate')
plt.ylabel('Average throws')
plt.axis([0.8, 1, 15, 30])
plt.legend([p1,p2], ['strategy 1','strategy 2'], 'upper right')

savefig('qlearning.png') # save the figure to a file
plt.show() # show the figure