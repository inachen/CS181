import matplotlib.pyplot as plt
from pylab import *

plt.clf()
xs = range(5)
ys = [3, 5, 1, 10, 8]
p1, = plt.plot(xs, ys, color='b')
plt.title('sample graph')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.axis([0, 4, 0, 12])

plt.legend((p1,), ('data',), 'lower right')
savefig('figure.pdf') # save the figure to a file
plt.show() # show the figure