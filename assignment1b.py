import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

h = 0.01 # step size
tend=2
t = np.arange(0, tend+h, h)
x0 = 1 # initial x value
y0 = 0 # initial y value

x = np.zeros(len(t))
x[0] = x0
y = np.zeros(len(t))
y[0] = y0

for i in range(0,len(t)-1):
    x[i+1] = x[i] - 3*h*y[i]/(x[i]**2+y[i]**2)
    y[i+1] = y[i] + 3*h*x[i]/(x[i]**2+y[i]**2)

fig, ax = plt.subplots()
#plt.axes(projection='polar')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Fluid particle location at time t ='+str(32)+'s for delta t ='+str(h)+'s' )
def animate(i):
    plt.plot(x[i], y[i], 'b.')
ani = animation.FuncAnimation(fig, animate, frames =len(t), interval =100, blit=False)
plt.show()