import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

f1 = lambda t, theta, r: 3/(r**2) # d(theta)/dt
f2 = lambda t, theta, r: 0 # dr/dt
h = 1 # step size
tend=4
t = np.arange(0,tend+h,h)
theta0 = 0 # initial theta
r0 = 1 # initial r value

theta = np.zeros(len(t))
theta[0] = theta0
r = np.zeros(len(t))
r[0] = r0

for i in range(0,len(t)-1):
    theta[i+1] = theta[i] + 3*h/(r[i]**2)
    r[i+1] = r[i]

fig, ax = plt.subplots()
plt.axes(projection='polar')
plt.title('Fluid particle location at time t ='+str(tend)+'s for delta t ='+str(h)+'s' )
def animate(i):
    plt.polar(theta[i], r[i], 'g.')
ani = animation.FuncAnimation(fig, animate, frames =len(t), interval =100, blit=False)
plt.show()



