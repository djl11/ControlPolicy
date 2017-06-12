import numpy
import matplotlib.pyplot as plt
import time

# parameters
v_cruise = 1
v_min = 0.2
one_second = 100
t_max = one_second*20

# trajectory data
pos_traj = numpy.zeros(t_max)
vel_traj = numpy.zeros(t_max)
acc_traj = numpy.zeros(t_max)

plt.ion() # turn interactive

fig = plt.figure()

live_graph = fig.add_subplot(4,1,1)
live_graph.set_ylim(-1,1)
live_graph.set_xlim(0,10)
x = numpy.random.uniform()*8 + 2 # random starting state between 2-10
live_data = live_graph.plot(x, 0, 'or')[0]

pos_graph = fig.add_subplot(4,1,2)
pos_graph.set_ylim(-1,11)
pos_graph.set_xlim(0,t_max)
pos_data = pos_graph.plot(numpy.arange(0,t_max,1), pos_traj, 'b')[0]

vel_graph = fig.add_subplot(4,1,3)
vel_graph.set_ylim(-0.1,1.1*v_cruise)
vel_graph.set_xlim(0,t_max)
vel_data = vel_graph.plot(numpy.arange(0,t_max,1), vel_traj, 'g')[0]

acc_graph = fig.add_subplot(4,1,4)
acc_graph.set_ylim(-0.02,0.02)
acc_graph.set_xlim(0,t_max)
acc_data = acc_graph.plot(numpy.arange(0,t_max,1), acc_traj, 'k')[0]

plt.tight_layout() # turn on tight-layout

for t in range(0,t_max):

    # select velocity
    if x>1 and t>one_second:
        vel = v_cruise
    elif x>1 and t<one_second:
        vel = v_cruise*t/one_second
    elif x<1 and t > one_second:
        vel = (v_cruise-v_min)*x + v_min
    elif x<0:
        vel = 0

    pos_traj[t] = x
    vel_traj[t] = vel
    if t > 1:
        acc_traj[t] = vel - vel_traj[t-1]
    x -= vel/one_second

    live_data.set_xdata(x)
    pos_data.set_ydata(pos_traj)
    vel_data.set_ydata(vel_traj)
    acc_data.set_ydata(acc_traj)
    fig.canvas.draw()

input('press enter to end program')