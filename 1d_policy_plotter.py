import numpy
import matplotlib.pyplot as plt
import time

# parameters #
#------------#

#time
one_second = 1000
t_max = one_second*10
time_traj = numpy.arange(0,t_max,1)/one_second

#distance
decel_radius = 5
min_vel_radius = 5
origin = 0

#velocity
v_cruise = 20 #cm/s
v_min = 4 # cm/s

#acceleration
accel_limit = 40 # cm/s^2

#measurements
measurement_var = 0.0001

# trajectory data
#x_true = numpy.random.uniform()*6
x_true = 100
pos_traj = numpy.zeros(t_max)
vel_traj = numpy.zeros(t_max)
acc_traj = numpy.zeros(t_max)

vel = 0 # initial velocity
t_touch = 0 # initialise t_touch

for t in range(0,t_max):

    # In this section, all vel and accel are expressed w.r.t discrete t timestep

    x_measured = numpy.random.normal(x_true, measurement_var)

    # find desired velocity
    if x_measured>decel_radius:
        vel_des = v_cruise/one_second
    elif x_measured<decel_radius and x_measured>min_vel_radius:
        vel_des = ((v_cruise-v_min)*((1/(decel_radius-min_vel_radius))*x_measured-min_vel_radius*(1/(decel_radius-min_vel_radius))) + v_min)/one_second
    elif x_measured<min_vel_radius and x_measured>origin:
        vel_des = v_min/one_second

    # find actual velocity, based on desired acceleration limit
    if abs(vel_des - vel) < accel_limit/(one_second*one_second):
        vel = vel_des
    elif vel_des > vel:
        vel += accel_limit/(one_second*one_second)
    elif vel_des < vel:
        vel -= accel_limit/(one_second*one_second)

    # if contact made
    if x_true<origin:
        vel = 0
        if t_touch == 0:
            t_touch = t

    pos_traj[t] = x_true
    vel_traj[t] = vel
    if t > 1:
        acc_traj[t] = (vel - vel_traj[t-1])
    x_true -= vel


#re-normalise trajectories
vel_traj *= one_second
acc_traj *= one_second*one_second

# to prevent accel touch spike morphing graph axes
acc_traj_wo_touch = acc_traj.copy()
acc_traj_wo_touch[t_touch] = 0

# normalise touch time
t_touch /= one_second

# plotting

fig = plt.figure()

pos_graph = fig.add_subplot(3,1,1)
pos_graph.set_xlim(0,t_max/one_second)
pos_graph.set_ylim(-0.5,numpy.amax(pos_traj)*1.1)
pos_graph.plot(time_traj, pos_traj, 'b')
pos_graph.plot(numpy.array([t_touch, t_touch]), numpy.array([-0.5,numpy.amax(pos_traj)*1.1]), 'r--')
pos_graph.set_title('Displacement (cm)')

vel_graph = fig.add_subplot(3,1,2)
vel_graph.set_xlim(0,t_max/one_second)
vel_graph.set_ylim(-0.05,numpy.amax(vel_traj)*1.1)
vel_graph.plot(time_traj, vel_traj, 'g')
vel_graph.plot(numpy.array([t_touch, t_touch]), numpy.array([-0.05,numpy.amax(vel_traj)*1.1]), 'r--')
vel_graph.set_title('Velocity (cm/s)')

acc_graph = fig.add_subplot(3,1,3)
acc_graph.set_xlim(0,t_max/one_second)
acc_graph.set_ylim(numpy.amin(acc_traj_wo_touch)*1.1,numpy.amax(acc_traj_wo_touch)*1.1)
acc_graph.plot(time_traj, acc_traj, 'k')
acc_graph.plot(numpy.array([t_touch, t_touch]), numpy.array([numpy.amin(acc_traj_wo_touch)*1.1,numpy.amax(acc_traj_wo_touch)*1.1]), 'r--')
acc_graph.set_xlabel('Time (s)')
acc_graph.set_title('Acceleration (cm/s^2)')


plt.tight_layout() # turn on tight-layout

plt.show()