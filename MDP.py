import mdptoolbox
import numpy
import math

# parameters
num_actions = 21# 0-1 m/s
num_pos_states = 501
num_states = num_pos_states*num_actions # 0-10 m * 0-1 m/s

# transition matrix
transitions = numpy.zeros((num_actions,num_states,num_states))
no_vel_trans = numpy.zeros((num_states,num_states))
# setup base zero velocity transition matrix, for i = 0
for i in range(0,int(num_states/num_actions)):
    no_vel_trans[i*num_actions:i*num_actions+num_actions,i*num_actions] = numpy.ones(num_actions)
# setup full transition matrix based on position state
for i in range(0,num_actions):
    transitions[i,:,:] = numpy.roll(no_vel_trans,i*num_actions+i,1)

# reward matrix
reward = numpy.zeros((num_states,num_actions))
for i in range(0,num_states):
    for j in range(0,num_actions):
        reward[i,j] = -abs(math.floor((num_states-i)/num_actions))-pow((j-(i-math.floor(i/num_actions)*num_actions)),2)

# define value iteration
vi = mdptoolbox.mdp.ValueIteration(transitions,reward,0.99,0.00001,10000000,0)
vi.run()

# initialise state
state = numpy.array([100,0])

# print trajectory
for i in range(0,15):
    print("%.2f    %.2f" % (state[0], state[1]))
    vel = vi.policy[int(num_states-state[0]*num_actions-num_actions+state[1])]
    state = [(state[0]-vel) % num_pos_states, vel]

print('finished value iteration')