import os
import operator
import functools
import mdptoolbox
import numpy
numpy.set_printoptions(linewidth=1000,threshold=numpy.nan)
import math
import tkinter
import threading
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
from colour import Color

class TK_Interface:

    # Core Functions #
    #----------------#

    def gaussian(self, x, mu=0, sig=1):
        return numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sig, 2.)))

    def init_plots(self):

        plt.close()

        # trajectory plots
        self.traj_fig = plt.figure()
        self.pos_graph = self.traj_fig.add_subplot(3, 1, 1)
        self.vel_graph = self.traj_fig.add_subplot(3, 1, 2)
        self.acc_graph = self.traj_fig.add_subplot(3, 1, 3)

        # policy map
        self.policy_fig = plt.figure()
        self.policy_graph = self.policy_fig.add_subplot(1,1,1)

        self.traj_fig.tight_layout()  # turn on tight-layout

        #plt.ion()

        self.plot_samples = 0

    def update_plots(self, recomputed_flag):

        if recomputed_flag or self.plot_samples != int(self.e_num_samples.get()):

            self.plot_samples = int(self.e_num_samples.get())

            # position graph
            self.pos_graph.cla()
            self.pos_graph.set_title('Displacement (cm)')
            self.pos_graph.set_xlim(0, max(self.t_touch))
            min_pos = float(self.e_min_pos.get())
            max_pos = float(self.e_max_pos.get())
            mid_pos = (max_pos + min_pos) / 2
            y_min = mid_pos - (mid_pos - min_pos)
            y_max = mid_pos + (max_pos - mid_pos) * 1.1
            self.pos_graph.set_ylim(y_min, y_max)

            # velocity graph
            self.vel_graph.cla()
            self.vel_graph.set_title('Velocity (cm/s)')
            self.vel_graph.set_xlim(0, max(self.t_touch))
            min_vel = float(self.e_min_vel.get())
            max_vel = float(self.e_max_vel.get())
            mid_vel = (max_vel + min_vel) / 2
            y_min = mid_vel - (mid_vel - min_vel)
            y_max = mid_vel + (max_vel - mid_vel) * 1.1
            self.vel_graph.set_ylim(y_min, y_max)

            # acceleration graph
            self.acc_graph.cla()
            self.acc_graph.set_xlabel('Time (s)')
            self.acc_graph.set_title('Acceleration (cm/s^2)')
            self.acc_graph.set_xlim(0, max(self.t_touch))
            min_acc = min([min(sublist) for sublist in self.acc])
            max_acc = max([max(sublist) for sublist in self.acc])
            mid_acc = (max_acc + min_acc) / 2
            y_min = mid_acc - (mid_acc - min_acc) * 1.1
            y_max = mid_acc + (max_acc - mid_acc) * 1.1
            self.acc_graph.set_ylim(y_min, y_max)
            self.acc_x_axis = self.acc_graph.plot(numpy.array([0, max(self.t_touch)]), numpy.array([0, 0]), 'k')[0]  # x axis

            # policy map
            self.policy_graph.cla()
            self.policy_graph.set_xlabel('Velocity (cm/s)')
            self.policy_graph.set_ylabel('Displacement (cm)')
            policy_map = numpy.zeros((self.num_pos_states, self.num_actions))
            for j in range(self.num_pos_states):
                for k in range(self.num_actions):
                    policy_map[j, k] = self.vi.policy[
                        int(self.num_states - j * self.num_actions - self.num_actions + k)]
            self.policy_graph.imshow(policy_map, aspect='auto', interpolation='none',
                                     extent=[float(self.e_min_vel.get()), float(self.e_max_vel.get()),
                                             float(self.e_max_pos.get()), float(self.e_min_pos.get())])
            self.policy_graph.autoscale(False)

            # init traj arrays
            self.pos_lines = []
            self.pos_meas_lines = []
            self.pos_t_lines = []
            self.vel_lines = []
            self.vel_t_lines = []
            self.acc_lines = []
            self.acc_t_lines = []
            self.policy_lines = []

            for i in range(0,int(self.e_num_samples.get())):

                self.pos_meas_lines.append(self.pos_graph.plot(self.time[i], self.pos_meas[i], 'm', marker='x')[0])
                self.pos_lines.append(self.pos_graph.plot(self.time[i], self.pos[i], 'b', marker='o', markeredgecolor='w')[0])
                self.pos_t_lines.append(self.pos_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]), numpy.array([y_min, y_max]), 'r--')[0])

                self.vel_lines.append(self.vel_graph.plot(self.time[i], self.vel[i], 'r', marker='o', markeredgecolor='w')[0])
                self.vel_t_lines.append(self.vel_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]), numpy.array([y_min, y_max]), 'r--')[0])

                self.acc_lines.append(self.acc_graph.plot(self.time[i], self.acc[i], 'g', marker='o', markeredgecolor='w')[0])
                self.acc_t_lines.append(self.acc_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]), numpy.array([y_min, y_max]), 'r--')[0])

                self.policy_lines.append(self.policy_graph.plot(self.vel[i], self.pos[i], color='k', linestyle='-', linewidth=3, marker='o', markeredgecolor='w')[0])

        else:

            if (self.init_pos == float(self.e_max_pos.get())):
                    # max pos required in condition to prevent long trajectories from small starting positions
                    self.pos_graph.set_xlim(0, max(self.t_touch))
                    self.vel_graph.set_xlim(0, max(self.t_touch))
                    self.acc_graph.set_xlim(0, max(self.t_touch))
                    self.acc_x_axis.set_xdata(numpy.array([0, max(self.t_touch)]))

            # trajectories

            for i in range(0,int(self.e_num_samples.get())):

                self.pos_lines[i].set_xdata(self.time[i])
                self.pos_lines[i].set_ydata(self.pos[i])
                self.pos_meas_lines[i].set_xdata(self.time[i])
                self.pos_meas_lines[i].set_ydata(self.pos_meas[i])
                self.pos_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                self.vel_lines[i].set_xdata(self.time[i])
                self.vel_lines[i].set_ydata(self.vel[i])
                self.vel_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                self.acc_lines[i].set_xdata(self.time[i])
                self.acc_lines[i].set_ydata(self.acc[i])
                self.acc_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                # policy map
                self.policy_lines[i].set_xdata(self.vel[i])
                self.policy_lines[i].set_ydata(self.pos[i])

        self.traj_fig.canvas.draw()
        self.policy_fig.canvas.draw()

    def update_gui(self, event):

        # space, vel, and time resolutions
        self.vel_res = (float(self.e_max_vel.get()) - float(self.e_min_vel.get())) / (float(self.e_num_actions.get()) - 1)
        self.sv_vel_res.set('      vel res:        %.2f   cm/s    ' % self.vel_res)
        self.pos_res = (float(self.e_max_pos.get()) - float(self.e_min_pos.get())) / (float(self.e_num_positions.get()) - 1)
        self.sv_pos_res.set('      pos res:        %.2f   cm    ' % self.pos_res)
        self.control_freq = self.vel_res / self.pos_res
        self.sv_control_freq.set('      control freq:        %.2f   Hz    ' % self.control_freq)

        # initial state
        self.s_init_pos.config(from_=float(self.e_min_pos.get()), to=float(self.e_max_pos.get()), resolution=self.pos_res, tickinterval=(float(self.e_max_pos.get())-float(self.e_min_pos.get()))/5)
        #if self.init_pos > float(self.e_max_pos.get()) and self.init_pos < float(self.e_min_pos.get()):
        self.s_init_pos.set(float(self.e_max_pos.get())) # TRYING TO FIX SLIDER RESET
        self.s_init_vel.config(from_=float(self.e_min_vel.get()), to=float(self.e_max_vel.get()), resolution=self.vel_res, tickinterval=(float(self.e_max_vel.get())-float(self.e_min_vel.get()))/5)
        #if self.init_vel > float(self.e_max_vel.get()) and self.init_vel < float(self.e_min_vel.get()):
        self.s_init_vel.set(float(self.e_min_vel.get())) # TRYING TO FIX SLIDER RESET

        # parameters
        self.num_actions = int(self.e_num_actions.get())
        self.num_pos_states = int(self.e_num_positions.get())
        self.num_motion_bins = int(self.e_motion_bins.get())
        self.num_pos_meas_bins = int(self.e_pos_meas_bins.get())
        self.num_states = self.num_pos_states * self.num_actions

        # crude gaussian histograms

        # motion
        self.crude_motion_hist = self.gaussian(numpy.linspace(-3, 3, 2 * self.num_motion_bins + 1))
        self.crude_motion_hist = self.crude_motion_hist[1::2]
        self.crude_motion_hist = self.crude_motion_hist / numpy.sum(self.crude_motion_hist)

        # pos measurement
        self.crude_pos_meas_hist = self.gaussian(numpy.linspace(-3, 3, 2 * self.num_pos_meas_bins + 1))
        self.crude_pos_meas_hist = self.crude_pos_meas_hist[1::2]
        self.crude_pos_meas_hist = self.crude_pos_meas_hist / numpy.sum(self.crude_pos_meas_hist)

        self.motion_centre_idx = int(self.num_motion_bins / 2)
        self.pos_meas_centre_idx = int(self.num_pos_meas_bins / 2)



        self.tk_root.update_idletasks()

    def reset_gui(self):

        self.e_min_vel.delete(0, tkinter.END)
        self.e_max_vel.delete(0, tkinter.END)
        self.e_num_actions.delete(0, tkinter.END)
        self.e_min_pos.delete(0, tkinter.END)
        self.e_max_pos.delete(0, tkinter.END)
        self.e_num_positions.delete(0, tkinter.END)
        self.e_discount.delete(0, tkinter.END)
        self.e_epsilon.delete(0, tkinter.END)
        self.e_max_iter.delete(0, tkinter.END)
        self.e_dist_factor.delete(0, tkinter.END)
        self.e_vel_factor.delete(0, tkinter.END)
        self.e_vel_den_ratio.delete(0, tkinter.END)
        self.e_acc_factor.delete(0, tkinter.END)
        self.e_motion_bins.delete(0, tkinter.END)
        self.e_pos_meas_bins.delete(0, tkinter.END)
        self.e_num_samples.delete(0, tkinter.END)


        self.e_min_vel.insert(tkinter.END, '0')
        self.e_max_vel.insert(tkinter.END, '20')
        self.e_num_actions.insert(tkinter.END, '6')
        self.e_min_pos.insert(tkinter.END, '0')
        self.e_max_pos.insert(tkinter.END, '50')
        self.e_num_positions.insert(tkinter.END, '126')
        self.e_discount.insert(tkinter.END, '0.99')
        self.e_epsilon.insert(tkinter.END, '0.01')
        self.e_max_iter.insert(tkinter.END, '1000')
        self.e_dist_factor.insert(tkinter.END, '1')
        self.e_vel_factor.insert(tkinter.END, '1')
        self.e_vel_den_ratio.insert(tkinter.END, '0.05')
        self.e_acc_factor.insert(tkinter.END, '1')
        self.e_motion_bins.insert(tkinter.END, '1')
        self.e_pos_meas_bins.insert(tkinter.END, '3')
        self.e_num_samples.insert(tkinter.END, '1')
        self.c_sample_w_motion_noise = 0
        self.c_sample_w_pos_meas_noise = 0

        #self.init_pos = float(self.e_max_pos.get()) # TRYING TO FIX SLIDER RESET
        #self.init_vel = float(self.e_min_vel.get())

        self.update_gui('dummy_event')

    def compute_vi(self):

        # transition matrix
        transitions = numpy.zeros((self.num_actions, self.num_states, self.num_states))
        no_vel_trans = numpy.zeros((self.num_states, self.num_states))

        # setup base zero velocity transition matrix, for i = 0
        for i in range(0, int(self.num_states / self.num_actions)): # iterate over pos states
            for j in range(0,self.num_motion_bins): # iterate over bins
                bin_range = (j - self.motion_centre_idx) * self.num_actions
                horizontal_coord = i * self.num_actions + bin_range
                no_vel_trans[i * self.num_actions:i * self.num_actions + self.num_actions, horizontal_coord % self.num_states] = numpy.full(self.num_actions, self.crude_motion_hist[j])

        # setup full transition matrix based on position state
        for i in range(0, self.num_actions):

            # roll entire state
            transitions[i, :, :] = numpy.roll(no_vel_trans[:,:], i * self.num_actions + i, 1)

            #re-normalise border at top
            if (i < self.motion_centre_idx):  # if there are border terms
                for j in range(0, self.motion_centre_idx - i): # iterate over groups
                    # top-left (starting state border)
                    transitions[i,j*self.num_actions:(j+1)*self.num_actions, i] += numpy.sum(self.crude_motion_hist[0:self.motion_centre_idx - j - i])
                    # top-right (initial rolled over)
                    start = self.num_states - (self.motion_centre_idx - i) * self.num_actions + i
                    transitions[i, 0:(j + 1) * self.num_actions, start + j*self.num_actions] = 0


            #re-normalise border at bottom
            for j in range(0, self.motion_centre_idx + i):
                # bottom-right (final state border)
                start_j = self.num_states+ (j - self.motion_centre_idx - i) * self.num_actions
                end_j = self.num_states+ (j + 1 - self.motion_centre_idx - i) * self.num_actions
                transitions[i,start_j:end_j,self.num_states-self.num_actions+i] += numpy.sum(self.crude_motion_hist[0:j + 1])\
                                                                if j<float(self.e_motion_bins.get()) else 1
                # bottom-right (final state-rolled over)
                transitions[i,start_j:self.num_states, i+j*self.num_actions] = 0

        # reward matrix

        reward_entry_mat = numpy.zeros((self.num_pos_meas_bins))
        reward = numpy.zeros((self.num_states, self.num_actions))
        for i in range(0, self.num_states):
            for j in range(0, self.num_actions):

                # each bin averaged reward entry term

                for pos_it in range(0,self.num_pos_meas_bins):

                    # each component of the bin average

                    # dist to target
                    unbounded_dist = abs(math.floor((self.num_states - i -1) / self.num_actions)) + pos_it - self.pos_meas_centre_idx
                    if unbounded_dist <= 0:
                        dist_to_target = 0
                    elif unbounded_dist >= self.num_pos_states:
                        dist_to_target = abs(math.floor((self.num_states-1) / self.num_actions)) * self.pos_res
                    else:
                        dist_to_target = unbounded_dist * self.pos_res

                    # velocities
                    vel = j * self.vel_res
                    prev_vel = (i - math.floor(i / self.num_actions) * self.num_actions) * self.vel_res

                    # other terms
                    vel_over_dist = vel / (dist_to_target + float(self.e_max_pos.get()) * float(self.e_vel_den_ratio.get()))
                    delta_vel_sqaured = pow((vel - prev_vel), 2)

                    # final reward expression for specific belief state
                    reward_entry_mat[pos_it] = -float(self.e_dist_factor.get())*dist_to_target\
                                               -float(self.e_vel_factor.get()) * vel_over_dist\
                                               -float(self.e_acc_factor.get()) * delta_vel_sqaured

                # total reward of being in this state, given possibilities
                reward[i, j] = numpy.sum(numpy.dot(reward_entry_mat, self.crude_pos_meas_hist))

        # normalise total reward mat
        reward /= abs(numpy.sum(reward)/reward.size)

        # define value iteration
        self.vi = mdptoolbox.mdp.ValueIteration(transitions, reward, float(self.e_discount.get()), float(self.e_epsilon.get()), float(self.e_max_iter.get()), 0)
        self.vi.run()

    def compute_trajs(self):

        # parameters
        time_step = 1/float(self.control_freq)

        # initialise trajectories for plotting
        self.pos_raw = [[] for i in range(int(self.e_num_samples.get()))]
        self.pos_meas_raw = [[] for i in range(int(self.e_num_samples.get()))]
        self.vel_raw = [[] for i in range(int(self.e_num_samples.get()))]
        self.acc_raw = [[] for i in range(int(self.e_num_samples.get()))]
        self.time = [[] for i in range(int(self.e_num_samples.get()))]
        self.t_touch = []

        self.pos = [[] for i in range(int(self.e_num_samples.get()))]
        self.pos_meas = [[] for i in range(int(self.e_num_samples.get()))]
        self.vel = [[] for i in range(int(self.e_num_samples.get()))]
        self.acc = [[] for i in range(int(self.e_num_samples.get()))]

        # iterate over n samples
        for i in range(0,int(self.e_num_samples.get())):

            # initialise state
            state = [int(float(self.s_init_pos.get()) / self.pos_res), int(float(self.s_init_vel.get()) / self.vel_res)]
            t = 0

            # populate trajectories
            at_target = False
            traj_end = False
            while traj_end is False:

                if (state[0] == 0 and at_target == False):
                    self.t_touch.append(t)
                    at_target = True
                    traj_end = True

                # measurement
                if self.sample_w_pos_meas_noise.get() == 1:
                    pos_meas = state[0] + numpy.random.choice(numpy.linspace(-self.pos_meas_centre_idx, self.pos_meas_centre_idx, self.num_pos_meas_bins), 1, p=self.crude_pos_meas_hist)
                else:
                    pos_meas = state[0]

                if pos_meas < float(self.e_min_pos.get())/self.pos_res:
                    pos_meas = float(self.e_min_pos.get())/self.pos_res
                elif pos_meas > float(self.e_max_pos.get())/self.pos_res:
                    pos_meas = float(self.e_max_pos.get())/self.pos_res
                state_meas = [pos_meas, state[1]]

                target_vel = self.vi.policy[int(self.num_states - state_meas[0] * self.num_actions - self.num_actions + state_meas[1])]

                self.pos_raw[i].append(state[0]) # current pos
                self.pos_meas_raw[i].append(state_meas[0]) # current measured pos
                self.vel_raw[i].append(state[1]) # current velocity
                self.acc_raw[i].append(target_vel-state[1]) # current accel
                self.time[i].append(t)
                t += time_step

                # motion noise
                if self.sample_w_motion_noise.get() == 1:
                    new_pos = state[0] - target_vel + numpy.random.choice(numpy.linspace(-self.motion_centre_idx, self.motion_centre_idx, self.num_motion_bins), 1, p=self.crude_motion_hist)
                else:
                    new_pos = state[0] - target_vel

                if new_pos < float(self.e_min_pos.get())/self.pos_res:
                    new_pos = float(self.e_min_pos.get())/self.pos_res
                elif new_pos > float(self.e_max_pos.get())/self.pos_res:
                    new_pos = float(self.e_max_pos.get())/self.pos_res
                state = [new_pos, target_vel]

            # rescale trajectories to correct dimensions
            self.pos[i] = [j*self.pos_res for j in self.pos_raw[i]]
            self.pos_meas[i] = [j * self.pos_res for j in self.pos_meas_raw[i]]
            self.vel[i] = [j*self.vel_res for j in self.vel_raw[i]]
            self.acc[i] = [j*self.vel_res*self.control_freq for j in self.acc_raw[i]]

    def compute(self):
        self.compute_vi()
        self.compute_trajs()

    # GUI Interaction Functions #
    #---------------------------#

    def BP_compute(self):
        self.update_gui('dummy_event')
        self.compute()
        self.update_plots(recomputed_flag=True)

    def BP_resample(self):
        self.update_gui('dummy_event')
        self.compute_trajs()
        self.update_plots(recomputed_flag=False)

    def BP_reset(self):
        self.reset_gui()

    def BP_terminate(self):
        os._exit(os.EX_OK)

    def S_init_pos(self, v):
        self.init_pos = float(v)
        self.compute_trajs()
        self.update_plots(recomputed_flag=False)

    def S_init_vel(self, v):
        self.init_vel = float(v)
        self.compute_trajs()
        self.update_plots(recomputed_flag=False)

    # Main GUI Loop Funcion #
    #-----------------------#

    def tkinter_loop(self):

        # root gui
        self.tk_root = tkinter.Tk()

        # text box frame
        f_tb = tkinter.Frame(self.tk_root)
        f_tb.pack()

        # slider frame
        f_s = tkinter.Frame(self.tk_root)
        f_s.pack()

        # button frame
        f_b = tkinter.Frame(self.tk_root)
        f_b.pack()

        # plot frame
        f_p = tkinter.Frame(self.tk_root)
        f_p.pack()

        # for grid indexing
        row = 0
        column = 0

        # Velocity Parameters #
        #---------------------#

        column += 3

        l_vel = tkinter.Label(f_tb, text='velocity Parameters')
        l_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        l_min_vel = tkinter.Label(f_tb, text='min vel (cm/s)')
        l_min_vel.grid(row=row, column=column)

        column += 1

        self.e_min_vel = tkinter.Entry(f_tb)
        self.e_min_vel.grid(row=row, column=column)
        self.e_min_vel.bind("<FocusOut>", self.update_gui)

        column += 1

        l_max_vel = tkinter.Label(f_tb, text='max vel (cm/s)')
        l_max_vel.grid(row=row, column=column)

        column += 1

        self.e_max_vel = tkinter.Entry(f_tb)
        self.e_max_vel.grid(row=row, column=column)
        self.e_max_vel.bind("<FocusOut>", self.update_gui)

        column += 1

        l_num_actions = tkinter.Label(f_tb, text='num actions')
        l_num_actions.grid(row=row, column=column)

        column += 1

        self.e_num_actions = tkinter.Entry(f_tb)
        self.e_num_actions.grid(row=row, column=column)
        self.e_num_actions.bind("<FocusOut>", self.update_gui)

        column += 1

        self.sv_vel_res = tkinter.StringVar()
        l_vel_res = tkinter.Label(f_tb, textvariable=self.sv_vel_res)
        l_vel_res.grid(row=row, column=column)

        row += 1
        column -= 6

        # Position Parameters #
        #---------------------#

        column += 3

        l_pos = tkinter.Label(f_tb, text='Position Parameters')
        l_pos.grid(row=row, column=column)

        row += 1
        column -= 3

        l_min_pos = tkinter.Label(f_tb, text='min pos (cm)')
        l_min_pos.grid(row=row, column=column)

        column += 1

        self.e_min_pos = tkinter.Entry(f_tb)
        self.e_min_pos.grid(row=row, column=column)
        self.e_min_pos.bind("<FocusOut>", self.update_gui)

        column += 1

        l_max_pos = tkinter.Label(f_tb, text='max pos (cm)')
        l_max_pos.grid(row=row, column=column)

        column += 1

        self.e_max_pos = tkinter.Entry(f_tb)
        self.e_max_pos.grid(row=row, column=column)
        self.e_max_pos.bind("<FocusOut>", self.update_gui)

        column += 1

        l_num_positions = tkinter.Label(f_tb, text='num positions')
        l_num_positions.grid(row=row, column=column)

        column += 1

        self.e_num_positions = tkinter.Entry(f_tb)
        self.e_num_positions.grid(row=row, column=column)
        self.e_num_positions.bind("<FocusOut>", self.update_gui)

        column += 1

        self.sv_pos_res = tkinter.StringVar()
        l_pos_res = tkinter.Label(f_tb, textvariable=self.sv_pos_res)
        l_pos_res.grid(row=row, column=column)

        row += 1
        column -= 6

        # Value Iteration Parameters #
        #----------------------------#

        column += 3

        l_vi = tkinter.Label(f_tb, text='Value Iteration Parameters')
        l_vi.grid(row=row, column=column)

        row += 1
        column -= 3

        l_discount = tkinter.Label(f_tb, text='discount')
        l_discount.grid(row=row, column=column)

        column += 1

        self.e_discount = tkinter.Entry(f_tb)
        self.e_discount.grid(row=row, column=column)

        column += 1

        l_epsilon = tkinter.Label(f_tb, text='epsilon')
        l_epsilon.grid(row=row, column=column)

        column += 1

        self.e_epsilon = tkinter.Entry(f_tb)
        self.e_epsilon.grid(row=row, column=column)

        column += 1

        l_max_iter = tkinter.Label(f_tb, text='max iter')
        l_max_iter.grid(row=row, column=column)

        column += 1

        self.e_max_iter = tkinter.Entry(f_tb)
        self.e_max_iter.grid(row=row, column=column)

        column += 1

        # control frequency, to align properly on r.h.s.
        self.sv_control_freq = tkinter.StringVar()
        l_control_freq = tkinter.Label(f_tb, textvariable=self.sv_control_freq)
        l_control_freq.grid(row=row, column=column)

        row += 1
        column -= 6

        # Reward Parameters #
        #-------------------#

        column += 3

        l_rew = tkinter.Label(f_tb, text='Reward Parameters')
        l_rew.grid(row=row, column=column)

        row += 1
        column -= 3

        l_dist_factor = tkinter.Label(f_tb, text='dist factor')
        l_dist_factor.grid(row=row, column=column)

        column += 1

        self.e_dist_factor = tkinter.Entry(f_tb)
        self.e_dist_factor.grid(row=row, column=column)

        column += 1

        l_vel_factor = tkinter.Label(f_tb, text='vel factor')
        l_vel_factor.grid(row=row, column=column)

        column += 1

        self.e_vel_factor = tkinter.Entry(f_tb)
        self.e_vel_factor.grid(row=row, column=column)

        column += 1

        l_vel_den_ratio = tkinter.Label(f_tb, text='vel den ratio')
        l_vel_den_ratio.grid(row=row, column=column)

        column += 1

        self.e_vel_den_ratio = tkinter.Entry(f_tb)
        self.e_vel_den_ratio.grid(row=row, column=column)

        column += 1

        l_acc_factor = tkinter.Label(f_tb, text='acc^2 factor')
        l_acc_factor.grid(row=row, column=column)

        column += 1

        self.e_acc_factor = tkinter.Entry(f_tb)
        self.e_acc_factor.grid(row=row, column=column)

        row += 1
        column -=7

        # Noise Parameters #
        #------------------#

        column += 3

        l_noise = tkinter.Label(f_tb, text='Noise Parameters')
        l_noise.grid(row=row, column=column)

        row += 1
        column -= 3

        l_motion_bins = tkinter.Label(f_tb, text='motion bins')
        l_motion_bins.grid(row=row, column=column)

        column += 1

        self.e_motion_bins = tkinter.Entry(f_tb)
        self.e_motion_bins.grid(row=row, column=column)

        column += 1

        l_pos_meas_bins = tkinter.Label(f_tb, text='pos meas bins')
        l_pos_meas_bins.grid(row=row, column=column)

        column += 1

        self.e_pos_meas_bins = tkinter.Entry(f_tb)
        self.e_pos_meas_bins.grid(row=row, column=column)

        column += 1

        l_sample_w_motion_noise = tkinter.Label(f_tb, text='sample with motion noise?')
        l_sample_w_motion_noise.grid(row=row, column=column)

        column += 1

        self.sample_w_motion_noise = tkinter.IntVar()
        self.c_sample_w_motion_noise = tkinter.Checkbutton(f_tb,text="yes",variable=self.sample_w_motion_noise)
        self.c_sample_w_motion_noise.grid(row=row, column=column)

        column += 1

        l_sample_w_pos_meas_noise = tkinter.Label(f_tb, text='sample with pos meas noise?')
        l_sample_w_pos_meas_noise.grid(row=row, column=column)

        column += 1

        self.sample_w_pos_meas_noise = tkinter.IntVar()
        self.c_sample_w_pos_meas_noise = tkinter.Checkbutton(f_tb,text="yes",variable=self.sample_w_pos_meas_noise)
        self.c_sample_w_pos_meas_noise.grid(row=row, column=column)

        row += 1
        column -= 7

        # Plotting Options #
        #------------------#

        column += 3

        l_plotting = tkinter.Label(f_tb, text='Plotting Options')
        l_plotting.grid(row=row, column=column)

        row += 1
        column -= 3

        l_num_samples = tkinter.Label(f_tb, text='num samples')
        l_num_samples.grid(row=row, column=column)

        column += 1

        self.e_num_samples = tkinter.Entry(f_tb)
        self.e_num_samples.grid(row=row, column=column)

        row += 1
        column -= 1

        # Initial State #
        #---------------#

        row = 0
        column = 0

        column += 2

        l_init_cond = tkinter.Label(f_s, text='Initial State')
        l_init_cond.grid(row=row, column=column)

        row += 1
        column -= 2

        l_init_pos = tkinter.Label(f_s, text='init pos')
        l_init_pos.grid(row=row, column=column)

        column += 1

        self.s_init_pos = tkinter.Scale(f_s, length=300, width=45, orient=tkinter.HORIZONTAL)
        self.s_init_pos.grid(row=row, column=column)

        column += 1

        l_init_vel = tkinter.Label(f_s, text='init vel')
        l_init_vel.grid(row=row, column=column)

        column += 1

        self.s_init_vel = tkinter.Scale(f_s, length=300, width=45, orient=tkinter.HORIZONTAL)
        self.s_init_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        # Buttons #
        #---------#

        row = 0
        column = 0

        column += 1

        l_buttons = tkinter.Label(f_b, text='Buttons')
        l_buttons.grid(row=row, column=column)

        row += 1
        column -= 1

        b_compute = tkinter.Button(f_b, text='Compute', command=self.BP_compute)
        b_compute.grid(row=row, column=column)

        column += 1

        b_resample = tkinter.Button(f_b, text='Resample', command=self.BP_resample)
        b_resample.grid(row=row, column=column)

        column += 1

        b_reset = tkinter.Button(f_b, text='Reset', command=self.BP_reset)
        b_reset.grid(row=row, column=column)

        column += 1

        b_terminate = tkinter.Button(f_b, text='Terminate', command=self.BP_terminate)
        b_terminate.grid(row=row, column=column)

        row += 1
        column -= 2

        # Plots #
        #-------#

        self.init_plots()

        row = 0
        column = 0

        column += 1

        l_plots = tkinter.Label(f_p, text='Plots')
        l_plots.grid(row=row, column=column)

        row += 1
        column -= 1

        self.c_traj_plot = tkagg.FigureCanvasTkAgg(self.traj_fig, master=f_p)
        self.c_traj_plot.get_tk_widget().grid(row=row, column=column)
        self.c_traj_plot._tkcanvas.grid(row=row, column=column) # not sure if needed?

        column += 2

        self.c_policy_plot = tkagg.FigureCanvasTkAgg(self.policy_fig, master=f_p)
        self.c_policy_plot.get_tk_widget().grid(row=row, column=column)
        self.c_policy_plot._tkcanvas.grid(row=row, column=column) # not sure if needed?


        # Initialise GUI #
        #----------------#

        self.reset_gui()
        self.compute()
        self.update_plots(recomputed_flag=True)
        # set up sliders for plot updates
        self.s_init_pos.config(command = lambda v: self.S_init_pos(v))
        self.s_init_vel.config(command = lambda v: self.S_init_vel(v))

        # start mainloop #
        #----------------#

        self.tk_root.mainloop()

def main():

    tk_interface = TK_Interface()

    # start tkinter gui
    threading.Thread(target=tk_interface.tkinter_loop).start()

if __name__ == "__main__":
    main()
