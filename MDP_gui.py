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

    def compute_resolutions(self):
        self.vel_res = (float(self.e_max_vel.get()) - float(self.e_min_vel.get())) / (
            float(self.e_num_actions.get()) - 1)
        self.pos_res = (float(self.e_max_pos.get()) - float(self.e_min_pos.get())) / (
            float(self.e_num_positions.get()) - 1)
        self.control_freq = self.vel_res / self.pos_res

    def gaussian(self, x, mu=0, sig=1):
        return numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sig, 2.)))

    def init_plots(self):

        plt.close()

        plt.ion()

        # trajectory plots
        self.traj_fig = plt.figure()
        self.pos_graph = self.traj_fig.add_subplot(5,1,1)
        self.vel_graph = self.traj_fig.add_subplot(5,1,2)
        self.acc_graph = self.traj_fig.add_subplot(5,1,3)
        self.rew_graph = self.traj_fig.add_subplot(5,1,4)
        self.cum_rew_graph = self.traj_fig.add_subplot(5,1,5)

        plt.ioff()

        # policy map
        self.policy_fig = plt.figure()
        self.policy_graph = self.policy_fig.add_subplot(1,1,1)

        self.traj_fig.tight_layout()  # turn on tight-layout

        self.plot_samples = 0

    def update_plots(self, recomputed_flag):

        if recomputed_flag or self.plot_samples != int(self.e_num_samples.get()):

            self.plot_samples = int(self.e_num_samples.get())

            # t term
            self.t_term = numpy.nan
            if self.dof_comp.get() == 0:
                self.t_term = max(self.t_touch)
            elif self.dof_comp.get() == 1 or self.dof_comp.get() == 2:
                self.t_term = self.t_max

            # position graph
            self.pos_graph.cla()
            if self.dof_comp.get() == 0 or self.dof_comp.get() == 1:
                self.pos_graph.set_title('Displacement (cm)')
            elif self.dof_comp.get() == 2:
                self.pos_graph.set_title('Displacement (rad)')
            min_pos = float(self.e_min_pos.get())
            max_pos = float(self.e_max_pos.get())
            mid_pos = (max_pos + min_pos) / 2
            pos_y_min = mid_pos - (mid_pos - min_pos)
            pos_y_max = mid_pos + (max_pos - mid_pos) * 1.1
            if self.dof_comp.get() == 0:
                self.pos_graph.set_ylim(pos_y_min, pos_y_max)
            elif self.dof_comp.get() == 1:
                pos_min_est = float((self.num_motion_bins-1)/2+(self.num_pos_meas_bins-1)/2)*self.pos_res
                self.pos_graph.set_ylim(-pos_min_est, pos_y_max)
            elif self.dof_comp.get() == 2:
                self.pos_graph.set_ylim(-pos_y_max, pos_y_max)
            self.pos_graph.set_xlim(0, self.t_term)
            self.pos_x_axis = self.pos_graph.plot(numpy.array([0, self.t_term]), numpy.array([0, 0]), 'k')[0]  # x axis


            # velocity graph
            self.vel_graph.cla()
            if self.dof_comp.get() == 0 or self.dof_comp.get() == 1:
                self.vel_graph.set_title('Velocity (cm/s)')
            elif self.dof_comp.get() == 2:
                self.vel_graph.set_title('Velocity (rad/s)')
            min_vel = float(self.e_min_vel.get())
            max_vel = float(self.e_max_vel.get())
            mid_vel = (max_vel + min_vel) / 2
            vel_y_min = mid_vel - (mid_vel - min_vel)
            vel_y_max = mid_vel + (max_vel - mid_vel) * 1.1
            if self.dof_comp.get() == 0:
                self.vel_graph.set_ylim(vel_y_min, vel_y_max)
            elif self.dof_comp.get() == 1 or self.dof_comp.get() == 2:
                self.vel_graph.set_ylim(-vel_y_max, vel_y_max)
            self.vel_graph.set_xlim(0, self.t_term)
            self.vel_x_axis = self.vel_graph.plot(numpy.array([0, self.t_term]), numpy.array([0, 0]), 'k')[0]  # x axis


            # acceleration graph
            self.acc_graph.cla()
            if self.dof_comp.get() == 0 or self.dof_comp.get() == 1:
                self.acc_graph.set_title('Acceleration (cm/s^2)')
            elif self.dof_comp.get() == 2:
                self.acc_graph.set_title('Acceleration (rad/s^2)')
            min_acc = min([min(sublist) for sublist in self.acc])
            max_acc = max([max(sublist) for sublist in self.acc])
            mid_acc = (max_acc + min_acc) / 2
            self.acc_y_min = mid_acc - (mid_acc - min_acc) * 1.1
            self.acc_y_max = mid_acc + (max_acc - mid_acc) * 1.1
            if self.dof_comp.get() == 0:
                self.acc_graph.set_ylim(self.acc_y_min, self.acc_y_max)
            elif self.dof_comp.get() == 1 or self.dof_comp.get() == 2:
                self.acc_graph.set_ylim(-self.acc_y_max, self.acc_y_max)
            self.acc_graph.set_xlim(0, self.t_term)
            self.acc_x_axis = self.acc_graph.plot(numpy.array([0, self.t_term]), numpy.array([0, 0]), 'k')[0]  # x axis

            # rewards

            # mean instant and cum rewards
            self.mean_cum_rew = 0
            for i in range(0, int(self.e_num_samples.get())):
                self.mean_cum_rew += self.cum_rew[-1][i][-1]
            self.mean_cum_rew /= float(self.e_num_samples.get())
            self.mean_rew = self.mean_cum_rew / (self.t_term * self.control_freq)

            # reward graph
            self.rew_graph.cla()
            self.rew_graph.set_title('Reward')
            min_rew = min([min(sublist) for sublist in self.rew[-1]])
            max_rew = max([max(sublist) for sublist in self.rew[-1]])
            mid_rew = (max_rew + min_rew) / 2
            self.rew_y_min = mid_rew - (mid_rew - min_rew) * 1.1
            self.rew_y_max = mid_rew + (max_rew - mid_rew) * 1.1
            self.rew_graph.set_xlim(0, self.t_term)
            self.rew_x_axis = self.rew_graph.plot(numpy.array([0, self.t_term]), numpy.array([0, 0]), 'k')[0]  # x axis
            self.rew_graph.set_ylim(self.rew_y_min, self.rew_y_max)
            text_x = 8 * float(self.t_term) / 10
            text_y = float(self.rew_y_max - self.rew_y_min) / 10 + float(self.rew_y_min)
            self.rew_label = self.rew_graph.text(text_x, text_y, 'ave rew: %.2f' % (self.mean_rew))

            # cumulative reward graph
            self.cum_rew_graph.cla()
            self.cum_rew_graph.set_xlabel('Time (s)')
            self.cum_rew_graph.set_title('Cumulative Reward')
            min_cum_rew = min([min(sublist) for sublist in self.cum_rew[-1]])
            max_cum_rew = max([max(sublist) for sublist in self.cum_rew[-1]])
            mid_cum_rew = (max_cum_rew + min_cum_rew) / 2
            self.cum_rew_y_min = mid_cum_rew - (mid_cum_rew - min_cum_rew) * 1.1
            self.cum_rew_y_max = mid_cum_rew + (max_cum_rew - mid_cum_rew) * 1.1
            self.cum_rew_graph.set_xlim(0, self.t_term)
            self.cum_rew_x_axis = self.cum_rew_graph.plot(numpy.array([0, self.t_term]), numpy.array([0, 0]), 'k')[0]  # x axis
            self.cum_rew_graph.set_ylim(self.cum_rew_y_min, self.cum_rew_y_max)
            text_x = float(self.t_term) / 25
            text_y = float(self.cum_rew_y_max - self.cum_rew_y_min) / 10 + float(self.cum_rew_y_min)
            self.cum_rew_label = self.cum_rew_graph.text(text_x, text_y, 'ave cum-rew: %.2f' % (self.mean_cum_rew))

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
            self.vel_v_lines =[]
            self.acc_lines = []
            self.acc_t_lines = []
            self.rew_lines = [[] for i in range(0, self.num_rew_terms)]
            self.rew_t_lines = []
            self.cum_rew_lines = [[] for i in range(0, self.num_rew_terms)]
            self.cum_rew_t_lines = []
            self.policy_lines = []

            for i in range(0,int(self.e_num_samples.get())):

                self.pos_meas_lines.append(self.pos_graph.plot(self.time[i], self.pos_meas[i], 'm', marker='x')[0])
                self.pos_lines.append(self.pos_graph.plot(self.time[i], self.pos[i], 'b', marker='o', markeredgecolor='w')[0])
                self.pos_t_lines.append(self.pos_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]), numpy.array([pos_y_min, pos_y_max]), 'r--')[0])

                self.vel_lines.append(self.vel_graph.plot(self.time[i], self.vel[i], 'r', marker='o', markeredgecolor='w')[0])
                self.vel_t_lines.append(self.vel_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]), numpy.array([vel_y_min, vel_y_max]), 'r--')[0])
                self.vel_v_lines.append(self.vel_graph.plot(numpy.array([0, max(self.t_touch)]), numpy.array([self.v_touch[i], self.v_touch[i]]), 'r--')[0])

                self.acc_lines.append(self.acc_graph.plot(self.time[i], self.acc[i], 'g', marker='o', markeredgecolor='w')[0])
                self.acc_t_lines.append(self.acc_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]), numpy.array([self.acc_y_min, self.acc_y_max]), 'r--')[0])

                self.rew_lines[0].append(self.rew_graph.plot(self.time[i], self.rew[0][i], 'b', marker='o', markeredgecolor='w')[0])
                self.rew_lines[1].append(self.rew_graph.plot(self.time[i], self.rew[1][i], 'r', marker='o', markeredgecolor='w')[0])
                self.rew_lines[2].append(self.rew_graph.plot(self.time[i], self.rew[2][i], 'g', marker='o', markeredgecolor='w')[0])
                self.rew_lines[3].append(self.rew_graph.plot(self.time[i], self.rew[3][i], 'k', marker='o', markeredgecolor='w')[0])
                self.rew_t_lines.append(self.rew_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]),numpy.array([self.rew_y_min, self.rew_y_max]), 'r--')[0])

                self.cum_rew_lines[0].append(self.cum_rew_graph.plot(self.time[i], self.cum_rew[0][i], 'b', marker='o', markeredgecolor='w')[0])
                self.cum_rew_lines[1].append(self.cum_rew_graph.plot(self.time[i], self.cum_rew[1][i], 'r', marker='o', markeredgecolor='w')[0])
                self.cum_rew_lines[2].append(self.cum_rew_graph.plot(self.time[i], self.cum_rew[2][i], 'g', marker='o', markeredgecolor='w')[0])
                self.cum_rew_lines[3].append(self.cum_rew_graph.plot(self.time[i], self.cum_rew[3][i], 'k', marker='o', markeredgecolor='w')[0])
                self.cum_rew_t_lines.append(self.cum_rew_graph.plot(numpy.array([self.t_touch[i], self.t_touch[i]]),numpy.array([self.cum_rew_y_min, self.cum_rew_y_max]), 'r--')[0])

                self.policy_lines.append(self.policy_graph.plot(self.vel[i], self.pos[i], color='k', linestyle='-', linewidth=3, marker='o', markeredgecolor='w')[0])

        else:

            if (self.init_pos == float(self.e_max_pos.get())):

                    # max pos required in condition to prevent long trajectories from small starting positions
                    self.t_term = numpy.nan
                    if self.dof_comp.get() == 0:
                        self.t_term = max(self.t_touch)
                    elif self.dof_comp.get() == 1 or self.dof_comp.get() == 2:
                        self.t_term = self.t_max
                    self.pos_graph.set_xlim(0, self.t_term)
                    self.vel_graph.set_xlim(0, self.t_term)
                    self.acc_graph.set_xlim(0, self.t_term)
                    self.rew_graph.set_xlim(0, self.t_term)
                    self.cum_rew_graph.set_xlim(0, self.t_term)

                    # acceleration y limits
                    self.acc_x_axis.set_xdata(numpy.array([0, self.t_term]))
                    min_acc = min([min(sublist) for sublist in self.acc])
                    max_acc = max([max(sublist) for sublist in self.acc])
                    mid_acc = (max_acc + min_acc) / 2
                    self.acc_y_min = mid_acc - (mid_acc - min_acc) * 1.1
                    self.acc_y_max = mid_acc + (max_acc - mid_acc) * 1.1
                    self.acc_graph.set_ylim(self.acc_y_min, self.acc_y_max)

                    # reward y limits
                    self.rew_x_axis.set_xdata(numpy.array([0, self.t_term]))
                    min_rew = min([min(sublist) for sublist in self.rew[-1]])
                    max_rew = max([max(sublist) for sublist in self.rew[-1]])
                    mid_rew = (max_rew + min_rew) / 2
                    self.rew_y_min = mid_rew - (mid_rew - min_rew) * 1.1
                    self.rew_y_max = mid_rew + (max_rew - mid_rew) * 1.1
                    self.rew_graph.set_ylim(self.rew_y_min, self.rew_y_max)

                    # cumulative reward y limits
                    self.cum_rew_x_axis.set_xdata(numpy.array([0, self.t_term]))
                    min_cum_rew = min([min(sublist) for sublist in self.cum_rew[-1]])
                    max_cum_rew = max([max(sublist) for sublist in self.cum_rew[-1]])
                    mid_cum_rew = (max_cum_rew + min_cum_rew) / 2
                    self.cum_rew_y_min = mid_cum_rew - (mid_cum_rew - min_cum_rew) * 1.1
                    self.cum_rew_y_max = mid_cum_rew + (max_cum_rew - mid_cum_rew) * 1.1
                    self.cum_rew_graph.set_ylim(self.cum_rew_y_min, self.cum_rew_y_max)

                    for i in range(0, int(self.e_num_samples.get())):
                        self.vel_v_lines[i].set_xdata(numpy.array([0, self.t_term]))

            # trajectories

            for i in range(0,int(self.e_num_samples.get())):

                # position lines
                self.pos_lines[i].set_xdata(self.time[i])
                self.pos_lines[i].set_ydata(self.pos[i])
                self.pos_meas_lines[i].set_xdata(self.time[i])
                self.pos_meas_lines[i].set_ydata(self.pos_meas[i])
                self.pos_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                # velocity lines
                self.vel_lines[i].set_xdata(self.time[i])
                self.vel_lines[i].set_ydata(self.vel[i])
                self.vel_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))
                self.vel_v_lines[i].set_ydata(numpy.array([self.v_touch[i], self.v_touch[i]]))

                # update accel y lims
                min_acc = min([min(sublist) for sublist in self.acc])
                max_acc = max([max(sublist) for sublist in self.acc])
                mid_acc = (max_acc + min_acc) / 2
                if (mid_acc - (mid_acc - min_acc) * 1.1 < self.acc_y_min):
                    self.acc_y_min = mid_acc - (mid_acc - min_acc) * 1.1
                if (mid_acc + (max_acc - mid_acc) * 1.1 > self.acc_y_max):
                    self.acc_y_max = mid_acc + (max_acc - mid_acc) * 1.1
                self.acc_graph.set_ylim(self.acc_y_min, self.acc_y_max)

                # acceleration lines
                self.acc_lines[i].set_xdata(self.time[i])
                self.acc_lines[i].set_ydata(self.acc[i])
                self.acc_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                # re-calculate mean cum and instant rewards
                self.mean_cum_rew = 0
                for j in range(0, int(self.e_num_samples.get())):
                    self.mean_cum_rew += float(self.cum_rew[-1][j][-1])
                self.mean_cum_rew /= float(self.e_num_samples.get())
                self.mean_rew = self.mean_cum_rew / (self.t_term*self.control_freq)

                # update rew y lims
                min_rew = min([min(sublist) for sublist in self.rew[-1]])
                max_rew = max([max(sublist) for sublist in self.rew[-1]])
                mid_rew = (max_rew + min_rew) / 2
                if (mid_rew - (mid_rew - min_rew) * 1.1 < self.rew_y_min):
                    self.rew_y_min = mid_rew - (mid_rew - min_rew) * 1.1
                if (mid_rew + (max_rew - mid_rew) * 1.1 > self.rew_y_max):
                    self.rew_y_max = mid_rew + (max_rew - mid_rew) * 1.1
                self.rew_graph.set_ylim(self.rew_y_min, self.rew_y_max)

                # reward lines
                for j in range(0,self.num_rew_terms):
                    self.rew_lines[j][i].set_xdata(self.time[i])
                    self.rew_lines[j][i].set_ydata(self.rew[j][i])
                    self.rew_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                # reward label
                text_x = 8 * float(self.t_term) / 10
                text_y = float(self.rew_y_max - self.rew_y_min) / 10 + float(self.rew_y_min)
                self.rew_label.set_text('av rew: %.2f' % (self.mean_rew))
                self.rew_label.set_position((text_x, text_y))

                # update cum rew y lims
                min_cum_rew = min([min(sublist) for sublist in self.cum_rew[-1]])
                max_cum_rew = max([max(sublist) for sublist in self.cum_rew[-1]])
                mid_cum_rew = (max_cum_rew + min_cum_rew) / 2
                if (mid_cum_rew - (mid_cum_rew - min_cum_rew) * 1.1 < self.cum_rew_y_min):
                    self.cum_rew_y_min = mid_cum_rew - (mid_cum_rew - min_cum_rew) * 1.1
                if (mid_cum_rew + (max_cum_rew - mid_cum_rew) * 1.1 > self.cum_rew_y_max):
                    self.cum_rew_y_max = mid_cum_rew + (max_cum_rew - mid_cum_rew) * 1.1
                self.cum_rew_graph.set_ylim(self.cum_rew_y_min, self.cum_rew_y_max)

                # cumulative reward lines
                for j in range(0,self.num_rew_terms):
                    self.cum_rew_lines[j][i].set_xdata(self.time[i])
                    self.cum_rew_lines[j][i].set_ydata(self.cum_rew[j][i])
                    self.cum_rew_t_lines[i].set_xdata(numpy.array([self.t_touch[i], self.t_touch[i]]))

                # cum reward label
                text_x = float(self.t_term) / 25
                text_y = float(self.cum_rew_y_max - self.cum_rew_y_min) / 10 + float(self.cum_rew_y_min)
                self.cum_rew_label.set_text("av cum-rew: %.2f" % (self.mean_cum_rew))
                self.cum_rew_label.set_position((text_x, text_y))

                # policy map
                self.policy_lines[i].set_xdata(self.vel[i])
                self.policy_lines[i].set_ydata(self.pos[i])

        self.traj_fig.canvas.draw()
        self.policy_fig.canvas.draw()

    def update_gui(self, event = 'dummy'):

        # dof mode
        if self.dof_comp.get() == 0:
            if event == 'dof selection':
                self.e_t_max.delete(0, tkinter.END)
                self.e_t_max.insert(tkinter.END, '')
                self.e_t_max.config(state='disable')

                self.e_max_pos.config(state='normal')
                self.e_max_pos.delete(0, tkinter.END)
                self.e_max_pos.insert(tkinter.END, '50')

                self.l_min_pos.config(text='min pos (cm)')
                self.l_max_pos.config(text ='max pos (cm)')

                self.l_min_vel.config(text='min vel (cm/s)')
                self.l_max_vel.config(text='max vel (cm/s)')

            self.compute_resolutions()
            self.sv_pos_res.set('      pos res:        %.2f   cm    ' % self.pos_res)
            self.sv_vel_res.set('      vel res:        %.2f   cm/s    ' % self.vel_res)
            self.sv_control_freq.set('      control freq:        %.2f   Hz    ' % self.control_freq)

        elif self.dof_comp.get() == 1:
            if event == 'dof selection':
                self.e_t_max.config(state='normal')
                self.e_t_max.delete(0, tkinter.END)
                self.e_t_max.insert(tkinter.END, '5')

                self.e_max_pos.config(state='normal')
                self.e_max_pos.delete(0, tkinter.END)
                self.e_max_pos.insert(tkinter.END, '50')

                self.e_t_max.config(state='normal')

                self.l_min_pos.config(text='min pos (cm)')
                self.l_max_pos.config(text ='max pos (cm)')

                self.l_min_vel.config(text='min vel (cm/s)')
                self.l_max_vel.config(text='max vel (cm/s)')

            self.t_max = int(self.e_t_max.get())

            self.compute_resolutions()
            self.sv_pos_res.set('      pos res:        %.2f   cm    ' % self.pos_res)
            self.sv_vel_res.set('      vel res:        %.2f   cm/s    ' % self.vel_res)
            self.sv_control_freq.set('      control freq:        %.2f   Hz    ' % self.control_freq)

        elif self.dof_comp.get() == 2:
            if event == 'dof selection':

                self.e_t_max.config(state='normal')
                self.e_t_max.delete(0, tkinter.END)
                self.e_t_max.insert(tkinter.END, '15')

                self.e_max_pos.delete(0, tkinter.END)
                self.e_max_pos.insert(tkinter.END, '180')
                self.e_max_pos.config(state='disable')

                self.e_t_max.config(state='normal')

                self.l_min_pos.config(text='min pos (deg)')
                self.l_max_pos.config(text='max pos (deg)')

                self.l_min_vel.config(text='min vel (deg/s)')
                self.l_max_vel.config(text='max vel (deg/s)')

            self.t_max = int(self.e_t_max.get())

            self.compute_resolutions()
            self.sv_pos_res.set('      pos res:        %.2f   deg    ' % self.pos_res)
            self.sv_vel_res.set('      vel res:        %.2f   deg/s    ' % self.vel_res)
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

        # perform updates
        self.tk_root.update_idletasks()

    def reset_gui(self):

        self.e_min_pos.delete(0, tkinter.END)
        self.e_max_pos.delete(0, tkinter.END)
        self.e_num_positions.delete(0, tkinter.END)
        self.e_min_vel.delete(0, tkinter.END)
        self.e_max_vel.delete(0, tkinter.END)
        self.e_num_actions.delete(0, tkinter.END)
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

        self.dof_comp.set(0)
        self.e_min_pos.insert(tkinter.END, '0')
        self.e_min_pos.config(state='disable')
        self.e_max_pos.insert(tkinter.END, '50')
        self.e_num_positions.insert(tkinter.END, '126')
        self.e_min_vel.insert(tkinter.END, '0')
        self.e_max_vel.insert(tkinter.END, '20')
        self.e_num_actions.insert(tkinter.END, '6')
        self.e_discount.insert(tkinter.END, '0.99')
        self.e_epsilon.insert(tkinter.END, '0.01')
        self.e_max_iter.insert(tkinter.END, '1000')
        self.e_dist_factor.insert(tkinter.END, '1')
        self.e_vel_factor.insert(tkinter.END, '1')
        self.e_vel_den_ratio.insert(tkinter.END, '0.05')
        self.e_acc_factor.insert(tkinter.END, '1')
        self.e_motion_bins.insert(tkinter.END, '15')
        self.e_pos_meas_bins.insert(tkinter.END, '15')
        self.e_num_samples.insert(tkinter.END, '5')
        self.e_t_max.insert(tkinter.END, '3')
        self.sample_w_motion_noise.set(1)
        self.sample_w_pos_meas_noise.set(1)

        #self.init_pos = float(self.e_max_pos.get()) # TRYING TO FIX SLIDER RESET
        #self.init_vel = float(self.e_min_vel.get())

        self.update_gui()

    def compute_vi(self):

        # transition matrix
        transitions = numpy.zeros((self.num_actions, self.num_states, self.num_states))
        no_vel_trans = numpy.zeros((self.num_states, self.num_states))

        # setup base zero velocity transition matrix, for vel = 0
        for i in range(0, int(self.num_states / self.num_actions)): # iterate over pos states
            for j in range(0,self.num_motion_bins): # iterate over bins
                bin_range = (j - self.motion_centre_idx) * self.num_actions
                horizontal_coord = i * self.num_actions + bin_range
                no_vel_trans[i * self.num_actions:i * self.num_actions + self.num_actions, horizontal_coord % self.num_states] = numpy.full(self.num_actions, self.crude_motion_hist[j])

        # set up full transition matrix based on position state
        for i in range(0, self.num_actions):

            # roll entire state
            transitions[i, :, :] = numpy.roll(no_vel_trans[:,:], i * self.num_actions + i, 1)

            #re-normalise border at top
            if self.dof_comp.get() == 0 or self.dof_comp.get() == 1: # z or x-y mode
                if (i < self.motion_centre_idx):  # if there are border terms
                    for j in range(0, self.motion_centre_idx - i): # iterate over groups
                        # top-left (starting state border)
                        transitions[i,j*self.num_actions:(j+1)*self.num_actions, i] += numpy.sum(self.crude_motion_hist[0:self.motion_centre_idx - j - i])
                        # top-right (initial rolled over)
                        start = self.num_states - (self.motion_centre_idx - i) * self.num_actions + i
                        transitions[i, 0:(j + 1) * self.num_actions, start + j*self.num_actions] = 0
            elif self.dof_comp.get() == 2: # ang mode
                if (i < self.motion_centre_idx):  # if there are border terms
                    for j in range(0, self.motion_centre_idx - i): # iterate over groups
                        start = self.num_states - (self.motion_centre_idx - i) * self.num_actions + i
                        # mirror rollover terms about lhs border matrix
                        transitions[i, 0:(j + 1) * self.num_actions, -(start-i+j*self.num_actions) % self.num_states + i] += \
                            transitions[i, 0:(j + 1) * self.num_actions, start + j * self.num_actions]
                        transitions[i, 0:(j + 1) * self.num_actions, start + j*self.num_actions] = 0


            #re-normalise border at bottom
            if self.dof_comp.get() == 0: # z mode
                for j in range(0, self.motion_centre_idx + i):
                    # bottom-right (final state border)
                    start_j = self.num_states+ (j - self.motion_centre_idx - i) * self.num_actions
                    end_j = self.num_states+ (j + 1 - self.motion_centre_idx - i) * self.num_actions
                    transitions[i,start_j:end_j,self.num_states-self.num_actions+i] += numpy.sum(self.crude_motion_hist[0:j + 1])\
                                                                    if j<float(self.e_motion_bins.get()) else 1
                    # bottom-right (final state-rolled over)
                    transitions[i,start_j:self.num_states, i+j*self.num_actions] = 0

            elif self.dof_comp.get() == 1 or self.dof_comp.get() == 2: # x-y or ang mode
                for j in range(0, self.motion_centre_idx + i):
                    start_j = self.num_states+ (j - self.motion_centre_idx - i) * self.num_actions
                    # mirror rollover terms about rhs border of matrix
                    transitions[i, start_j:self.num_states, (i-(j+2) * self.num_actions) % self.num_states] += \
                        transitions[i, start_j:self.num_states, i + j * self.num_actions]
                    transitions[i, start_j:self.num_states, i + j * self.num_actions] = 0

        # reward matrix
        self.num_rew_terms = 3 + 1 # 3 terms, and one final unified
        reward_entry_mats = []
        self.reward_mats = []

        for i in range(0,self.num_rew_terms):
            reward_entry_mats.append(numpy.zeros((self.num_pos_meas_bins)))
            self.reward_mats.append(numpy.zeros((self.num_states, self.num_actions)))

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
                    reward_entry_mats[0][pos_it] = -float(self.e_dist_factor.get())*dist_to_target
                    reward_entry_mats[1][pos_it] = -float(self.e_vel_factor.get()) * vel_over_dist
                    reward_entry_mats[2][pos_it] = -float(self.e_acc_factor.get()) * delta_vel_sqaured
                    reward_entry_mats[-1][pos_it] = reward_entry_mats[0][pos_it] + reward_entry_mats[1][pos_it] + reward_entry_mats[2][pos_it]


                # total reward of being in this state, given possibilities
                self.reward_mats[0][i, j] = numpy.sum(numpy.dot(reward_entry_mats[0], self.crude_pos_meas_hist))
                self.reward_mats[1][i, j] = numpy.sum(numpy.dot(reward_entry_mats[1], self.crude_pos_meas_hist))
                self.reward_mats[2][i, j] = numpy.sum(numpy.dot(reward_entry_mats[2], self.crude_pos_meas_hist))
                self.reward_mats[-1][i, j] = numpy.sum(numpy.dot(reward_entry_mats[-1], self.crude_pos_meas_hist))


        # normalise total reward mats
        divisor = abs(numpy.sum(self.reward_mats[-1]) / self.reward_mats[-1].size)
        self.reward_mats[0] /= divisor
        self.reward_mats[1] /= divisor
        self.reward_mats[2] /= divisor
        self.reward_mats[-1] /= divisor


        # define value iteration
        self.vi = mdptoolbox.mdp.ValueIteration(transitions, self.reward_mats[-1], float(self.e_discount.get()), float(self.e_epsilon.get()), float(self.e_max_iter.get()), 0)
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
        self.pos = [[] for i in range(int(self.e_num_samples.get()))]
        self.pos_meas = [[] for i in range(int(self.e_num_samples.get()))]
        self.vel = [[] for i in range(int(self.e_num_samples.get()))]
        self.acc = [[] for i in range(int(self.e_num_samples.get()))]
        self.rew = [[[] for i in range(int(self.e_num_samples.get()))] for j in range(0,self.num_rew_terms)]
        self.cum_rew = [[[] for i in range(int(self.e_num_samples.get()))] for j in range(0,self.num_rew_terms)]
        self.t_touch = []
        self.v_touch = []

        # iterate over n samples
        for i in range(0,int(self.e_num_samples.get())):

            # initialise state
            prev_state = [numpy.nan,numpy.nan]
            state = [int(float(self.s_init_pos.get()) / self.pos_res), int(float(self.s_init_vel.get()) / self.vel_res)]
            t = 0

            # populate trajectories
            at_target = False
            traj_end = False

            while traj_end is False:

                # z dof mode
                if self.dof_comp.get() == 0:
                    if state[0] == 0:
                        self.t_touch.append(t)
                        self.v_touch.append(state[1]*self.vel_res)
                        at_target = True
                        traj_end = True

                    if (self.sample_w_motion_noise.get() == 0 and self.sample_w_pos_meas_noise.get() == 0 \
                                and prev_state == state and at_target == False):
                        self.t_touch.append(0)
                        self.v_touch.append(0)
                        traj_end = True

                # x-y and ang dof mode
                if self.dof_comp.get() == 1 or self.dof_comp.get() == 2:
                    if t > self.t_max:
                        self.t_touch.append(0)
                        self.v_touch.append(0)
                        traj_end = True

                # measurement
                if self.sample_w_pos_meas_noise.get() == 1:
                    pos_meas = state[0] + numpy.random.choice(numpy.linspace(-self.pos_meas_centre_idx, self.pos_meas_centre_idx, self.num_pos_meas_bins), 1, p=self.crude_pos_meas_hist)
                else:
                    pos_meas = state[0]

                # clip measurements at borders
                if self.dof_comp.get() == 0:
                    if pos_meas < int(float(self.e_min_pos.get())/self.pos_res):
                        pos_meas = int(float(self.e_min_pos.get())/self.pos_res)
                    elif pos_meas > int(float(self.e_max_pos.get())/self.pos_res):
                        pos_meas = int(float(self.e_max_pos.get())/self.pos_res)
                elif self.dof_comp.get() == 1:
                    if pos_meas > int(float(self.e_max_pos.get())/self.pos_res):
                        pos_meas = int(float(self.e_max_pos.get()) / self.pos_res)
                    elif pos_meas < -int(float(self.e_max_pos.get())/self.pos_res):
                        pos_meas = -int(float(self.e_max_pos.get()) / self.pos_res)
                elif self.dof_comp.get() == 2:
                    pos_meas = int(pos_meas+float(self.e_max_pos.get())/self.pos_res) % int(2*float(self.e_max_pos.get())/self.pos_res) - int(float(self.e_max_pos.get())/self.pos_res)
                    if pos_meas == int(-float(self.e_max_pos.get())/self.pos_res):
                        pos_meas = int(float(self.e_max_pos.get())/self.pos_res) # to favor 180 over -180
                state_meas = [pos_meas, state[1]]

                policy_state = [abs(state_meas[0]), abs(state_meas[1])]

                target_vel = self.vi.policy[int(self.num_states - policy_state[0] * self.num_actions - self.num_actions + policy_state[1])]

                # correct velocity sign
                if pos_meas < 0:
                    target_vel *= -1

                # update raw trajectory arrays
                self.pos_raw[i].append(state[0]) # current pos
                self.pos_meas_raw[i].append(state_meas[0]) # current measured pos
                self.vel_raw[i].append(state[1]) # current velocity
                self.acc_raw[i].append(target_vel-state[1]) # current accel
                self.time[i].append(t)

                # update reward arrays
                for j in range(0,self.num_rew_terms):
                    self.rew[j][i].append(self.reward_mats[j][int(self.num_states - policy_state[0] * self.num_actions - self.num_actions + policy_state[1]), abs(target_vel)])
                    self.cum_rew[j][i].append(self.cum_rew[j][i][-1] + self.rew[j][i][-1]) if t > 0 else  self.cum_rew[j][i].append(self.rew[j][i][-1])

                # motion noise
                if self.sample_w_motion_noise.get() == 1:
                    new_pos = state[0] - target_vel + numpy.random.choice(numpy.linspace(-self.motion_centre_idx, self.motion_centre_idx, self.num_motion_bins), 1, p=self.crude_motion_hist)
                else:
                    new_pos = state[0] - target_vel

                # new position border clipping
                if self.dof_comp.get() == 0:
                    if new_pos < int(float(self.e_min_pos.get())/self.pos_res):
                        new_pos = int(float(self.e_min_pos.get())/self.pos_res)
                    elif new_pos > int(float(self.e_max_pos.get())/self.pos_res):
                        new_pos = int(float(self.e_max_pos.get())/self.pos_res)
                elif self.dof_comp.get() == 1:
                    if new_pos > int(float(self.e_max_pos.get())/self.pos_res):
                        new_pos = int(float(self.e_max_pos.get())/self.pos_res)
                    elif new_pos < -int(float(self.e_max_pos.get())/self.pos_res):
                        new_pos = -int(float(self.e_max_pos.get()) / self.pos_res)
                elif self.dof_comp.get() == 2:
                    new_pos = int(new_pos+float(self.e_max_pos.get())/self.pos_res) % int(2*float(self.e_max_pos.get())/self.pos_res) - int(float(self.e_max_pos.get())/self.pos_res)
                    if new_pos == int(-float(self.e_max_pos.get())/self.pos_res):
                        new_pos = int(float(self.e_max_pos.get())/self.pos_res) # to favor 180 over -180

                prev_state = state
                state = [new_pos, target_vel]

                # iterate time
                t += time_step

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

        # DOF Component #
        #---------------#

        column += 3

        l_dof_comp = tkinter.Label(f_tb, text='DOF Component')
        l_dof_comp.grid(row=row, column=column)

        row += 1
        column -= 3

        l_z = tkinter.Label(f_tb, text='z:')
        l_z.grid(row=row, column=column)

        column += 1

        self.dof_comp = tkinter.IntVar()
        self.r_dof_comp_z = tkinter.Radiobutton(f_tb, variable=self.dof_comp, value=0)
        self.r_dof_comp_z.grid(row=row, column=column)

        column += 1

        l_xy = tkinter.Label(f_tb, text='x-y:')
        l_xy.grid(row=row, column=column)

        column += 1

        self.r_dof_comp_xy = tkinter.Radiobutton(f_tb, variable=self.dof_comp, value=1)
        self.r_dof_comp_xy.grid(row=row, column=column)

        column += 1

        l_ang = tkinter.Label(f_tb, text='ang:')
        l_ang.grid(row=row, column=column)

        column += 1

        self.r_dof_comp_ang = tkinter.Radiobutton(f_tb, variable=self.dof_comp, value=2)
        self.r_dof_comp_ang.grid(row=row, column=column)

        row += 1
        column -= 5

        # Position Parameters #
        #---------------------#

        column += 3

        l_pos = tkinter.Label(f_tb, text='Position Parameters')
        l_pos.grid(row=row, column=column)

        row += 1
        column -= 3

        self.l_min_pos = tkinter.Label(f_tb)
        self.l_min_pos.grid(row=row, column=column)

        column += 1

        self.e_min_pos = tkinter.Entry(f_tb, width=5)
        self.e_min_pos.grid(row=row, column=column)
        self.e_min_pos.bind("<FocusOut>", self.update_gui)

        column += 1

        self.l_max_pos = tkinter.Label(f_tb)
        self.l_max_pos.grid(row=row, column=column)

        column += 1

        self.e_max_pos = tkinter.Entry(f_tb, width=5)
        self.e_max_pos.grid(row=row, column=column)
        self.e_max_pos.bind("<FocusOut>", self.update_gui)

        column += 1

        l_num_positions = tkinter.Label(f_tb, text='num positions')
        l_num_positions.grid(row=row, column=column)

        column += 1

        self.e_num_positions = tkinter.Entry(f_tb, width=5)
        self.e_num_positions.grid(row=row, column=column)
        self.e_num_positions.bind("<FocusOut>", self.update_gui)

        column += 1

        self.sv_pos_res = tkinter.StringVar()
        l_pos_res = tkinter.Label(f_tb, textvariable=self.sv_pos_res)
        l_pos_res.grid(row=row, column=column)

        row += 1
        column -= 6

        # Velocity Parameters #
        #---------------------#

        column += 3

        l_vel = tkinter.Label(f_tb, text='velocity Parameters')
        l_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        self.l_min_vel = tkinter.Label(f_tb)
        self.l_min_vel.grid(row=row, column=column)

        column += 1

        self.e_min_vel = tkinter.Entry(f_tb, width=5)
        self.e_min_vel.grid(row=row, column=column)
        self.e_min_vel.bind("<FocusOut>", self.update_gui)

        column += 1

        self.l_max_vel = tkinter.Label(f_tb)
        self.l_max_vel.grid(row=row, column=column)

        column += 1

        self.e_max_vel = tkinter.Entry(f_tb, width=5)
        self.e_max_vel.grid(row=row, column=column)
        self.e_max_vel.bind("<FocusOut>", self.update_gui)

        column += 1

        l_num_actions = tkinter.Label(f_tb, text='num actions')
        l_num_actions.grid(row=row, column=column)

        column += 1

        self.e_num_actions = tkinter.Entry(f_tb, width=5)
        self.e_num_actions.grid(row=row, column=column)
        self.e_num_actions.bind("<FocusOut>", self.update_gui)

        column += 1

        self.sv_vel_res = tkinter.StringVar()
        l_vel_res = tkinter.Label(f_tb, textvariable=self.sv_vel_res)
        l_vel_res.grid(row=row, column=column)

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

        self.e_discount = tkinter.Entry(f_tb, width=5)
        self.e_discount.grid(row=row, column=column)

        column += 1

        l_epsilon = tkinter.Label(f_tb, text='epsilon')
        l_epsilon.grid(row=row, column=column)

        column += 1

        self.e_epsilon = tkinter.Entry(f_tb, width=5)
        self.e_epsilon.grid(row=row, column=column)

        column += 1

        l_max_iter = tkinter.Label(f_tb, text='max iter')
        l_max_iter.grid(row=row, column=column)

        column += 1

        self.e_max_iter = tkinter.Entry(f_tb, width=5)
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

        self.e_dist_factor = tkinter.Entry(f_tb, width=5)
        self.e_dist_factor.grid(row=row, column=column)

        column += 1

        l_vel_factor = tkinter.Label(f_tb, text='vel factor')
        l_vel_factor.grid(row=row, column=column)

        column += 1

        self.e_vel_factor = tkinter.Entry(f_tb, width=5)
        self.e_vel_factor.grid(row=row, column=column)

        column += 1

        l_vel_den_ratio = tkinter.Label(f_tb, text='vel den ratio')
        l_vel_den_ratio.grid(row=row, column=column)

        column += 1

        self.e_vel_den_ratio = tkinter.Entry(f_tb, width=5)
        self.e_vel_den_ratio.grid(row=row, column=column)

        column += 1

        l_acc_factor = tkinter.Label(f_tb, text='acc^2 factor')
        l_acc_factor.grid(row=row, column=column)

        column += 1

        self.e_acc_factor = tkinter.Entry(f_tb, width=5)
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

        self.e_motion_bins = tkinter.Entry(f_tb, width=5)
        self.e_motion_bins.grid(row=row, column=column)

        column += 1

        l_pos_meas_bins = tkinter.Label(f_tb, text='pos meas bins')
        l_pos_meas_bins.grid(row=row, column=column)

        column += 1

        self.e_pos_meas_bins = tkinter.Entry(f_tb, width=5)
        self.e_pos_meas_bins.grid(row=row, column=column)

        row += 1
        column -= 3

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

        self.e_num_samples = tkinter.Entry(f_tb, width=5)
        self.e_num_samples.grid(row=row, column=column)

        column += 1

        self.l_t_max = tkinter.Label(f_tb, text='t max')
        self.l_t_max.grid(row=row, column=column)

        column += 1

        self.e_t_max = tkinter.Entry(f_tb, width=5)
        self.e_t_max.grid(row=row, column=column)

        column += 1

        l_sample_w_motion_noise = tkinter.Label(f_tb, text='sample with motion noise?')
        l_sample_w_motion_noise.grid(row=row, column=column)

        column += 1

        self.sample_w_motion_noise = tkinter.IntVar()
        self.c_sample_w_motion_noise = tkinter.Checkbutton(f_tb, text="yes", variable=self.sample_w_motion_noise)
        self.c_sample_w_motion_noise.grid(row=row, column=column)

        column += 1

        l_sample_w_pos_meas_noise = tkinter.Label(f_tb, text='sample with pos meas noise?')
        l_sample_w_pos_meas_noise.grid(row=row, column=column)

        column += 1

        self.sample_w_pos_meas_noise = tkinter.IntVar()
        self.c_sample_w_pos_meas_noise = tkinter.Checkbutton(f_tb, text="yes", variable=self.sample_w_pos_meas_noise)
        self.c_sample_w_pos_meas_noise.grid(row=row, column=column)

        row += 1
        column -= 7

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

        #column += 1

        l_plots = tkinter.Label(f_p, text='Plots')
        l_plots.grid(row=row, column=column)

        row += 1
        #column -= 1

        #self.c_traj_plot = tkagg.FigureCanvasTkAgg(self.traj_fig, master=f_p)
        #self.c_traj_plot.get_tk_widget().grid(row=row, column=column)
        #self.c_traj_plot._tkcanvas.grid(row=row, column=column) # not sure if needed?

        #column += 2

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
        # set up mode selector for gui updates
        self.r_dof_comp_z.config(command = lambda: self.update_gui('dof selection'))
        self.r_dof_comp_xy.config(command = lambda: self.update_gui('dof selection'))
        self.r_dof_comp_ang.config(command = lambda: self.update_gui('dof selection'))

        # start mainloop #
        #----------------#

        self.tk_root.mainloop()

def main():

    tk_interface = TK_Interface()

    # start tkinter gui
    threading.Thread(target=tk_interface.tkinter_loop).start()

if __name__ == "__main__":
    main()