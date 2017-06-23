import os
import mdptoolbox
import numpy
import math
import tkinter
import threading
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import datetime
import transforms3d

numpy.set_printoptions(linewidth=1000,threshold=numpy.nan)

class TK_Interface:

    # Core Functions #
    #----------------#

    def compute_ef_coords(self):

        origin = numpy.array([0, 0, 0, 1])

        dx = 0.5
        dy = 0.5
        dz = 0.5

        T = numpy.array([dx, dy, dz])

        roll = 0
        pitch = 0
        yaw = 0

        R = transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')

        Z = numpy.array([1,1,1])

        Affine_Mat = transforms3d.affines.compose(T, R, Z)

        Ef_coords = Affine_Mat * origin

        print(Ef_coords)

    def init_plot(self):

        plt.close()

        plt.ion()

        # trajectory plots
        self.traj_fig = plt.figure()
        self.traj_graph = self.traj_fig.add_subplot(1,1,1, projection='3d')
        self.traj_graph.set_xlim(0,1)
        self.traj_graph.set_ylim(0,1)
        self.traj_graph.set_zlim(0,1)

        self.EF_x_line = self.traj_graph.plot()

        plt.ioff()

        self.traj_fig.tight_layout()  # turn on tight-layout

        self.plot_samples = 0

    def update_plot(self, recomputed_flag):

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
            min_rew = min([min([min(sublist) for sublist in sublist_1]) for sublist_1 in self.rew])
            max_rew = max([max([max(sublist) for sublist in sublist_1]) for sublist_1 in self.rew])
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
            min_cum_rew = min([min([min(sublist) for sublist in sublist_1]) for sublist_1 in self.cum_rew])
            max_cum_rew = max([max([max(sublist) for sublist in sublist_1]) for sublist_1 in self.cum_rew])
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
            self.discrete_policy_map = numpy.zeros((self.num_pos_states, self.num_actions))
            for j in range(self.num_pos_states):
                for k in range(self.num_actions):
                    self.discrete_policy_map[j, k] = self.vi.policy[
                        int(self.num_states - j * self.num_actions - self.num_actions + k)]

            if float(self.policy_mode.get()) == 0:
                self.policy_map = self.policy_graph.imshow(self.discrete_policy_map, aspect='auto', interpolation='none',
                                        extent=[float(self.e_min_vel.get()), float(self.e_max_vel.get()),
                                        float(self.e_max_pos.get()), float(self.e_min_pos.get())])
                self.prev_policy_mode = 0
            elif float(self.policy_mode.get()) == 1:
                self.compute_continuous_policy_map()
                self.policy_map = self.policy_graph.imshow(self.continuous_policy_map, aspect='auto', interpolation='none',
                                         extent=[float(self.e_min_vel.get()), float(self.e_max_vel.get()),
                                         float(self.e_max_pos.get()), float(self.e_min_pos.get())])
                self.prev_policy_mode = 1
            self.policy_graph.autoscale(False)

            # init traj arrays
            self.pos_lines = []
            self.pos_meas_lines = []
            self.pos_t_lines = []
            self.vel_lines = []
            self.vel_meas_lines = []
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

                self.vel_meas_lines.append(self.vel_graph.plot(self.time[i], self.vel_meas[i], 'm', marker='x')[0])
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
                    min_rew = min([min([min(sublist) for sublist in sublist_1]) for sublist_1 in self.rew])
                    max_rew = max([max([max(sublist) for sublist in sublist_1]) for sublist_1 in self.rew])
                    mid_rew = (max_rew + min_rew) / 2
                    self.rew_y_min = mid_rew - (mid_rew - min_rew) * 1.1
                    self.rew_y_max = mid_rew + (max_rew - mid_rew) * 1.1
                    self.rew_graph.set_ylim(self.rew_y_min, self.rew_y_max)
                    print('y lims set')

                    # cumulative reward y limits
                    self.cum_rew_x_axis.set_xdata(numpy.array([0, self.t_term]))
                    min_cum_rew = min([min([min(sublist) for sublist in sublist_1]) for sublist_1 in self.cum_rew])
                    max_cum_rew = max([max([max(sublist) for sublist in sublist_1]) for sublist_1 in self.cum_rew])
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
                self.vel_meas_lines[i].set_xdata(self.time[i])
                self.vel_meas_lines[i].set_ydata(self.vel_meas[i])
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
                self.mean_rew = self.mean_cum_rew / (self.t_term*self.control_freq) if self.t_term > 0 else 0

                # update rew y lims
                min_rew = min([min([min(sublist) for sublist in sublist_1]) for sublist_1 in self.rew])
                max_rew = max([max([max(sublist) for sublist in sublist_1]) for sublist_1 in self.rew])
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
                min_cum_rew = min([min([min(sublist) for sublist in sublist_1]) for sublist_1 in self.cum_rew])
                max_cum_rew = max([max([max(sublist) for sublist in sublist_1]) for sublist_1 in self.cum_rew])
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
                if float(self.policy_mode.get()) == 0 and self.prev_policy_mode != 0:
                    self.policy_map.set_data(self.discrete_policy_map)
                    self.prev_policy_mode = 0
                elif float(self.policy_mode.get()) == 1 and self.prev_policy_mode != 1:
                    self.compute_continuous_policy_map()
                    self.policy_map.set_data(self.continuous_policy_map)
                    self.prev_policy_mode = 1
                self.policy_lines[i].set_xdata(self.vel[i])
                self.policy_lines[i].set_ydata(self.pos[i])

        self.traj_fig.canvas.draw()
        self.policy_fig.canvas.draw()

    # GUI Interaction Functions #
    #---------------------------#

    def S_init_pos(self, v):
        self.init_pos = float(v)
        #self.update_plot(recomputed_flag=False)

    def S_init_vel(self, v):
        self.init_vel = float(v)
        #self.update_plot(recomputed_flag=False)

    # Main GUI Loop Funcion #
    #-----------------------#

    def tkinter_loop(self):

        # root gui
        self.tk_root = tkinter.Tk()

        # for grid indexing
        row = 0
        column = 0

        # Initial State #
        #---------------#

        column += 2

        l_init_cond = tkinter.Label(self.tk_root, text='Initial State')
        l_init_cond.grid(row=row, column=column)

        row += 1
        column -= 2

        l_init_pos = tkinter.Label(self.tk_root, text='init pos')
        l_init_pos.grid(row=row, column=column)

        column += 1

        self.s_init_pos = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL)
        self.s_init_pos.grid(row=row, column=column)

        column += 1

        l_init_vel = tkinter.Label(self.tk_root, text='init vel')
        l_init_vel.grid(row=row, column=column)

        column += 1

        self.s_init_vel = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL)
        self.s_init_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        # Buttons #
        #---------#

        b_terminate = tkinter.Button(f_b, text='Terminate', command=self.BP_terminate)
        b_terminate.grid(row=row, column=column)

        row += 1
        column -= 2

        # Plots #
        #-------#

        self.init_plot()

        row = 0
        column = 0

        l_plots = tkinter.Label(f_p, text='Plots')
        l_plots.grid(row=row, column=column)

        row += 1

        self.c_policy_plot = tkagg.FigureCanvasTkAgg(self.policy_fig, master=f_p)
        self.c_policy_plot.get_tk_widget().grid(row=row, column=column)
        self.c_policy_plot._tkcanvas.grid(row=row, column=column) # not sure if needed?


        # Initialise GUI #
        #----------------#

        #self.update_plot()
        # set up sliders for plot updates
        self.s_init_pos.config(command = lambda v: self.S_init_pos(v))
        self.s_init_vel.config(command = lambda v: self.S_init_vel(v))

        # start mainloop #
        #----------------#

        self.tk_root.mainloop()

def main():

    tk_interface = TK_Interface()

    # start tkinter gui
    #threading.Thread(target=tk_interface.tkinter_loop).start()

    tk_interface.compute_ef_coords()

if __name__ == "__main__":
    main()