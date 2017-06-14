import os
import operator
import functools
import mdptoolbox
import numpy
import math
import tkinter
import threading
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt

class TK_Interface:

    # Core Functions #
    #----------------#

    def __init__(self):
        self.pos_raw = []
        self.vel_raw = []
        self.acc_raw = []
        self.time = []

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

    def update_plots(self, recomputed_flag):

        if recomputed_flag:

            # trajectory plots

            x_min = 0
            x_max = max(self.time)

            self.pos_graph.cla()
            self.pos_graph.set_title('Displacement (cm)')
            self.pos_graph.set_xlim(x_min, x_max)
            min_pos = float(self.e_min_pos.get())
            max_pos = float(self.e_max_pos.get())
            mid_pos = (max_pos + min_pos) / 2
            y_min = mid_pos - (mid_pos - min_pos)
            y_max = mid_pos + (max_pos - mid_pos) * 1.1
            self.pos_graph.set_ylim(y_min, y_max)
            self.pos_line = self.pos_graph.plot(self.time, self.pos, 'b')[0]
            self.pos_t_line = self.pos_graph.plot(numpy.array([x_max, x_max]), numpy.array([y_min, y_max]), 'r--')[0]

            self.vel_graph.cla()
            self.vel_graph.set_title('Velocity (cm/s)')
            self.vel_graph.set_xlim(x_min, x_max)
            min_vel = float(self.e_min_vel.get())
            max_vel = float(self.e_max_vel.get())
            mid_vel = (max_vel + min_vel) / 2
            y_min = mid_vel - (mid_vel - min_vel)
            y_max = mid_vel + (max_vel - mid_vel) * 1.1
            self.vel_graph.set_ylim(y_min, y_max)
            self.vel_line = self.vel_graph.plot(self.time, self.vel, 'g')[0]
            self.vel_t_line = self.vel_graph.plot(numpy.array([x_max, x_max]), numpy.array([y_min, y_max]), 'r--')[0]

            self.acc_graph.cla()
            self.acc_graph.set_xlabel('Time (s)')
            self.acc_graph.set_title('Acceleration (cm/s^2)')
            self.acc_graph.set_xlim(x_min, x_max)
            min_acc = min(self.acc)
            max_acc = max(self.acc)
            mid_acc = (max_acc + min_acc) / 2
            y_min = mid_acc - (mid_acc - min_acc) * 1.1
            y_max = mid_acc + (max_acc - mid_acc) * 1.1
            self.acc_graph.set_ylim(y_min, y_max)
            self.acc_graph.plot(numpy.array([x_min, x_max]), numpy.array([0, 0]), 'k') # x axis
            self.acc_line = self.acc_graph.plot(self.time, self.acc, 'r')[0]
            self.acc_t_line = self.acc_graph.plot(numpy.array([x_max, x_max]), numpy.array([y_min, y_max]), 'r--')[0]

            # policy map

            self.policy_graph.cla()
            self.policy_graph.set_xlabel('Velocity (cm/s)')
            self.policy_graph.set_title('Displacement (cm)')
            policy_map = numpy.zeros((self.num_pos_states, self.num_actions))
            for i in range(self.num_pos_states):
                for j in range(self.num_actions):
                    policy_map[i,j] = self.vi.policy[int(self.num_states - i * self.num_actions - self.num_actions + j)]
            self.policy_graph.imshow(policy_map, aspect='auto', extent=[float(self.e_min_vel.get()),float(self.e_max_vel.get()),float(self.e_max_pos.get()),float(self.e_min_pos.get())])
            self.policy_graph.autoscale(False)
            self.policy_line = self.policy_graph.plot(self.vel_raw, self.pos_raw, color='k', linestyle='-', linewidth=3, marker='o', markeredgecolor='w')[0]


        else:

            # trajectories
            x_max = max(self.time)
            self.pos_line.set_xdata(self.time)
            self.pos_line.set_ydata(self.pos)
            self.pos_t_line.set_xdata(numpy.array([x_max, x_max]))
            self.vel_line.set_xdata(self.time)
            self.vel_line.set_ydata(self.vel)
            self.vel_t_line.set_xdata(numpy.array([x_max, x_max]))
            self.acc_line.set_xdata(self.time)
            self.acc_line.set_ydata(self.acc)
            self.acc_t_line.set_xdata(numpy.array([x_max, x_max]))

            # policy map
            self.policy_line.set_xdata(self.vel)
            self.policy_line.set_ydata(self.pos)

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
        self.s_init_pos.set(float(self.e_max_pos.get()))
        self.s_init_vel.config(from_=float(self.e_min_vel.get()), to=float(self.e_max_vel.get()), resolution=self.vel_res, tickinterval=(float(self.e_max_vel.get())-float(self.e_min_vel.get()))/5)
        self.s_init_vel.set(float(self.e_min_vel.get()))

        self.tk_root.update_idletasks()

    def reset_gui(self):

        self.e_min_vel.delete(0, tkinter.END)
        self.e_max_vel.delete(0, tkinter.END)
        self.e_num_actions.delete(0, tkinter.END)
        self.e_min_pos.delete(0, tkinter.END)
        self.e_max_pos.delete(0, tkinter.END)
        self.e_num_positions.delete(0, tkinter.END)
        self.e_dist_factor.delete(0, tkinter.END)
        self.e_acc_factor.delete(0, tkinter.END)
        self.e_discount.delete(0, tkinter.END)
        self.e_epsilon.delete(0, tkinter.END)
        self.e_max_iter.delete(0, tkinter.END)

        self.e_min_vel.insert(tkinter.END, '0')
        self.e_max_vel.insert(tkinter.END, '20')
        self.e_num_actions.insert(tkinter.END, '6')
        self.e_min_pos.insert(tkinter.END, '0')
        self.e_max_pos.insert(tkinter.END, '50')
        self.e_num_positions.insert(tkinter.END, '126')
        self.e_dist_factor.insert(tkinter.END, '1')
        self.e_acc_factor.insert(tkinter.END, '1')
        self.e_discount.insert(tkinter.END, '0.99')
        self.e_epsilon.insert(tkinter.END, '0.01')
        self.e_max_iter.insert(tkinter.END, '1000')

        self.update_gui('dummy_event')

    def compute_vi(self):

        # parameters
        self.num_actions = int(self.e_num_actions.get())
        self.num_pos_states = int(self.e_num_positions.get())
        self.num_states = self.num_pos_states * self.num_actions

        # transition matrix
        transitions = numpy.zeros((self.num_actions, self.num_states, self.num_states))
        no_vel_trans = numpy.zeros((self.num_states, self.num_states))
        # setup base zero velocity transition matrix, for i = 0
        for i in range(0, int(self.num_states / self.num_actions)):
            no_vel_trans[i * self.num_actions:i * self.num_actions + self.num_actions, i * self.num_actions] = numpy.ones(self.num_actions)
        # setup full transition matrix based on position state
        for i in range(0, self.num_actions):
            transitions[i, :, :] = numpy.roll(no_vel_trans, i * self.num_actions + i, 1)

        # reward matrix
        reward = numpy.zeros((self.num_states, self.num_actions))
        for i in range(0, self.num_states):
            for j in range(0, self.num_actions):
                reward[i, j] = -float(self.e_dist_factor.get())*abs(math.floor((self.num_states - i) / self.num_actions))*self.pos_res - float(self.e_acc_factor.get())*pow(
                    (j*self.vel_res - (i - math.floor(i / self.num_actions) * self.num_actions)*self.vel_res), 2)

        # define value iteration
        self.vi = mdptoolbox.mdp.ValueIteration(transitions, reward, float(self.e_discount.get()), float(self.e_epsilon.get()), float(self.e_max_iter.get()), 0)
        self.vi.run()

    def compute_traj(self):

        # parameters
        time_step = 1/float(self.control_freq)

        # initialise state
        state = [int(float(self.s_init_pos.get())/self.pos_res), int(float(self.s_init_vel.get())/self.vel_res)]

        # initialise trajectories for plotting
        self.pos_raw.clear()
        self.vel_raw.clear()
        self.acc_raw.clear()
        self.time.clear()
        t = 0

        # populate trajectories
        traj_end = False
        while traj_end is False:

            if (state[0] == 0 and state[1] == 0):
                traj_end = True

            target_vel = self.vi.policy[int(self.num_states - state[0] * self.num_actions - self.num_actions + state[1])]

            self.pos_raw.append(state[0]) # current state
            self.vel_raw.append(state[1]) # current velocity
            self.acc_raw.append(target_vel-state[1]) # current accel
            self.time.append(t)
            t += time_step

            new_pos = (state[0] - target_vel) % self.num_pos_states
            state = [new_pos, target_vel]

        # rescale trajectories to correct dimensions
        self.pos = [i*self.pos_res for i in self.pos_raw]
        self.vel = [i*self.vel_res for i in self.vel_raw]
        self.acc = [i*self.vel_res*self.control_freq for i in self.acc_raw]

    def compute(self):
        self.compute_vi()
        self.compute_traj()

    # GUI Interaction Functions #
    #---------------------------#

    def BP_compute(self):
        self.update_gui('dummy_event')
        self.compute()
        self.update_plots(recomputed_flag=True)

    def BP_reset(self):
        self.reset_gui()

    def BP_terminate(self):
        os._exit(os.EX_OK)

    def S_init_pos(self, v):
        self.init_pos = v
        self.compute_traj()
        self.update_plots(recomputed_flag=False)

    def S_init_vel(self, v):
        self.init_vel = v
        self.compute_traj()
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

        l_vel = tkinter.Label(f_tb, text='velocity parameters')
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

        l_pos = tkinter.Label(f_tb, text='Position parameters')
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

        l_vi = tkinter.Label(f_tb, text='Value Iteration parameters')
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

        l_rew = tkinter.Label(f_tb, text='Reward parameters')
        l_rew.grid(row=row, column=column)

        row += 1
        column -= 3

        l_dist_factor = tkinter.Label(f_tb, text='dist factor')
        l_dist_factor.grid(row=row, column=column)

        column += 1

        self.e_dist_factor = tkinter.Entry(f_tb)
        self.e_dist_factor.grid(row=row, column=column)

        column += 1

        l_acc_factor = tkinter.Label(f_tb, text='acc^2 factor')
        l_acc_factor.grid(row=row, column=column)

        column += 1

        self.e_acc_factor = tkinter.Entry(f_tb)
        self.e_acc_factor.grid(row=row, column=column)

        row += 1
        column -=3

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
