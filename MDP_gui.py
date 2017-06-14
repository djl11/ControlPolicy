import os
import operator
import functools
import mdptoolbox
import numpy
import math
import tkinter
import threading
import matplotlib.pyplot as plt

class TK_Interface:

    def __init__(self):
        self.pos = []
        self.vel = []
        self.acc = []
        self.time = []

    def init_plot(self):

        plt.close()

        plt.ion()

        self.fig = plt.figure()

        self.pos_graph = self.fig.add_subplot(3, 1, 1)
        self.pos_graph.set_title('Displacement (cm)')

        self.vel_graph = self.fig.add_subplot(3, 1, 2)
        self.vel_graph.set_title('Velocity (cm/s)')

        self.acc_graph = self.fig.add_subplot(3, 1, 3)
        self.acc_graph.set_xlabel('Time (s)')
        self.acc_graph.set_title('Acceleration (cm/s^2)')

        plt.tight_layout()  # turn on tight-layout


    def update_plot(self, recomputed_flag):

        if recomputed_flag:

            self.pos_graph.cla()
            self.pos_graph.set_xlim(0, max(self.time))
            min_pos = float(self.e_min_pos.get())
            max_pos = float(self.e_max_pos.get())
            mid_pos = (max_pos + min_pos) / 2
            self.pos_graph.set_ylim(mid_pos - (mid_pos - min_pos) * 1.1, mid_pos + (max_pos - mid_pos) * 1.1)
            self.pos_line = self.pos_graph.plot(self.time, self.pos, 'b')[0]

            self.vel_graph.cla()
            self.vel_graph.set_xlim(0, max(self.time))
            min_vel = float(self.e_min_vel.get())
            max_vel = float(self.e_max_vel.get())
            mid_vel = (max_vel + min_vel) / 2
            self.vel_graph.set_ylim(mid_vel - (mid_vel - min_vel) * 1.1, mid_vel + (max_vel - mid_vel) * 1.1)
            self.vel_line = self.vel_graph.plot(self.time, self.vel, 'g')[0]

            self.acc_graph.cla()
            self.acc_graph.set_xlim(0, max(self.time))
            min_acc = min(self.acc)
            max_acc = max(self.acc)
            mid_acc = (max_acc + min_acc) / 2
            self.acc_graph.set_ylim(mid_acc - (mid_acc - min_acc) * 1.1, mid_acc + (max_acc - mid_acc) * 1.1)
            self.acc_line = self.acc_graph.plot(self.time, self.acc, 'r')[0]

        else:

            self.pos_line.set_xdata(self.time)
            self.pos_line.set_ydata(self.pos)
            self.vel_line.set_xdata(self.time)
            self.vel_line.set_ydata(self.vel)
            self.acc_line.set_xdata(self.time)
            self.acc_line.set_ydata(self.acc)

        self.fig.canvas.draw()



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

        print('computing...')

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
                reward[i, j] = -float(self.e_dist_factor.get())*abs(math.floor((self.num_states - i) / self.num_actions)) - float(self.e_acc_factor.get())*pow(
                    (j - (i - math.floor(i / self.num_actions) * self.num_actions)), 2)

        # define value iteration
        self.vi = mdptoolbox.mdp.ValueIteration(transitions, reward, float(self.e_discount.get()), float(self.e_epsilon.get()), float(self.e_max_iter.get()), 0)
        self.vi.run()

        print('computed vi')

    def compute_traj(self):

        # parameters
        time_step = 1/float(self.control_freq)

        # initialise state
        state = [int(float(self.s_init_pos.get())/self.pos_res), int(float(self.s_init_vel.get())/self.vel_res)]

        # initialise trajectories for plotting
        self.pos.clear()
        self.vel.clear()
        self.acc.clear()
        self.time.clear()
        t = 0

        # populate trajectories
        traj_end = False
        while traj_end is False:

            if (state[0] == 0 and state[1] == 0):
                traj_end = True

            target_vel = self.vi.policy[int(self.num_states - state[0] * self.num_actions - self.num_actions + state[1])]

            self.pos.append(state[0]) # current state
            self.vel.append(state[1]) # current velocity
            self.acc.append(target_vel-state[1]) # current accel
            self.time.append(t)
            t += time_step

            new_pos = (state[0] - target_vel) % self.num_pos_states
            state = [new_pos, target_vel]

        # rescale trajectories to correct dimensions
        self.pos = [i*self.pos_res for i in self.pos]
        self.vel = [i*self.vel_res for i in self.vel]
        self.acc = [i*self.vel_res*self.control_freq for i in self.acc]


    def compute(self):
        self.compute_vi()
        self.compute_traj()

    # GUI Interaction Functions #
    #---------------------------#

    def BP_compute(self):
        self.compute()
        self.update_plot(recomputed_flag=True)

    def BP_reset(self):
        self.reset_gui()

    def BP_terminate(self):
        os._exit(os.EX_OK)

    def S_init_pos(self, v):
        self.init_pos = v
        self.compute_traj()
        self.update_plot(recomputed_flag=False)

    def S_init_vel(self, v):
        self.init_vel = v
        self.compute_traj()
        self.update_plot(recomputed_flag=False)


    # Main GUI Loop Funcion #
    #-----------------------#

    def tkinter_loop(self):

        self.tk_root = tkinter.Tk()

        row = 0
        column = 0

        # Initialise Plot #
        #-----------------#

        self.init_plot()

        # Velocity Parameters #
        #---------------------#

        column += 4

        l_vel = tkinter.Label(self.tk_root, text='velocity parameters')
        l_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        l_min_vel = tkinter.Label(self.tk_root, text='min vel (cm/s)')
        l_min_vel.grid(row=row, column=column)

        column += 1

        self.e_min_vel = tkinter.Entry(self.tk_root)
        self.e_min_vel.grid(row=row,column=column)
        self.e_min_vel.bind("<FocusOut>", self.update_gui)

        column += 1

        l_max_vel = tkinter.Label(self.tk_root, text='max vel (cm/s)')
        l_max_vel.grid(row=row, column=column)

        column += 1

        self.e_max_vel = tkinter.Entry(self.tk_root)
        self.e_max_vel.grid(row=row,column=column)
        self.e_max_vel.bind("<FocusOut>", self.update_gui)

        column += 1

        l_num_actions = tkinter.Label(self.tk_root, text='num actions')
        l_num_actions.grid(row=row, column=column)

        column += 1

        self.e_num_actions = tkinter.Entry(self.tk_root)
        self.e_num_actions.grid(row=row,column=column)
        self.e_num_actions.bind("<FocusOut>", self.update_gui)

        column += 1

        self.sv_vel_res = tkinter.StringVar()
        l_vel_res = tkinter.Label(self.tk_root, textvariable=self.sv_vel_res)
        l_vel_res.grid(row=row, column=column)

        row += 1
        column -= 6

        # Position Parameters #
        #---------------------#

        column += 3

        l_vel = tkinter.Label(self.tk_root, text='Position parameters')
        l_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        l_min_pos = tkinter.Label(self.tk_root, text='min pos (cm)')
        l_min_pos.grid(row=row, column=column)

        column += 1

        self.e_min_pos = tkinter.Entry(self.tk_root)
        self.e_min_pos.grid(row=row,column=column)
        self.e_min_pos.bind("<FocusOut>", self.update_gui)

        column += 1

        l_max_pos = tkinter.Label(self.tk_root, text='max pos (cm)')
        l_max_pos.grid(row=row, column=column)

        column += 1

        self.e_max_pos = tkinter.Entry(self.tk_root)
        self.e_max_pos.grid(row=row,column=column)
        self.e_max_pos.bind("<FocusOut>", self.update_gui)

        column += 1

        l_num_positions = tkinter.Label(self.tk_root, text='num positions')
        l_num_positions.grid(row=row, column=column)

        column += 1

        self.e_num_positions = tkinter.Entry(self.tk_root)
        self.e_num_positions.grid(row=row,column=column)
        self.e_num_positions.bind("<FocusOut>", self.update_gui)

        column += 1

        self.sv_pos_res = tkinter.StringVar()
        l_pos_res = tkinter.Label(self.tk_root, textvariable=self.sv_pos_res)
        l_pos_res.grid(row=row, column=column)

        row += 1
        column -= 6

        # Reward Parameters #
        #-------------------#

        column += 3

        l_rew = tkinter.Label(self.tk_root, text='Reward parameters')
        l_rew.grid(row=row, column=column)

        row += 1
        column -= 3

        l_dist_factor = tkinter.Label(self.tk_root, text='dist factor')
        l_dist_factor.grid(row=row, column=column)

        column += 1

        self.e_dist_factor = tkinter.Entry(self.tk_root)
        self.e_dist_factor.grid(row=row, column=column)

        column += 1

        l_acc_factor = tkinter.Label(self.tk_root, text='acc^2 factor')
        l_acc_factor.grid(row=row, column=column)

        column += 1

        self.e_acc_factor = tkinter.Entry(self.tk_root)
        self.e_acc_factor.grid(row=row, column=column)

        column += 3

        # control frequency, to align properly on r.h.s.
        self.sv_control_freq = tkinter.StringVar()
        l_control_freq = tkinter.Label(self.tk_root, textvariable=self.sv_control_freq)
        l_control_freq.grid(row=row, column=column)

        row += 1
        column -=6

        # Value Iteration Parameters #
        #----------------------------#

        column += 3

        l_vi = tkinter.Label(self.tk_root, text='Value Iteration parameters')
        l_vi.grid(row=row, column=column)

        row += 1
        column -= 3

        l_discount = tkinter.Label(self.tk_root, text='discount')
        l_discount.grid(row=row, column=column)

        column += 1

        self.e_discount = tkinter.Entry(self.tk_root)
        self.e_discount.grid(row=row, column=column)

        column += 1

        l_epsilon = tkinter.Label(self.tk_root, text='epsilon')
        l_epsilon.grid(row=row, column=column)

        column += 1

        self.e_epsilon = tkinter.Entry(self.tk_root)
        self.e_epsilon.grid(row=row, column=column)

        column += 1

        l_max_iter = tkinter.Label(self.tk_root, text='max iter')
        l_max_iter.grid(row=row, column=column)

        column += 1

        self.e_max_iter = tkinter.Entry(self.tk_root)
        self.e_max_iter.grid(row=row, column=column)

        row += 1
        column -= 5

        # Initial State #
        #---------------#

        column += 3

        l_init_cond = tkinter.Label(self.tk_root, text='Initial State')
        l_init_cond.grid(row=row, column=column)

        row += 1
        column -= 3

        l_init_pos = tkinter.Label(self.tk_root, text='init pos')
        l_init_pos.grid(row=row, column=column)

        column += 1

        self.s_init_pos = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, command = lambda v: self.S_init_pos(v))
        self.s_init_pos.grid(row=row, column=column)

        column += 1

        l_init_vel = tkinter.Label(self.tk_root, text='init vel')
        l_init_vel.grid(row=row, column=column)

        column += 1

        self.s_init_vel = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, command=lambda v: self.S_init_vel(v))
        self.s_init_vel.grid(row=row, column=column)

        row += 1
        column -= 3

        # Buttons #
        #---------#

        column += 3

        l_buttons = tkinter.Label(self.tk_root, text='Buttons')
        l_buttons.grid(row=row, column=column)

        row += 1
        column -= 3

        column += 2

        b_compute = tkinter.Button(self.tk_root, text='Compute', command=self.BP_compute)
        b_compute.grid(row=row, column=column)

        column += 1

        b_reset = tkinter.Button(self.tk_root, text='Reset', command=self.BP_reset)
        b_reset.grid(row=row, column=column)

        column += 1

        b_terminate = tkinter.Button(self.tk_root, text='Terminate', command=self.BP_terminate)
        b_terminate.grid(row=row, column=column)

        # Initialise GUI #
        #----------------#

        self.reset_gui()
        self.compute()
        self.update_plot(recomputed_flag=True)

        # start mainloop #
        #----------------#

        self.tk_root.mainloop()

def main():

    tk_interface = TK_Interface()

    # start tkinter gui
    threading.Thread(target=tk_interface.tkinter_loop).start()

if __name__ == "__main__":
    main()
