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
from mpl_toolkits.mplot3d import axes3d, Axes3D

numpy.set_printoptions(linewidth=1000,threshold=numpy.nan)

class TK_Interface:

    def __init__(self):
        self.origin = numpy.array([0, 0, 0])
        self.x_vec = numpy.array([0.25, 0, 0])
        self.y_vec = numpy.array([0, 0.25, 0])
        self.z_vec = numpy.array([0, 0, 0.25])

        self.init_roll = 0
        self.init_pitch = 0
        self.init_yaw = 0

        self.init_x = 0
        self.init_y = 0
        self.init_z = 0

    # Core Functions #
    #----------------#

    def compute_ef_coords(self):

        R = transforms3d.euler.euler2mat(self.init_roll, self.init_pitch, self.init_yaw, 'sxyz')

        T = numpy.array([self.init_x, self.init_y, self.init_z])

        self.Ef_origin = numpy.matmul(R, self.origin) + T
        self.Ef_x_axis = numpy.matmul(R, self.x_vec) + T
        self.Ef_y_axis = numpy.matmul(R, self.y_vec) + T
        self.Ef_z_axis = numpy.matmul(R, self.z_vec) + T

    def init_plot(self):

        plt.close()

        plt.ion()

        # trajectory plots
        self.traj_fig = plt.figure()
        self.traj_graph = self.traj_fig.add_subplot(111, projection='3d')
        self.traj_graph.set_xlim(-1,1)
        self.traj_graph.set_ylim(-1,1)
        self.traj_graph.set_zlim(-1,1)

        self.traj_fig.tight_layout()  # turn on tight-layout

        # x axis
        self.x_axis = self.traj_graph.plot(numpy.array([self.Ef_origin[0], self.Ef_x_axis[0]]), \
                                           numpy.array([self.Ef_origin[1], self.Ef_x_axis[1]]), \
                                           numpy.array([self.Ef_origin[2], self.Ef_x_axis[2]]), \
                                           'r')[0]  # x axis
        self.y_axis = self.traj_graph.plot(numpy.array([self.Ef_origin[0], self.Ef_y_axis[0]]), \
                                           numpy.array([self.Ef_origin[1], self.Ef_y_axis[1]]), \
                                           numpy.array([self.Ef_origin[2], self.Ef_y_axis[2]]), \
                                           'g')[0]  # y axis
        self.z_axis = self.traj_graph.plot(numpy.array([self.Ef_origin[0], self.Ef_z_axis[0]]), \
                                           numpy.array([self.Ef_origin[1], self.Ef_z_axis[1]]), \
                                           numpy.array([self.Ef_origin[2], self.Ef_z_axis[2]]), \
                                           'b')[0]  # z axis


    def update_plot(self):

        self.x_axis.set_xdata(numpy.array([self.Ef_origin[0], self.Ef_x_axis[0]]))
        self.x_axis.set_ydata(numpy.array([self.Ef_origin[1], self.Ef_x_axis[1]]))
        self.x_axis.set_3d_properties(numpy.array([self.Ef_origin[2], self.Ef_x_axis[2]]))

        self.y_axis.set_xdata(numpy.array([self.Ef_origin[0], self.Ef_y_axis[0]]))
        self.y_axis.set_ydata(numpy.array([self.Ef_origin[1], self.Ef_y_axis[1]]))
        self.y_axis.set_3d_properties(numpy.array([self.Ef_origin[2], self.Ef_y_axis[2]]))

        self.z_axis.set_xdata(numpy.array([self.Ef_origin[0], self.Ef_z_axis[0]]))
        self.z_axis.set_ydata(numpy.array([self.Ef_origin[1], self.Ef_z_axis[1]]))
        self.z_axis.set_3d_properties(numpy.array([self.Ef_origin[2], self.Ef_z_axis[2]]))

        self.traj_fig.canvas.draw()

    # GUI Interaction Functions #
    #---------------------------#

    def BP_terminate(self):
        os._exit(os.EX_OK)

    def S_init_roll(self, v):
        self.init_roll = float(v)
        self.compute_ef_coords()
        self.update_plot()

    def S_init_pitch(self, v):
        self.init_pitch = float(v)
        self.compute_ef_coords()
        self.update_plot()

    def S_init_yaw(self, v):
        self.init_yaw = float(v)
        self.compute_ef_coords()
        self.update_plot()

    def S_init_x(self, v):
        self.init_x = float(v)
        self.compute_ef_coords()
        self.update_plot()

    def S_init_y(self, v):
        self.init_y = float(v)
        self.compute_ef_coords()
        self.update_plot()

    def S_init_z(self, v):
        self.init_z = float(v)
        self.compute_ef_coords()
        self.update_plot()

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


        l_init_roll = tkinter.Label(self.tk_root, text='initial roll:')
        l_init_roll.grid(row=row, column=column)

        row += 1

        self.s_init_roll = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, \
                                         from_=0, to=2*math.pi, resolution=2*math.pi/360, tickinterval=2*math.pi/5)
        self.s_init_roll.grid(row=row, column=column)

        row += 1

        l_init_pitch = tkinter.Label(self.tk_root, text='initial pitch:')
        l_init_pitch.grid(row=row, column=column)

        row += 1

        self.s_init_pitch = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, \
                                         from_=0, to=2*math.pi, resolution=2*math.pi/360, tickinterval=2*math.pi/5)
        self.s_init_pitch.grid(row=row, column=column)

        row += 1

        l_init_yaw = tkinter.Label(self.tk_root, text='initial yaw:')
        l_init_yaw.grid(row=row, column=column)

        row += 1

        self.s_init_yaw = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, \
                                         from_=0, to=2*math.pi, resolution=2*math.pi/360, tickinterval=2*math.pi/5)
        self.s_init_yaw.grid(row=row, column=column)

        row += 1

        l_init_x = tkinter.Label(self.tk_root, text='initial x:')
        l_init_x.grid(row=row, column=column)

        row += 1

        self.s_init_x = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, \
                                         from_=-1, to=1, resolution=1/200, tickinterval=1/2)
        self.s_init_x.grid(row=row, column=column)

        row += 1

        l_init_y = tkinter.Label(self.tk_root, text='initial y:')
        l_init_y.grid(row=row, column=column)

        row += 1

        self.s_init_y = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, \
                                         from_=-1, to=1, resolution=1/200, tickinterval=1/2)
        self.s_init_y.grid(row=row, column=column)

        row += 1

        l_init_z = tkinter.Label(self.tk_root, text='initial z:')
        l_init_z.grid(row=row, column=column)

        row += 1

        self.s_init_z = tkinter.Scale(self.tk_root, length=300, width=45, orient=tkinter.HORIZONTAL, \
                                         from_=-1, to=1, resolution=1/200, tickinterval=1/2)
        self.s_init_z.grid(row=row, column=column)

        row += 1

        # Buttons #
        #---------#

        b_terminate = tkinter.Button(self.tk_root, text='Terminate', command=self.BP_terminate)
        b_terminate.grid(row=row, column=column)

        row += 1

        # Initialise GUI #
        #----------------#

        self.init_plot()
        self.update_plot()
        # set up sliders for plot updates
        self.s_init_roll.config(command = lambda v: self.S_init_roll(v))
        self.s_init_pitch.config(command = lambda v: self.S_init_pitch(v))
        self.s_init_yaw.config(command = lambda v: self.S_init_yaw(v))
        self.s_init_x.config(command = lambda v: self.S_init_x(v))
        self.s_init_y.config(command = lambda v: self.S_init_y(v))
        self.s_init_z.config(command = lambda v: self.S_init_z(v))

        # start mainloop #
        #----------------#

        self.tk_root.mainloop()

def main():

    tk_interface = TK_Interface()

    # start tkinter gui
    threading.Thread(target=tk_interface.tkinter_loop).start()

    tk_interface.compute_ef_coords()

if __name__ == "__main__":
    main()