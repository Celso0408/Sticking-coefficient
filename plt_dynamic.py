import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, a, bope, b, c, d, cont_eq, Time_step, recorded_step):

        self.t = a 
        self.bope = bope
        self.x = b 
        self.y = c 
        self.z = d
        self.cont_eq = cont_eq
        self.time_step = Time_step
        self.rec_step = recorded_step

        print(self.y.shape) 

        fig = plt.figure(figsize = (13,9))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)

        self.line1 = Line2D([], [], color = 'blue', linewidth = 2)
        ax1.add_line(self.line1)
        ax1.set_xlim(0, 10)
        ax1.set_xlim(0, 10)
        ax1.set_xticklabels([])
        ax1.set_ylabel("$|\chi(z,t)|^2$")

        self.line2 = Line2D([], [], color = 'red', linewidth = 2)
        ax2.add_line(self.line2)
        ax2.set_xlim(0, 10)
        ax2.set_xticklabels([])
        ax2.set_ylim(-100, 100)
        ax2.set_ylabel("$S(z,t)$")
        Time = 0.00
        self.text = ax2.text(8.5, 0.2, str(Time))
        self.text.set_text(Time)
        
        self.line3 = Line2D([], [], color = 'black', linewidth = 2)
        self.line3a = Line2D([], [], color='red', linestyle='dashed')
        ax3.add_line(self.line3)
        ax3.add_line(self.line3a)
        ax3.legend([self.line3, self.line3a], ['TDPES', 'BOPE'])
        ax3.set_xlim(0, 10.0)
        ax3.set_ylim(-0.5, 2.2)
        ax3.set_ylabel("TDPES$(z,t)$")
        ax3.set_xlabel("$z(a.u)$")

        self.line4 = Line2D([], [], color = 'black', linewidth = 2)
        ax4.add_line(self.line4)
        ax4.set_xlim(0, 10)
        ax4.set_ylim(-0.2, 0.2)
        ax4.set_ylabel("Residual continuity $ (z,t) $")
        ax4.set_xlabel("$z(a.u)$")

        animation.TimedAnimation.__init__(self, fig, interval=100, blit=True)

    def _draw_frame(self, framedata):
        for i in range(self.x.shape[0]):
            i = framedata
            #print(i)
            Time = round(i*self.rec_step*self.time_step/41,3)
            Time = "Time = " + str(Time) + " (fs)"
            self.text.set_text(Time)

            self.line1.set_data(self.t, list(self.x[i][:]))
            self.line2.set_data(self.t, list(self.y[i][:]))
            self.line3.set_data(self.t, list(self.z[i][:]))
            self.line3a.set_data(self.t, self.bope)
            self.line4.set_data(self.t, list(self.cont_eq[i][:]))  
            self._drawn_artists = [self.line1, self.line2, self.line3, self.line3a, self.line4]
        ...

    def new_frame_seq(self):
        #print(self.x.shape[0])
        return iter(range(self.x.shape[0]))

    def _init_draw(self):
        lines = [self.line1, self.line2, self.line3, self.line3a, self.line4]

        for l in lines:
            l.set_data([], [])
        ...