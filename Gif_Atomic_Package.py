import matplotlib.pyplot as plt
import imageio
import numpy as np
import matplotlib.text as text
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.colors import LightSource
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import FancyArrowPatch
Nz = 800+1					# Discretization of the Box
Lz = 8.75					# Box Length
Na = 1000+1					# Number of output times
T  = 2000					# Maximum simulated time
FLAG = "Results_t="				# Collision.jl output flag
time = np.ones(Na)
for j in range(0,Na):
        time[j]  = int(j)

x = np.zeros(Nz)
A = np.zeros(Nz)
B = np.zeros(Nz)

for i in range(0,Nz): 
	x[i] = i*(Lz/Nz)

for i in range(0,Nz): 
	B[i] = i*Nz

def create_frame(t):
    fig = plt.figure(figsize=(6,6))   
    y = np.loadtxt(FLAG+f'{int(t)}')
    plt.plot(x[:],y[:], color = 'blue', linestyle="solid", marker = 's', markersize = 2)
    plt.plot(A[:],B[:], color = 'black', linestyle="solid", marker = 's', linewidth=1.0, markersize = 1.0, markeredgewidth=1.0, markevery=1)
    plt.xlim([-0.1,9.01])
    plt.ylim([-0.01,0.81])
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    plt.ylabel(r'$|\langle \chi(z,t)|\chi(z,t)\rangle |^2 $', fontsize = 16)
    plt.xlabel(r'$z$ (In Bohrs)', fontsize = 24)
    plt.title(f' time =  {t*(T/(Na-1))}', fontsize= 16)
    plt.savefig(f'./img/img_{t}.png', transparent = False, facecolor = 'white')    
    plt.close()

for t in time:
    create_frame(t)

frames = []
for t in time:
    image = imageio.v2.imread(f'./img/img_{t}.png')
    frames.append(image)

imageio.mimsave('./example.gif', # output gif
                frames,          # array of input frames
                duration = 100)         # optional: frames per second
