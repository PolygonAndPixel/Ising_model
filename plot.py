import matplotlib.pyplot as plt
import numpy as np
import sys
temp = []
mag = []
energy = []
first = True
fig = plt.figure()

with open(sys.argv[1], "r") as f:
    
    for line in f:
        if first:
            first = False
            continue
        values = line.split()
        if len(values) > 1:
            lattice = int(values[0])
            temp.append(float(values[1]))
            energy.append(float(values[2]))
            mag.append(float(values[3]))
        else:
            ax1 = fig.add_subplot(211)
            ax1.set_xlim(temp[0], temp[-1])
            
            plt.plot(temp, energy, label="energy with lattice {}".format(lattice))
            ax2 = fig.add_subplot(212)
            plt.plot(temp, mag, label="magnetization with lattice {}".format(lattice))
            ax2.set_xlim(temp[0], temp[-1])
            temp = []
            mag = []
            energy = []
plt.legend()
title = ax1.set_title("2D Ising Model - Energy")
title = ax2.set_title("2D Ising Model - Magnetization")
plt.show()
