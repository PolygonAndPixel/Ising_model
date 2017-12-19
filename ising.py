"""
Simulate the 2d Ising model.

@author: Hieronymus, Maicon
@author: Wagner, Tassilo
@date:   14.12.2017

Usage:
    python ising.py n s
    
    n: grid_size in each dimension, e.g. 10, 15 or 500
    s: number of steps to simulate, e.g. 1000
    d: number of steps per step, e.g. 10 (speeds up visualization
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys
import timeit

if len(sys.argv) < 4:
    print("Please give three arguments as int to specify grid_size and number of"
          " steps to simulate and number of MC steps per step")
    exit()
    
n = int(sys.argv[1])
s = int(sys.argv[2])
d = int(sys.argv[3])
# Set MC values

temperature = 3.0 # Play with this value

# Generate a grid = coordinates of the beads
x, y = np.mgrid[range(n), range(n)]

# Initialize spins 0 and 1 by random. 
spins = np.random.randint(2, size=(n,n))*2 - 1

###############################BENCHMARK#######################################
n = 4096
d = 100 * 16 * 16 * 256
# Generate a grid = coordinates of the beads
x, y = np.mgrid[range(n), range(n)]

# Initialize spins 0 and 1 by random. 
spins = np.random.randint(2, size=(n,n))*2 - 1
print("Benchmark on a {} x {} grid with {} updates".format(n, n, d))

def update_spin():
    # implement something here
    for i in range(d):
        # Take random spin
        x_c, y_c = np.random.randint(n, size=(2))
        # Calculate contribution to the energy
        energy = 0 
        # 4 neighbours
        for row in range(2): 
            for col in range(2):
                energy += spins[x_c][y_c] * spins[(x_c+row)%n][(y_c+col)%n]
        energy *= -1
        # Flip and calculate new contribution
        my_spin = spins[x_c][y_c] * (-1)
        energy_after = 0
        for row in range(2): 
            for col in range(2):
                energy_after += my_spin * spins[(x_c+row)%n][(y_c+col)%n]
        energy_after *= -1
        # Energy less -> keep
        if energy_after < energy:
            spins[x_c][y_c] *= (-1)
        # Energy more -> keep with prob exp(-beta(H_mu - H_nu))
        else:
            p = np.random.uniform()
            if p < np.exp(-temperature*(energy_after - energy)):
                spins[x_c][y_c] *= (-1)
elapsed_time = timeit.timeit(update_spin, number=1)
print("Elapsed time: {}".format(elapsed_time))
fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title("2D Ising Model at T={}, Grid_Size={}".format(temperature, n))


image = plt.imshow(spins, cmap='tab20c', vmin=0, vmax=1,
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   interpolation='nearest', origin='lower')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

###############################ANIME###########################################
"""
fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title("2D Ising Model at T={}, Grid_Size={}".format(temperature, n))


image = plt.imshow(spins, cmap='tab20c', vmin=0, vmax=1,
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   interpolation='nearest', origin='lower')
ax.set_xlabel('X')
ax.set_ylabel('Y')

def update_spin(i):
    # implement something here
    for i in range(d):
        # Take random spin
        x_c, y_c = np.random.randint(n, size=(2))
        # Calculate contribution to the energy
        energy = 0 
        # 4 neighbours
        for row in range(2): 
            for col in range(2):
                energy += spins[x_c][y_c] * spins[(x_c+row)%n][(y_c+col)%n]
        energy *= -1
        # Flip and calculate new contribution
        my_spin = spins[x_c][y_c] * (-1)
        energy_after = 0
        for row in range(2): 
            for col in range(2):
                energy_after += my_spin * spins[(x_c+row)%n][(y_c+col)%n]
        energy_after *= -1
        # Energy less -> keep
        if energy_after < energy:
            spins[x_c][y_c] *= (-1)
        # Energy more -> keep with prob exp(-beta(H_mu - H_nu))
        else:
            p = np.random.uniform()
            if p < np.exp(-temperature*(energy_after - energy)):
                spins[x_c][y_c] *= (-1)
    image.set_data(spins)
    return image
    
anime = animation.FuncAnimation(fig, update_spin, interval=1, frames=s,
                                repeat=False, blit=False)
    

plt.show()
"""
