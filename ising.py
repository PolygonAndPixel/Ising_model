"""
Simulate the 2d Ising model.

@author: Hieronymus, Maicon
@author: Wagner, Tassilo
@date:   14.12.2017

Usage:
    python ising.py n s d t
    
    n: grid_size in each dimension, e.g. 10, 16 or 256
    s: number of steps to simulate, e.g. 100000
    d: number of steps per step, e.g. 500 (speeds up visualization)
    t: temperature of the model
    
or
    python ising.py filename
  
"""
from itertools import product
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import sys
import timeit

print("Usage:\n[filename]: Create data for a plot and save it in [filename]")
print("[int] [int] [int] [float]: specify grid_size and number "
       "of steps to simulate and number of MC steps per step for an "
       "animation and the temperature.\n")
def update_spin(beta, d, n):
    for k in range(d):
        # Take random spin
        x_c, y_c = np.random.randint(n, size=(2))
        # Calculate contribution to the energy
        energy = 0 
        # 4 neighbours
        for row, col in product([-1, 0, 1], [-1, 0, 1]): 
            if row != col and row != -col:
                energy += spins[(x_c+row)%n][(y_c+col)%n]
        energy *= spins[x_c][y_c] 
        # Energy less -> keep
        if energy <= 0:
            spins[x_c][y_c] *= (-1)
        # Energy more -> keep with prob exp(-beta(H_mu - H_nu))
        else:
            if np.random.uniform() < np.exp(-beta*(energy*2)):
                spins[x_c][y_c] *= (-1) 
                
if len(sys.argv) == 2:
###############################For Plots#######################################
    n = 256
    d = 1000000
    # Generate a grid = coordinates of the beads
    x, y = np.mgrid[range(n), range(n)]
    print(np.shape(x))
    filename = sys.argv[1]
   
    print("Benchmark on a {} x {} grid with {} updates".format(n, n, d))

    with open(filename, 'w') as f:
        f.write("lattice_size\ttemperature\tenergy\tmagnetization\n")
    delta_t = 0.5
    n_t = int(5.0/delta_t)
    for t in range(n_t):
        temperature = delta_t * (t+1)
        beta = 1.0/temperature
        global_energy = 0.0
        magnet = 0.0
        avg = 10
        for k in range(avg):
            # Initialize spins 0 and 1 by random. 
            spins = np.random.randint(2, size=(n,n))*2 - 1
            update_spin(beta, d, n)
            global_energy += np.sum([[spins[i][j] * spins[neighbor1][neighbor2] 
                            for neighbor1, neighbor2 in 
                            [[(i-1)%n, j], [(i+1)%n, j], 
                             [i, (j-1)%n], [i, (j+1)%n]]] 
                            for i,j in 
                            list(product(range(n), range(n)))]) / (n*n*-2)
            magnet += np.abs(float(np.sum(spins)) / (n*n))
            
        global_energy /= avg
        magnet /= avg
        with open(filename, 'a') as f:
            f.write("{}\t{}\t{}\t{}\t\n".format(n, 
                    temperature, global_energy, magnet))
    with open(filename, 'a') as f:
            f.write("next\n")
    print(spins)

if len(sys.argv) == 5:
###############################ANIME###########################################
    try:
        n = int(sys.argv[1])
        s = int(sys.argv[2])
        d = int(sys.argv[3])
        temperature = float(sys.argv[4])
    except:
        print("Please give three arguments as int to specify grid_size "
              "and number of steps to simulate and number of MC steps per "
              "step for an animation and the temperature as float.")
        exit()
    np.random.seed(241514)
    # Set MC values
    beta = 1.0/temperature # Play with this value

    # Generate a grid = coordinates of the beads
    x, y = np.mgrid[range(n), range(n)]

    # Initialize spins 0 and 1 by random. 
    spins = np.random.randint(2, size=(n,n))*2 - 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = ax.set_title("2D Ising Model at T={}, Grid_Size={}".format(beta, n))

    image = plt.imshow(spins, cmap='tab20c', vmin=0, vmax=1,
                       extent=[x.min(), x.max(), y.min(), y.max()],
                       interpolation='nearest', origin='lower')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def update_spin_2(i):
        for k in range(d):
            # Take random spin
            x_c, y_c = np.random.randint(n, size=(2))
            # Calculate contribution to the energy
            energy = 0 
            # 4 neighbours
            for row, col in product([-1, 0, 1], [-1, 0, 1]): 
                if row != col and row != -col:
                    energy += spins[(x_c+row)%n][(y_c+col)%n]
            energy *= spins[x_c][y_c] 
            # Energy less -> keep
            if energy <= 0:
                spins[x_c][y_c] *= (-1)
            # Energy more -> keep with prob exp(-beta(H_mu - H_nu))
            else:
                if np.random.uniform() < np.exp(-beta*(energy*2)):
                    spins[x_c][y_c] *= (-1) 
        image.set_data(spins)       
        # Uncomment the following lines for live calculation of energy
        # and magnetization. Takes a lot of processing time.
        if i%10 == 0:
            energy = np.sum([[spins[i][j] * spins[neighbor1][neighbor2] 
                        for neighbor1, neighbor2 in 
                        [[(i-1)%n, j], [(i+1)%n, j], 
                         [i, (j-1)%n], [i, (j+1)%n]]] 
                        for i,j in 
                        list(product(range(n), range(n)))]) / (n*n*-2)
            magnet = float(np.sum(spins)) / (n*n)
            
            ax.set_title("2D Ising Model at T={}, Grid_Size={}, "
                         "\nEnergy={}, Magnet={}".format(temperature, 
                                                         n, 
                                                         energy, 
                                                         magnet))
        return image
        
    anime = animation.FuncAnimation(fig, update_spin_2, interval=1, frames=s,
                                    repeat=False, blit=False)
        
    plt.show()

