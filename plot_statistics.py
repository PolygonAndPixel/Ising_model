from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import sys

import pandas as pd

temp = []
mag = []
energy = []
lattice = []
first = True

color = cycle(cm.rainbow(np.linspace(0,1,6)))


with open(sys.argv[1], "r") as f:
    
    for line in f:
        if first:
            first = False
            continue
        values = line.split()
        if len(values) > 1:
            lattice.append(int(values[0]))
            temp.append(float(values[1]))
            energy.append(float(values[2]))
            mag.append(float(values[3]))
        
dataset = list(zip(temp, energy, mag, lattice))       
df = pd.DataFrame(data = dataset, 
                  columns=["Temperature", "Energy", 
                           "Magnetization", "Lattice Size"])
####################Total values###############                           
fig, ax = plt.subplots()
# Plot energy
for key, grp in df.groupby(["Lattice Size"]):
    ax = grp.plot(ax=ax, kind="scatter", x="Temperature", 
                  y="Energy", label=key, c=next(color)) 

plt.legend(loc="best")
plt.title("Energy")
plt.xlabel("Temperature $T$ $[J/k_B]$")
plt.ylabel("Energy $[J]$")
#plt.show()  

fig, ax = plt.subplots()             
# Plot magnetization
for key, grp in df.groupby(["Lattice Size"]):
    ax = grp.plot(ax=ax, kind="scatter", x="Temperature", 
                  y="Magnetization", label=key, c=next(color))
plt.legend(loc="best")
plt.title("Magnetization")
plt.xlabel("Temperature $T$ $[J/k_B]$")
plt.ylabel("Magnetization $M$")
#plt.show() 
                      
#df.to_csv("data.csv", index=False, header=True)
####################Means###############
# Calculate mean magnetization and energy given different lattice sizes
# and temperatures
means = df.groupby(["Lattice Size", "Temperature"]).mean()
temperatures = df["Temperature"].unique()

# print(means["Magnetization"].get(128))
# print(means["Magnetization"].index.get_level_values(0).unique())
fig, ax = plt.subplots()  
for lattice_size in means["Magnetization"].index.get_level_values(0).unique():
    ax.plot(temperatures, means["Magnetization"].get(lattice_size), 
    label=lattice_size, c=next(color))
plt.legend(loc="best")
plt.title("Magnetization means")
plt.xlabel("Temperature $T$ $[J/k_B]$")
plt.ylabel("Magnetization $<M>$")
#plt.show()

fig, ax = plt.subplots()  
for lattice_size in means["Energy"].index.get_level_values(0).unique():
    ax.plot(temperatures, means["Energy"].get(lattice_size), 
    label=lattice_size, c=next(color))
plt.legend(loc="best")
plt.title("Energy means")
plt.xlabel("Temperature $T$ $[J/k_B]$")
plt.ylabel("Energy $<E>$")
#plt.show()

####################Calculate different values##############
fig, ax = plt.subplots() 
# magnetic susceptibility over temperature
# Magnetic susceptibility = 1/T * (<M^2> - <M>^2)
magn_susc_lattice = {}
for key, grp in df.groupby(["Lattice Size"]):
    magn_susc = []
    temp_mag = grp.get(["Temperature", "Magnetization"])

    for temp in list(grp["Temperature"].unique()):
        where = np.where((temp_mag["Temperature"] == temp) == True)
        all_mag = list(temp_mag["Magnetization"])
        magnetics = np.asarray([all_mag[idx] for idx in where[0]])
        magn_susc.append(np.var(magnetics)/temp)

    plt.scatter(x=list(grp["Temperature"].unique()), y=magn_susc,  
             label=key, c=next(color))
             
    magn_susc_lattice[key] = list(grp["Temperature"])[np.argmax(magn_susc)]
plt.legend(loc="best")
plt.title("Magnetic susceptibility")
plt.xlabel("Temperature $T$ $[J/k_B]$")
plt.ylabel("Magnetic susceptibility $\\chi$ $[\\mu/k_B]$")

# T_c is around where magnetic susceptiility gets infinitely big
# That is saved in magn_susc_lattice already
fig, ax = plt.subplots() 
lat = []
t_c = []

for key in magn_susc_lattice:
    lat.append(1/(key*key))
    t_c.append(magn_susc_lattice[key])

plt.scatter(x=lat, y=t_c)

plt.title("Finding the critical temperature")
plt.xlabel("Lattice size $[1/N^2]$")
plt.ylabel("Critical temperature $T_c$ $[\\frac{J}{k_B}]$")
#plt.show()

fig, ax = plt.subplots() 
# Specific heat over temperature
# Specific heat = 1/T^2 * (<E^2> - <E>^2)

for key, grp in df.groupby(["Lattice Size"]):
    spec_heat = []
    temp_en = grp.get(["Temperature", "Energy"])

    for temp in list(grp["Temperature"].unique()):
        where = np.where((temp_en["Temperature"] == temp) == True)
        all_mag = list(temp_en["Energy"])
        energies = np.asarray([all_mag[idx] for idx in where[0]])
        spec_heat.append(np.var(energies)/(temp*temp))

    plt.scatter(x=list(grp["Temperature"].unique()), y=spec_heat,  
             label=key, c=next(color))
             
plt.legend(loc="best")
plt.title("Specific Heat")
plt.xlabel("Temperature $T$ $[J/k_B]$")
plt.ylabel("Specific Heat $C_v$ $[J/k_B^2]$")
plt.show()
