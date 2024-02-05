import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Reduced units
T_star = 0.71  # Reduced temperature
n_star = 0.844  # Reduced density
num_molecules = 84

# Calculate box size for the given reduced density in 2D
box_area = num_molecules / n_star
box_size = np.sqrt(box_area)  # Since area = length^2 in 2D

# Maximum displacement in a single step
max_displacement = 0.1  

# Initialize positions in a 2D lattice
np.random.seed(0)
positions = np.random.rand(num_molecules, 2) * box_size

def lennard_jones_potential(r):
    """ Calculate Lennard-Jones potential for a distance r in reduced units """
    r6 = (1 / r)**6  # sigma is 1 in reduced units
    r12 = r6**2
    return 4 * (r12 - r6)  # epsilon is 1 in reduced units

def total_potential_energy(positions):
    """ Compute the total potential energy of the system """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    diff = diff - np.round(diff / box_size) * box_size  # Periodic boundary conditions
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(distances, np.inf)
    return np.sum(lennard_jones_potential(distances)) / 2

def metropolis_step(positions, total_energy):
    """ Perform one step of the Metropolis algorithm """
    molecule_idx = np.random.randint(num_molecules)
    old_position = positions[molecule_idx].copy()
    displacement = (np.random.rand(2) - 0.5) * max_displacement
    positions[molecule_idx] += displacement
    positions[molecule_idx] %= box_size
    new_energy = total_potential_energy(positions)
    delta_energy = new_energy - total_energy
    if delta_energy > 0 and -delta_energy / T_star < np.log(np.random.rand()):
        positions[molecule_idx] = old_position
    else:
        total_energy = new_energy
    return positions, total_energy

# Initialize the total energy
total_energy = total_potential_energy(positions)

# Number of steps for the simulation
num_steps = 1000

# Perform the simulation and store positions at each step
positions_history = [positions.copy()]
for step in range(num_steps * num_molecules):
    positions, total_energy = metropolis_step(positions, total_energy)
    if step % num_molecules == 0:
        positions_history.append(positions.copy())

# Setting up the animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
scatter = ax.scatter(positions_history[0][:, 0], positions_history[0][:, 1], color='blue')

def init():
    scatter.set_offsets(positions_history[0])
    return scatter,

def update(frame):
    scatter.set_offsets(positions_history[frame])
    return scatter,

anim = animation.FuncAnimation(fig, update, frames=len(positions_history), init_func=init, blit=True, interval=50)

# Uncomment the next line to save the animation as a GIF
# anim.save('molecular_dynamics_simulation.gif', writer='imagemagick', fps=30)

plt.show()
