import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
kT = 1.0  # Boltzmann constant times temperature
epsilon = 1.0  # Depth of potential well
sigma = 1.0  # Finite distance where potential is zero
num_molecules = 30  # Number of molecules
box_size = 10  # Size of the simulation box
max_displacement = 0.1  # Maximum displacement in a single step

np.random.seed(0)
positions = np.random.rand(num_molecules, 2) * box_size

def lennard_jones_potential(r):
    """ Calculate Lennard-Jones potential for a distance r """
    r6 = (sigma / r)**6
    r12 = r6**2
    return 4 * epsilon * (r12 - r6)

def total_potential_energy(positions):
    """ Compute the total potential energy of the system """
    energy = 0.0
    for i in range(num_molecules):
        for j in range(i+1, num_molecules):
            distance = np.linalg.norm(positions[i] - positions[j])
            energy += lennard_jones_potential(distance)
    return energy

def metropolis_step(positions, total_energy):
    """ Perform one step of the Metropolis algorithm """
    # Choose a random molecule
    molecule_idx = np.random.randint(num_molecules)
    old_position = positions[molecule_idx].copy()

    # Move the molecule to a new position
    displacement = (np.random.rand(2) - 0.5) * max_displacement
    positions[molecule_idx] += displacement

    # Apply periodic boundary conditions
    positions[molecule_idx] %= box_size

    # Calculate the energy change
    new_energy = total_potential_energy(positions)
    delta_energy = new_energy - total_energy

    # Metropolis criterion
    if delta_energy > 0 and np.exp(-delta_energy / kT) < np.random.rand():
        # Reject the move, revert to the old position
        positions[molecule_idx] = old_position
    else:
        # Accept the move, update total energy
        total_energy = new_energy

    return positions, total_energy

total_energy = total_energy = total_potential_energy(positions)
num_steps = 200

# Perform the simulation
positions_history = [positions.copy()]
for step in range(num_steps * num_molecules):
    positions, total_energy = metropolis_step(positions, total_energy)
    positions_history.append(positions.copy())




# Animation
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1])

def update(frame):
    positions = positions_history[frame * num_molecules]
    scat.set_offsets(positions)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)

ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Simulation of Molecules with Lennard-Jones Potential')

plt.show()
