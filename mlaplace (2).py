import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
from mpi4py import MPI

ROWS, COLUMNS = 1000, 1000
MAX_TEMP_ERROR = 0.01

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rows_per_process = ROWS // size
extra_rows = ROWS % size

if rank < extra_rows:
    rows_per_process += 1

temperature_local = np.zeros((rows_per_process + 2, COLUMNS + 2))
last_temp_local = np.zeros((rows_per_process + 2, COLUMNS + 2))

def initialize_temperature():
    for i in range(rows_per_process):
        global_i = i + rank * (ROWS // size) + min(rank, extra_rows)
        temperature_local[i+1, COLUMNS+1] = 100 * math.sin((math.pi/2/ROWS) * global_i)
    
    if rank == size - 1:
        for j in range(COLUMNS + 2):
            temperature_local[rows_per_process+1, j] = 100 * math.sin((math.pi/2/COLUMNS) * j)

initialize_temperature()
last_temp_local[:] = temperature_local[:]

if rank == 0:
    max_iterations = int(input("Maximum iterations: "))
else:
    max_iterations = None

max_iterations = comm.bcast(max_iterations, root=0)

dt = 100
iteration = 1

while dt > MAX_TEMP_ERROR and iteration <= max_iterations:
    for i in range(1, rows_per_process + 1):
        for j in range(1, COLUMNS + 1):
            temperature_local[i, j] = 0.25 * (last_temp_local[i+1, j] + last_temp_local[i-1, j] +
                                              last_temp_local[i, j+1] + last_temp_local[i, j-1])

    # Exchange between PEs with ghost rows
    if rank > 0:
        comm.Send(temperature_local[1], dest=rank-1, tag=0)
        comm.Recv(temperature_local[0], source=rank-1, tag=1)
    if rank < size - 1:
        comm.Send(temperature_local[rows_per_process], dest=rank+1, tag=1)
        comm.Recv(temperature_local[rows_per_process+1], source=rank+1, tag=0)

    dt_local = np.max(np.abs(temperature_local - last_temp_local))
    dt = comm.allreduce(dt_local, op=MPI.MAX)

    last_temp_local[:] = temperature_local[:]

    if rank == 0 and iteration % 100 == 0:
        print(f"Iteration {iteration}, dt = {dt}")

    iteration += 1

# Gather results
gathered_data = comm.gather(temperature_local[1:rows_per_process+1, 1:COLUMNS+1], root=0)

if rank == 0:
    full_temperature = np.vstack(gathered_data)
    print(f"Iterations completed: {iteration-1}")
    print(f"Final dt: {dt}")
    plt.imshow(full_temperature[:, 1:COLUMNS+1], norm=mcolors.LogNorm(vmin=0.1, vmax=50, clip=True))
    plt.colorbar()
    print("Combined plot saved as 'combined_temp_distr.png'")
    plt.savefig("combined_temp_distr.png")

# Debug: Print final local temperatures for each rank
print(f"Rank {rank}: Final local temperature shape: {temperature_local.shape}")
print(f"Rank {rank}: Final local temperature sum: {np.sum(temperature_local)}")
