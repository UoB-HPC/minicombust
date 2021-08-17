from mpi4py import MPI
from typing import Iterable, Dict, List
from numpy.typing import NDArray
from minicombust.data_structures import T, LocalId, GlobalId
import numpy as np

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
nproc = comm.Get_size()
print('Hello from rank {}'.format(my_rank))


# In lieu of partitioning and a real mesh, we'll use this simple example
# from the image in the docs

# Who each ranks neighbours are
mesh_neighbour_ranks = {
    0:[1],
    1:[0,2],
    2:[1,3],
    3:[2]
}

# How many of a rank's ghost cells are owned by each neighbour 
mesh_neighbour_rank_to_num_ghost_cells = {
    0: {1:2},
    1: {0:2, 2:2},
    2: {1:2, 3:2},
    3: {2:2}
}

# How many of a rank's border cells are held as ghost cells by each neighbour 
mesh_neighbour_rank_to_num_border_cells = {
    0 : {1:2},
    1 : {0:2, 2:2},
    2 : {1:2, 3:2},
    3 : {2:2}
}

# Which global cell ids must be received from which neighbour, for each rank 
mesh_neighbour_rank_to_ghost_global_cell_ids = {
    0: {1: [2,3]},
    1: {0: [0,1], 2: [4,5]},
    2: {1: [2,3], 3: [6,7]},
    3: {2: [4,5]}
}

# Which global cell ids must be sent to which neighbour, for each rank
mesh_neighbour_rank_to_border_global_cell_ids = {
    0: {1: [0,1]},
    1: {0: [2,3], 2: [2,3]},
    2: {1: [4,5], 3: [4,5]},
    3: {2: [6,7]}
}

# map from global cell id for local cell id for each rank (only bother with border cells and ghost cells)
mesh_global_cell_to_local_cell_id = {
    0: {0:0, 1:1, 2:2, 3:3},
    1: {0:0, 1:1, 2:2, 3:3, 4:4, 5:5},
    2: {2:0, 3:1, 4:2, 5:3, 6:4, 7:5},
    3: {4:0, 5:1, 6:2, 7:3}
}

# map from local cell id for global cell id for each rank (only bother with border cells and ghost cells)
mesh_local_cell_to_global_cell_id = {
    0: {0:0, 1:1, 2:2, 3:3},
    1: {0:0, 1:1, 2:2, 3:3, 4:4, 5:5},
    2: {0:2, 1:3, 2:4, 3:5, 4:6, 5:7},
    3: {0:4, 1:5, 2:6, 3:7}
}

# This is all of the mesh that each local process needs to store:
neighbour_ranks: List[int] = mesh_neighbour_ranks[my_rank]
neighbour_rank_to_num_ghost_cells: Dict[int, np.int64] = mesh_neighbour_rank_to_num_ghost_cells[my_rank]
neighbour_rank_to_num_boundary_cells: Dict[int, np.int64] = mesh_neighbour_rank_to_num_border_cells[my_rank]
neighbour_rank_to_boundary_global_cell_ids = mesh_neighbour_rank_to_border_global_cell_ids[my_rank]
neighbour_rank_to_ghost_global_cell_ids = mesh_neighbour_rank_to_ghost_global_cell_ids[my_rank]
global_cell_to_local_cell_id: Dict[GlobalId, LocalId] = mesh_global_cell_to_local_cell_id[my_rank]
local_cell_to_global_cell_id: Dict[LocalId, GlobalId] = mesh_local_cell_to_global_cell_id[my_rank]
neighbour_rank_to_received_cell_values: Dict[int, NDArray] = {i: np.zeros((len(neighbour_rank_to_ghost_global_cell_ids[i])), dtype=T) for i in neighbour_ranks}
cells: NDArray[T] = np.zeros((len(local_cell_to_global_cell_id),), dtype=T)

# Just some dummy values for this example
cells[:] = my_rank

def gather_local_cell_values_for_ids(ids : Iterable[LocalId]) -> NDArray[T]:
    return np.take_along_axis(cells, np.array(list(ids)), axis=None)

def send_to_single_neighbour(dst_rank: int):
    global_cell_ids = neighbour_rank_to_boundary_global_cell_ids[dst_rank]
    local_cell_ids = (global_cell_to_local_cell_id[i] for i in global_cell_ids)
    src_cell_vals = gather_local_cell_values_for_ids(local_cell_ids)
    #print("{} sends border to {}: {}".format(my_rank, dst_rank, len(src_cell_vals)))
    return comm.Isend(src_cell_vals, dest=dst_rank)

def receive_from_single_neighbour(src_rank: int):
    arr = neighbour_rank_to_received_cell_values[src_rank]
    #print("{} receives ghost from {}: {}".format(my_rank, src_rank, len(arr)))
    return comm.Irecv(arr, source=src_rank)

def perform_updates_on_local_cells():
    # Insert fluid solver of your choice here
    pass

def scatter_received_halos_to_cells():
    def scatter_single_rank(rank):       
        global_cell_ids = neighbour_rank_to_ghost_global_cell_ids[rank]
        local_cell_ids = np.array([global_cell_to_local_cell_id[i] for i in global_cell_ids])
        np.put_along_axis(cells, local_cell_ids, neighbour_rank_to_received_cell_values[rank], axis=None)

    for rank in neighbour_ranks:
        scatter_single_rank(rank)


def halo_exchange():
    # We use non-blocking comms for the exchange
    # to avoid deadlock. Dependencies are mesh-dependent
    # and so are difficult to create perfect blocking comms for.
    # But this might be worth optimising more
    reqs = []
    for rank in range(len(neighbour_ranks)):
        reqs.append(receive_from_single_neighbour(neighbour_ranks[rank]))
        reqs.append(send_to_single_neighbour(neighbour_ranks[rank]))
    MPI.Request.Waitall(reqs)
    scatter_received_halos_to_cells()

def main_loop():
    halo_exchange()
    for timestep, idx in enumerate(range(10)):
        perform_updates_on_local_cells()
        halo_exchange()
        if my_rank == 0:
            print("Timestep {} complete".format(idx))

def main():
    main_loop()

if __name__ == "__main__":
    main()