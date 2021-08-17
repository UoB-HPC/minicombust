from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple, Final, Iterable
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix
from .utils import *
import numba

# Type Aliases

# For shorthand, we refer to the floating point type used to represent
# scalar values as T, which allows easy type checking using mypy. This
# also allows us to easily switch to, e.g. np.float32
T = np.float64

# Many items in this 3D code use an array of 3 Ts, so we create a type
# shorthand for it
T3 = NDArray[T]

# And also a convenience method to init a T3
def allocateT3() -> T3:
    return np.zeros((3,), dtype=T)


# Representing ids (pointers?). We could switch to int64 if that made sense
LocalId = np.int32
GlobalId = np.int64

NoneId = 0

class FluidType(Enum):
    SOLID = 0
    FLUID = 1
    SYMMETRY_PLANE = 2
    WALL = 3
    PRESSURE = 4
    CYCLIC = 5
    DOMAIN_BOUNDARY = 6


class BoundaryType(Enum):
    WALL = 1
    INLET = 2
    OUTLET = 3
    PRESSURE = 4
    SYMMETRY_PLANE = 5
    THIN = 6
    CYCLIC = 7
    DOMAIN_BOUNDARY = 8

# The bits of the A matrix
class CoefficientTerms:
    # reciprocal coefficients (for pressure correction), indexed by cell id
    r: NDArray[T]
    u: NDArray[T]  # u component indexed by cell id
    v: NDArray[T]  # v component indexed by cell id
    w: NDArray[T]  # w component indexed by cell id

# The bits of the S matrix
class SourceTerms:
    u: NDArray[T]  # u component indexed by cell id
    v: NDArray[T]  # v component indexed by cell id
    w: NDArray[T]  # w component indexed by cell id


class Storage:
    """
    A God object that holds the SoA definitions for all the data
    We use static methods in specific classes `Cells` etc to
    define utilities for traversing these
    """
    def init(self, num_cells: int, num_faces, num_vertices, MaxVerticesPerFace: Final[int]):
        self.cell_centre_coords = np.zeros((num_cells, 3), dtype=T)
        self.cell_volumes = np.zeros((num_cells,), dtype=T)
        self.cell_fluid_types = np.zeros((num_cells), dtype=np.int8)
        self.cell_local_id_to_global_id = np.zeros((num_cells,), dtype=GlobalId)
        self.cell_is_remote = np.zeros((num_cells,), dtype=bool)

        # Which cells are ajacent to which other cells (local or ghost only)
        self.cell_connectivity: spmatrix = csr_matrix()
        # boundaries are at index num_cells onwards
        # cell_connectivity contains a 0 or no entry where there is no adjacency between cells
        # contains the face id when it is set
        # we don't inddicate self-connectivity

    
        self.face_areas = np.zeros((num_faces,), dtype=T)


        self.vertex_coords = np.zeros((num_vertices, 3), dtype=T)
        self.vertex_connectivity: spmatrix = csr_matrix()

        self.face_to_adjacent_cells = np.zeros((num_faces, 2), dtype=LocalId)
        # Not every face is a boundary, so we use a sparse vector (single row). Face_ids are column indexes
        # values are the boundary id (ints)
        self.face_to_boundary_ids : spmatrix = csr_matrix((1, num_faces), dtype=LocalId)
        self.face_to_vertex_ids = np.zeros((num_vertices, MaxVerticesPerFace), dtype=LocalId)
        self.face_areas: NDArray[T] = np.zeros((num_faces,), dtype=T)
        self.face_normal_components: NDArray[T] = np.zeros((num_faces, 3), dtype=T)
        self.face_centre_coords: NDArray[T] = np.zeros((num_faces, 3), dtype=T)
        self.face_interpolation_factors: NDArray[T] = np.zeros((num_faces,), dtype=T)
        #self.face_rlencos: T  # impl. coeff.: area/|Xpn|/vect_cosangle(n,Xpn)
        #self.face_xnac: T3 = allocateT3()  # auxillary vectors TODO what is this?
        #self.face_xpac: T3 = allocateT3()  # TODO what is this?


        self.split: T  # TODO what is this?
        self.face_normals: NDArray[T]  # 2D


class Cells:
    """
    Utility functions for cell by local cell id
    """
    def __init__(self, storage: Storage, num_cells: np.int64):
        self.__storage = storage
        self.__num_cells = num_cells

    @numba.njit
    def global_id(self, local_id: LocalId) -> GlobalId:
        return self.__storage.cell_local_id_to_global_id[local_id]

    def cell_centre_coords(self, local_id: LocalId) -> T3:
        return self.__storage.cell_centre_coords[local_id]

    def volume(self, local_id: LocalId):
        return self.__storage.cell_volumes[local_id]

    def fluid_type(self, local_id: LocalId):
        return self.__storage.cell_fluid_types[local_id]

    def faces(self, local_id: LocalId) -> Iterable[LocalId]: # face ids
        tmp = self.__storage.cell_connectivity.getrow(local_id)
        return tmp[tmp.nonzero()]

    def neighours_without_boundaries(self, local_id: LocalId) -> Iterable[LocalId]: # cell_ids
        tmp = self.__storage.cell_connectivity.getrow(local_id)[:self.__num_cells]
        return tmp.nonzero()

    def neighbours_with_offsetted_boundary_ids(self, local_id: LocalId) -> Iterable[LocalId]: # offsets into a combined cell_id / boundary array
        tmp = self.__storage.cell_connectivity.getrow(local_id)
        return tmp.nonzero()    

    def vertex_coords(self, local_id: LocalId) -> Iterable[LocalId]:
        faces  = self.faces(local_id)



class Face:
    """
    Utility functions for face by local face id
    """
    def __init__(self, storage: Storage, num_faces: np.int64):
        self.__storage = storage
        self.__num_faces = num_faces

    def boundary_id(self, face_id: LocalId) -> Optional[LocalId]:
        tmp = self.__storage.face_to_boundary_ids[face_id]
        return None if tmp == NoneId else tmp
    
    def adjacent_cells(self, face_id: LocalId) -> Tuple[LocalId, Optional[LocalId]]:
        cells = self.__storage.face_to_adjacent_cells[face_id]
        return cells[0], None if cells[1] == NoneId else cells[1]

    def vertex_ids(self, face_id: LocalId) -> Iterable[LocalId]:
        vertices = self.__storage.face_to_vertex_ids[face_id]
        return [v for v in vertices if v != NoneId]


        self.area: T
        self.normal_components: T3 = allocateT3()
        self.centre_coords: T3 = allocateT3()
        self.interpolation_factor: T  # $\lambda$
        self.rlencos: T  # impl. coeff.: area/|Xpn|/vect_cosangle(n,Xpn)
        self.xnac: T3 = allocateT3()  # auxillary vectors TODO what is this?
        self.xpac: T3 = allocateT3()  # TODO what is this?


class Boundary:
    def init(self):
        self.face_id: id                     # Id of the face this boundary belongs to
        self.vertex_ids: List[id]
        self.region_id: id                  # region id as set in rtable TODO what is rtable
        # normal distance from cell face center to cell center
        self.distance_to_cell_centre: T3
        self.yplus: T                        # y+ TODO what is this?
        self.uplus: T                        # u+ TODO what is this?
        self.shear_stress: T3                # shear stress components
        self.normal_stress: T3               # normal stress components
        self.heat_transfer_coeff: T          # h
        self.heat_flux: T                    # q (in W/m2)
        self.local_wall_temperature: T       # symbol T in equations


class BoundaryRegion:
    def init(self):
        self.name: str
        self.region_type: BoundaryType
        self.velocity: T3 = allocateT3()     # uvw
        self.density: T = 1.205
        self.temperature: T = 293.            # symbol T in eqs
        self.pressure: T = 0.                # P
        self.resistance: T = 0.               # R
        self.turbulent_kinetic_energy: T = 1.e-6        # k
        self.turbulent_dissipation: T = 1.0             # \epsilon
        self.is_no_slip: bool = True                # slip/no_slip
        self.is_standard: bool = True                # standard/roughness
        self.elog: T = 9.                     # E-parameter
        self.ylog: T = 11.                    # + value where u+=y+ matches wall law
        self.roughness: T = 0.03              # z0
        self.is_split_flow_rate: bool = True        # split or fixed
        self.split_velocity: T = 1.0          # TODO what is this
        self.is_adiabatic: bool = True               # adiab or fixed temperature
        self.is_fixed_flux_temperature: bool = True  # fixed flux temperature
        self.num_boundaries_using_this_region: np.int32 = 0
        self.area: T

        self.total_mass_flow: T = 0.          # total mass flow in this region
        self.mass_flux_corr_factor: T = 1.   # TODO correlation or correction???
        self.prescribed_mass_flow: T = 0.     # "mffixed"
        self.is_prescribed_flux_mass: T = False
        self.is_table_of_profiles: T = False

        self.face_normal: T3 = allocateT3()
        self.face_tangent: T3 = allocateT3()

        self.is_perfect_matching: bool = True  # Perfect matching or search
        self.central_point: T3 = allocateT3()
        self.direction: T3 = np.array((1., 0., 0.), dtype=T)
        self.translation: T3 = allocateT3()
        self.angle: T = 0.  # rotation angle (=0.0 => translation)


class Geometry:
    def __init__(self):
        # Can these actually just be len(struct?)
        # These will always just be the local + ghost numbers!
        num_vertices: np.int64
        num_cells: np.int64 
        num_boundaries: np.int64
        num_regions: np.int64
        num_faces: np.int64
        num_int: np.int64  # ??? TODO what is this
        num_non_zero: np.int64
        num_outlets: np.int64


        storage = Storage(num_cells=num_cells, 
            num_faces=num_faces,
            num_verties = num_vertices)

        cells = Cells(storage, num_cells)
        
      
class Particle:
    def __init__(self):
        self.starting_coords: T3 = allocateT3()
        self.starting_velocity: T3 = allocateT3()
        self.current_coords: T3 = allocateT3()
        self.current_velocity: T3 = allocateT3()
        self.current_acceleration: T3 = allocateT3()
        self.starting_cell_id: id
        self.current_cell_id: id
        self.current_face: id
        self.current_density: Optional[T]
        self.diameter: T
        self.mass: T
        self.is_wall: bool


class ParticleTrackData:
    def __init__(self):
        self.x: T3 = allocateT3()    # coordinates
        # velocity: np.ndarray[ScalarType] = np.zeros((3,), type=ScalarType)
        self.v: T3 = allocateT3()
        self.cell: id                 # cell the particle is in
        self.time: T                  # travelling time since start


class Particles:
    # The actual variables
    tracks: List[List[ParticleTrackData]] = []
    particles: List[Particle]
    tmp_particles: List[Particle]


def initialise_precomputed_values(storage: Storage):
    # calculate cell volumes
    storage.cell_volumes = 0  # TODO
    # calculate face areas
