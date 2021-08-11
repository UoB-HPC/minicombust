from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

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
id = np.int32


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
    r: NDArray[T] # reciprocal coefficients (for pressure correction), indexed by cell id 
    u: NDArray[T] # u component indexed by cell id
    v: NDArray[T] # v component indexed by cell id
    w: NDArray[T] # w component indexed by cell id

# The bits of the S matrix
class SourceTerms:
    u: NDArray[T] # u component indexed by cell id
    v: NDArray[T] # v component indexed by cell id
    w: NDArray[T] # w component indexed by cell id

class Cell:
    def init(self):
        self.x: T3 = allocateT3()
        self.volume: T
        self.fluid_type: FluidType

class Face:
    def init(self):
        self.boundary_id: id
        self.first_cell_id: id
        self.second_cell_id: id
        self.vertex_ids: list[id]
        self.area: T
        self.normal_components: T3 = allocateT3()
        self.centre_coords: T3 = allocateT3()
        self.interpolation_factor: T # $\lambda$
        self.rlencos: T # impl. coeff.: area/|Xpn|/vect_cosangle(n,Xpn)
        self.xnac: T3 = allocateT3() # auxillary vectors TODO what is this?
        self.xpac: T3 = allocateT3() # TODO what is this?

class Boundary:
    def init(self):
        self.face_id: id                     # Id of the face this boundary belongs to 
        self.vertex_ids: list[id]           
        self.region_id : id                  # region id as set in rtable TODO what is rtable
        self.distance_to_cell_centre : T3    # normal distance from cell face center to cell center
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
        self.pressure : T = 0.                # P
        self.resistance: T = 0.               # R
        self.turbulent_kinetic_energy : T= 1.e-6        # k 
        self.turbulent_dissipation: T = 1.0             # \epsilon
        self.is_no_slip : bool = True                # slip/no_slip
        self.is_standard: bool = True                # standard/roughness
        self.elog: T = 9.                     # E-parameter
        self.ylog: T = 11.                    # + value where u+=y+ matches wall law
        self.roughness: T = 0.03              # z0         
        self.is_split_flow_rate: bool =  True        # split or fixed
        self.split_velocity: T = 1.0          # TODO what is this
        self.is_adiabatic: bool = True               # adiab or fixed temperature
        self.is_fixed_flux_temperature: bool = True  # fixed flux temperature
        self.num_boundaries_using_this_region: np.int32 = 0
        self.area: T

        self.total_mass_flow: T = 0.          # total mass flow in this region
        self.mass_flux_corr_factor: T  = 1.   # TODO correlation or correction???
        self.prescribed_mass_flow: T = 0.     # "mffixed"
        self.is_prescribed_flux_mass: T = False 
        self.is_table_of_profiles: T = False

        self.face_normal : T3 = allocateT3()
        self.face_tangent : T3 = allocateT3()

        self.is_perfect_matching : bool = True # Perfect matching or search
        self.central_point: T3 = allocateT3()
        self.direction: T3 = np.array((1., 0., 0.), dtype=T)
        self.translation: T3 = allocateT3()
        self.angle: T = 0. # rotation angle (=0.0 => translation)

class Geometry:
    # Can these actually just be len(struct?)
    num_vertices: np.int64
    num_cells: np.int64
    num_boundaries: np.int64
    num_regions: np.int64
    num_faces: np.int64
    num_int: np.int64 # ??? TODO what is this
    num_non_zero: np.int64
    num_outlets: np.int64

    split: T # TODO what is this?

    vertices: NDArray[T] # 2D Probably want to use scipy.sparse.csr_matrix instead? TODO What values are stored here?
    face_normals: NDArray[T] # 2D
    faces: list[Face]

    num_faces_per_cell: NDArray[np.int32] # number of faces for each cell (indexed by cell id)
    num_nodes_per_cell: NDArray[np.int32] # number of faces for each cell (indexed by node id)

    faceToCells: dict[Face, tuple[Cell, Cell]]
    cellToFaces: dict[Cell, list[Face]]

    cells: list[Cell]
    faces: list[Face]
    boundaries: list[Boundary]
    regions: list[BoundaryRegion]


class Particle:
    def __init__(self):
        self.starting_coords: T3 = allocateT3()
        self.starting_velocity: T3 = allocateT3()
        self.current_coords: T3 = allocateT3()
        self.current_velocity: T3 = allocateT3()
        self.current_acceleration: T3 = allocateT3()
        self.starting_cell_id : id
        self.current_cell_id : id
        self.current_face : id
        self.current_density : Optional[T]
        self.diameter : T
        self.mass: T
        self.is_wall: bool

class ParticleTrackData:
        def __init__(self):
            self.x : T3 = allocateT3()    # coordinates 
            self.v : T3 = allocateT3()    # velocity: np.ndarray[ScalarType] = np.zeros((3,), type=ScalarType)
            self.cell: id                 # cell the particle is in
            self.time: T                  # travelling time since start

class Particles:
    # The actual variables
    tracks: list[list[ParticleTrackData]] = []
    particles: list[Particle]
    tmp_particles: list[Particle]