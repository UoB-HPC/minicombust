import numba
import numpy as np
from typing import Final
from numpy.typing import NDArray
from .data_structures import Geometry, T
from .utils import minmax

def limit_slope(Φ: NDArray[T], dΦdX: NDArray[T], mesh: Final[Geometry]) -> NDArray[T]:
    """
        Notes
        -----
        Adapted from Dolfyn gradients.f90, GradientPhiLimiterVenkatarishnan
        MiniCombust only supports "approach 3" (using nodes instead of face centres for limiter)
    """
    assert(len(Φ) == mesh.num_cells + mesh.num_boundaries)
    assert(Φ.ndim == 1)  # It's a 1D tensor

    assert(Φ.ndim == 1)  # It's a 2D tensor
    assert(dΦdX.shape[0] == Φ.shape[0])  # that's the same length as Φ
    assert(dΦdX.shape[1] == 3)  # Its nx3

    cells = Geometry.cells

    for local_cell_id, cell in enumerate(mesh.cells):
        Φ_for_this_cell = Φ[local_cell_id]
        neighbour_Φs = Φ[cells.neighbours_with_offsetted_boundary_ids(
            local_cell_id)]
        Φ_min, Φ_max = minmax(neighbour_Φs)

        # Determine the smallest and largest deltas to neighbours for this cell
        Δ_min = Φ_min - Φ_for_this_cell
        Δ_max = Φ_max - Φ_for_this_cell

        # Determine smallest limiter α considering slope to vertex coords
        surrounding_vertices_coords = cells.vertex_coords(local_cell_id)
        cell_centre = cells.coords[local_cell_id]
        ds = surrounding_vertices_coords - cell_centre
        Δ_faces = np.dot(dΦdX[local_cell_id, :], ds)

        @numba.njit
        def venkatarishnan_factor(Δ_faces: NDArray[T], Δ_min: T, Δ_max: T):
            α = np.arraylike(Δ_faces)
            min_α = np.iinfo(T).max  # i.e. MAX_FLOAT
            for Δ_face, i in enumerate(Δ_faces):
                if np.abs(Δ_face) < 1.e-6:
                    r = 1000.0
                elif Δ_face > 0.0:
                    r = Δ_max / Δ_face
                else:
                    r = Δ_min / Δ_face
                α = (r**2+2.0*r)/(r**2+r+2.0)
                min_α = α if α < min_α else min_α
            return min_α

        α = venkatarishnan_factor(Δ_faces, Δ_min, Δ_max)
        dΦdX[local_cell_id, :] *= α
    return


NumPassesOfGaussGradientEstimation: Final[int] = 2.
def gradient(mesh: Geometry, Φ: NDArray[np.float64]) -> NDArray[np.float64]:
    """
        Notes
        -----
        Based on GradientPhiGauss and GradientPhi in Dolfyn (gradients.f90)
    """
    assert(Φ.ndim == 1)  # 1D array of length mesh.numCells + mesh.numBoundaries
    assert(len(Φ) == mesh.num_cells + mesh.num_boundaries)
    # dΦdX will have 3 elems for every one of Φ
    dΦdX = np.zeros((Φ.shape[0], 3))
    dΦdX_corrections = np.zeros((Φ.shape[0], 3))

    for _ in range(NumPassesOfGaussGradientEstimation):
        for face in mesh.faces:
            cell1, cell2 = face.adjacent_cells
            is_boundary_face = cell2 is None
            if not is_boundary_face:
                λ_cell1, λ_cell2 = 1 - face.interpolation_factor, face.interpolation_factor
                # TODO is it worth precomputing this weighted corrected coord and storing in face?
                # Correction to face using interpolation factor
                corrected_coords = cell1.coords * λ_cell1 + cell2.coords * λ_cell2
                dΦdX_corrected = dΦdX_corrections[cell1.local_id, :] * \
                    λ_cell1 + dΦdX_corrections[cell2.local_id, :] * λ_cell2

                # Now gradient at shifted position is known
                # Correct the value at the cell face centre
                Φ_face = Φ[cell1.local_id] * λ_cell1 + \
                    Φ[cell2.local_id] * λ_cell2  # standard
                Δ = np.dot(dΦdX_corrected, face.centre_coords -
                           corrected_coords)  # correction
                Φ_face += Δ

                # now only the value at the face center is known
                # multiply it by the area-components
                # this is basically Gauss' theorem
                dΦdX[cell1.local_id, :] += Φ_face * face.normal_components
                dΦdX[cell2.local_id, :] -= Φ_face * face.normal_components

            else:  # it's a boundary face
                Φ_face = Φ[mesh.num_cells + face.boundary_id]
                dΦdX[cell1.local_id, :] += Φ_face * face.normal_components
        dΦdX_corrections[:] = dΦdX
        # OR for  under relaxation
        # dΦdX_corrections[:] = dΦdX_corrections[:] + 0.95*( dΦdX - dΦdX_corrections)

    # normalise by cell volume dΦdX
    inv_cell_volumes = [1. / mesh.cells[i].volume for i in mesh.cells]
    dΦdX *= inv_cell_volumes

    limit_slope(Φ, dΦdX, mesh)

    return dΦdX
