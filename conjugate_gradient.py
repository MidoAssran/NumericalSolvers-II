# ----------------------------------------- #
# Conjugate Gradient Finite Difference Potential Solver
# ----------------------------------------- #
# Author: Mido Assran
# Date: Nov. 10, 2016
# Description: ConjugateGradientFiniteDifferencePotentialSolver determines
# the electric potential at all vertices in a finite element mesh
# of a coax.

import random
import numpy as np
from conductor_description import *
from utils import matrix_dot_matrix, matrix_transpose

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

DEBUG = False

class ConjugateGradientFiniteDifferencePotentialSolver(object):

    """
    :-------Instance Variables-------:
    :type _h: float -> The inter-mesh node spacing
    :type _num_x_points: float -> Number of mesh points in the x direction
    :type _num_y_points: float -> Number of mesh points in the y direction
    :type _potentials: np.array([float]) -> Electric potential at nodes
    """

    def __init__(self, h=0.02):
        """
        :type h: float
        :rtype: void
        """

        np.core.arrayprint._line_width = 200

        self._h = h

        self._conductor_indices = \
            self.map_coordinates_to_indices((INNER_COORDINATES[0],INNER_COORDINATES[1]))
        self._conductor_index_dimensions = \
            (INNER_HALF_DIMENSIONS[0] / self._h + 1, INNER_HALF_DIMENSIONS[1] / self._h + 1)

        # Create the finite difference system of linear equations
        self._A, self._b = self.create_fdm_equation()

        if DEBUG:
            print(self._A, "\n\n", self._b)


    # Helper function converts node indices to locations in the mesh
    def map_indices_to_coordinates(self, indices):
        """
        :type indices: (int, int)
        :rtype: (float, float)
        """
        h = self._h
        return (indices[0] * h , indices[1] * h)



    # Helper function that converts node locations in the mesh to indices
    def map_coordinates_to_indices(self, coordinates):
        """
        :type coordinates: (float, float)
        :rtype: (int, int)
        """
        h = self._h

        i, j = 0, 0
        x, y = 0, 0

        while (coordinates[0] - x) > (0.5 * h):
            x += h
            i += 1

        while (coordinates[1] - y) > (0.5 * h):
            y += h
            j += 1

        indices = (i, j)
        return indices


    def create_fd_grid(self):
        """
        :rtype: np.array([float, float])
        """
        h = self._h

        x_midpoint = INNER_COORDINATES[0] + INNER_HALF_DIMENSIONS[0]
        y_midpoint = INNER_COORDINATES[1] + INNER_HALF_DIMENSIONS[1]
        num_x_points = int(x_midpoint / h + 1)
        num_y_points = int(y_midpoint / h + 1)
        num_nodes = num_x_points * num_y_points

        # Initialize potentials matrix according to the boundary coniditions
        grid = np.empty((num_x_points, num_y_points))
        grid[:] = 0
        return grid


    def create_free_potentials_vector(self, fd_grid):
        """
        :type fd_grid: np.array([float, float])
        :rtype: (np.array([float]), list((int, int)))
        """
        h = self._h

        fp_map = []
        fpv = []
        y_i = 0

        while y_i < fd_grid.shape[1]:
            for x_i, _ in enumerate(fd_grid[:, y_i]):
                coordinates = self.map_indices_to_coordinates((x_i, y_i))

                # If not in a fixed potential conductor add to free vector
                if (not ((coordinates[0] >= INNER_COORDINATES[0])
                        and (coordinates[1] >= INNER_COORDINATES[1]))
                    and not (coordinates[0] == 0 or coordinates[1] == 0)):
                    fpv.append(0)
                    fp_map.append((x_i, y_i))
            y_i += 1

        fpv = np.array(fpv)
        return (fpv, fp_map)


    def map_node_to_indices(self, node_number):
        """
        :type node_number: int
        :rtype: (int, int)
        """
        return self.fp_map[node_number]

    def map_coordinates_to_node(self, coordiantes):
        indices = self.map_coordinates_to_indices(coordiantes)
        return self.map_indices_to_node(indices)

    def map_indices_to_node(self, indices):
        """
        :type indices: (int, int)
        :rtype: int
        """
        for i, v in enumerate(self.fp_map):
            if v == indices:
                return i
        return None

    def create_fd_equation_matrices(self, fd_grid, num_free_potentials):
        """
        :type fd_grid: np.array([float, float])
        :type num_free_potentials: float
        :rtype: (np.array([float, float]), np.array([float]))
        """

        w_c, h_c = self._conductor_index_dimensions
        i_c, j_c = self._conductor_indices
        num_x_points, num_y_points = fd_grid.shape
        A = np.empty([num_free_potentials, num_free_potentials])
        A[:] = 0.0
        b = np.empty([num_free_potentials])
        b[:] = 0.0

        for ref_p in range(num_free_potentials):

            A[ref_p, ref_p] = -4.0

            # Apply boundary conditions
            i, j = self.map_node_to_indices(ref_p)

            # Determine adjacent node numbers
            left_p = ref_p - 1
            right_p = ref_p + 1
            top_p = ref_p + (num_x_points - 1)
            bottom_p = ref_p - (num_x_points -1)
            if (j >= j_c):
                top_p = ref_p + (num_x_points - 1 - w_c)
            if (j == num_y_points - 1):
                bottom_p = ref_p - (num_x_points - 1 - w_c)


            # These might fail at the boundaries
            try:
                A[ref_p, top_p] = 1.0
            except:
                pass
            try:
                if bottom_p >= 0:
                    A[ref_p, bottom_p] = 1.0
            except:
                pass
            try:
                A[ref_p, right_p] = 1.0
            except:
                pass
            try:
                if left_p >= 0:
                    A[ref_p, left_p] = 1.0
            except:
                pass

            # Apply boundary conditions
            if (i == num_x_points - 1):
                # Apply neumann boundary conditions to A
                A[ref_p, left_p] += 1.0
                try:
                    A[ref_p, right_p] = 0.0
                except:
                    pass
            if (i == 1):
                # Apply dirichlet boundary conditions to b
                b[ref_p] -= 0
                try:
                    A[ref_p, left_p] = 0.0
                except:
                    pass
            if (i == i_c - 1) and (j >= j_c):
                # Apply dirichlet boundary conditions to b
                b[ref_p] -= CONDUCTOR_POTENTIAL
                try:
                    A[ref_p, right_p] = 0.0
                except:
                    pass
            if (i >= i_c) and (j == j_c - 1):
                # Apply dirichlet boundary conditions to b
                b[ref_p] -= CONDUCTOR_POTENTIAL
                try:
                    A[ref_p, top_p] = 0.0
                except:
                    pass
            if (j == num_y_points - 1):
                # Apply neumann bondary conditions to A
                A[ref_p, bottom_p] += 1.0
                try:
                    A[ref_p, top_p] = 0.0
                except:
                    pass
            if (j == 1):
                # Apply dirichlet boundary conditions to b
                b[ref_p] -= 0.0
                try:
                    A[ref_p, bottom_p] = 0.0
                except:
                    pass

        b = b.reshape(b.shape[0], 1)
        return A, b


    def create_fdm_equation(self):
        """
        :rtype: (np.array([float, float]), np.array([float]))
        """
        self.fd_grid = self.create_fd_grid()
        self.fp_v, self.fp_map = self.create_free_potentials_vector(self.fd_grid)
        A, b = self.create_fd_equation_matrices(fd_grid=self.fd_grid,
                                              num_free_potentials=len(self.fp_v)
                                              )
        return A, b


    def solve(self):
        """
        :rtype: [np.array([float]), np.array([float]), np.array([float])]
        """

        A = self._A
        b = self._b
        num_eigenvalues = A.shape[1]

        # Potentials
        x_h = []
        x = np.empty([A.shape[1], 1])
        x[:] = 0
        x_h.append(x)

        # Residuals
        r_h = []
        r = b - matrix_dot_matrix(A, x)
        r_h.append(r)

        # Search direction
        p_h = []
        p = r
        p_h.append(p)

        for k in range(num_eigenvalues):

            # Linear search
            alpha = (matrix_dot_matrix(matrix_transpose(p), r)
                    / matrix_dot_matrix(matrix_dot_matrix(matrix_transpose(p), A), p)
                    )[0,0]
            x = x + alpha * p

            # Find new search direction
            r = b - matrix_dot_matrix(A, x)
            beta =  -1.0 * (matrix_dot_matrix(matrix_dot_matrix(matrix_transpose(p), A), r)
                    / matrix_dot_matrix(matrix_dot_matrix(matrix_transpose(p), A), p)
                    )[0,0]
            p = r + beta * p

            # Log history
            x_h.append(x)
            r_h.append(r)
            p_h.append(p)

        return (x_h, r_h, p_h)

if __name__ == '__main__':
    print("\n", end="\n")
    print("# --------------- TEST --------------- #", end="\n")
    print("# -------- Conjugate Gradient -------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    cgfdps = ConjugateGradientFiniteDifferencePotentialSolver(h=0.02)
    potential_history, residual_history, search_history = cgfdps.solve()
    print("A:\n", cgfdps._A, end="\n\n")
    print("b:\n", cgfdps._b, end="\n\n")
    print("result = solve(A, b):\n", potential_history[-1], end="\n\n")
    print("# ------------------------------------ #", end="\n\n")
