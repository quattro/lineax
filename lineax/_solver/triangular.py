# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing_extensions import TypeAlias

import jax.experimental.sparse as sparse
import jax.lax as lax
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._operator import (
    AbstractLinearOperator,
    has_unit_diagonal,
    is_lower_triangular,
    is_upper_triangular,
)
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_TriangularState: TypeAlias = tuple[Array, bool, bool, PackedStructures, bool]


class Triangular(AbstractLinearSolver[_TriangularState], strict=True):
    """Triangular solver for linear systems.

    The operator should either be lower triangular or upper triangular.
    """

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Triangular` may only be used for linear solves with square matrices"
            )
        if not (is_lower_triangular(operator) or is_upper_triangular(operator)):
            raise ValueError(
                "`Triangular` may only be used for linear solves with triangular "
                "matrices"
            )
        return (
            operator.as_matrix(),
            is_lower_triangular(operator),
            has_unit_diagonal(operator),
            pack_structures(operator),
            False,  # transposed
        )

    def compute(
        self, state: _TriangularState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            trans = "T"
        else:
            trans = "N"
        solution = jsp.linalg.solve_triangular(
            matrix, vector, trans=trans, lower=lower, unit_diagonal=unit_diagonal
        )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _TriangularState, options: dict[str, Any]):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (
            matrix,
            lower,
            unit_diagonal,
            transposed_packed_structures,
            not transpose,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _TriangularState, options: dict[str, Any]):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        conj_state = (
            matrix.conj(),
            lower,
            unit_diagonal,
            packed_structures,
            transpose,
        )
        conj_options = {}
        return conj_state, conj_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


Triangular.__init__.__doc__ = """**Arguments:**

Nothing.
"""



_SparseTriangularState: TypeAlias = tuple[sparse.BCOO, bool, bool, PackedStructures, bool]


def _forward_triangular_solve_edges(bcoo_indices, bcoo_data, b):
    """
    Perform a forward triangular solve update using a sequential, edge-by-edge update.

    For each edge in the sorted BCOO arrays, update:

         b[i, :] += -bcoo_data[ndx] * b[j, :]

    where (i, j) = bcoo_indices[ndx], i.e. i is the destination row and j is the source row.
    The minus sign ensures that we subtract off the contributions, as in the standard
    forward substitution algorithm:

         x[i] = b[i] - sum_{j < i} L[i,j] * x[j]

    Parameters:
      bcoo_indices: int array of shape (nnz, 2) with each row [i, j] (destination, source)
      bcoo_data:    float array of shape (nnz,) containing the multipliers.
      b:            Dense array of shape (num_nodes, vector_length), initially holding the RHS.

    Returns:
      The updated dense array b (which holds the solution x).
    """
    nnz = bcoo_indices.shape[0]

    def body_fun(ndx, b):
        # Extract destination (i) and source (j) for the current edge.
        i = bcoo_indices[ndx, 0]
        j = bcoo_indices[ndx, 1]
        # For a forward triangular solve with unit diagonal, we subtract the contribution:
        update = -bcoo_data[ndx] * b[j]
        # Update row i by adding the (negative) contribution.
        b = b.at[i].add(update)
        return b

    # Process the edges sequentially.
    b = lax.fori_loop(0, nnz, body_fun, b)
    return b


class SparseTriangular(AbstractLinearSolver[_SparseTriangularState], strict=True):
    """Sparse Triangular solver for linear systems.

    The operator should either be sparse lower triangular or sparse upper triangular.
    """

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`SparseTriangular` may only be used for linear solves with sparse square matrices"
            )
        if not (is_lower_triangular(operator) or is_upper_triangular(operator)):
            raise ValueError(
                "`SparseTriangular` may only be used for linear solves with sparse triangular "
                "matrices"
            )
        return (
            operator.matrix,
            is_lower_triangular(operator),
            has_unit_diagonal(operator),
            pack_structures(operator),
            False,  # transposed
        )

    def compute(
        self, state: _SparseTriangularState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            trans = "T"
        else:
            trans = "N"
        solution = _forward_triangular_solve_edges(matrix.indices, matrix.data, vector)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _SparseTriangularState, options: dict[str, Any]):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (
            matrix,
            lower,
            unit_diagonal,
            transposed_packed_structures,
            not transpose,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _TriangularState, options: dict[str, Any]):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        conj_state = (
            matrix.conj(),
            lower,
            unit_diagonal,
            packed_structures,
            transpose,
        )
        conj_options = {}
        return conj_state, conj_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False


Triangular.__init__.__doc__ = """**Arguments:**

Nothing.
"""