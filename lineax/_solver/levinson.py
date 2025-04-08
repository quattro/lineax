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

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._operator import ToeplitzLinearOperator
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_LDState: TypeAlias = tuple[Array, PackedStructures]


class LevinsonDurbin(AbstractLinearSolver, strict=True):
    """Solver for Toeplitz systems.

    """

    def init(self, operator, options):
        del options
        if not isinstance(operator, ToeplitzLinearOperator):
            raise ValueError("LevinsonDurbin requires a ToeplitzLinearOperator")

        row_col = jnp.concatenate((operator.row[-1:0:-1], operator.column))
        packed_structures = pack_structures(operator)
        return row_col, packed_structures

    def compute(
            self,
            state: _LDState,
            vector: PyTree[Array],
            options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        row_col, packed_structures  = state
        vector = ravel_vector(vector, packed_structures)
        solution = _levinson(row_col, vector)
        solution = unravel_solution(solution, packed_structures)

        return solution, RESULTS.successful, {}

    def transpose(self, state: _LDState, options: dict[str, Any]):
        (q, r), transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (q, r), not transpose, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _LDState, options: dict[str, Any]):
        (q, r), transpose, structures = state
        conj_state = (
            (q.conj(), r.conj()),
            transpose,
            structures,
        )
        conj_options = {}
        return conj_state, conj_options

    def allow_dependent_columns(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        # We're able to pull an efficiency trick here.
        #
        # As we don't use a rank-revealing implementation, then we always require that
        # the operator have full rank.
        #
        # So if we have columns <= rows, then we know that all our columns are linearly
        # independent. We can return `False` and get a computationally cheaper jvp rule.
        return columns > rows

    def allow_dependent_rows(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return rows > columns


def _levinson(row_col, vector):
    # Determine the data type
    dtype = row_col.dtype
    n = vector.shape[0]

    # Initialize result arrays
    x = jnp.zeros(n, dtype=dtype)  # result
    g = jnp.zeros(n, dtype=dtype)  # workspace
    h = jnp.zeros(n, dtype=dtype)  # workspace

    #if row_col[n - 1] == 0: raise LinAlgError('Singular principal minor')

    # First step of the recursion
    x = x.at[0].set(vector[0] / row_col[n - 1])
    g = g.at[0].set(row_col[n - 2] / row_col[n - 1])
    h = h.at[0].set(row_col[n] / row_col[n - 1])

    if n == 1:
        return x

    n_indices = jnp.arange(n)

    # recursion loop for 1...(n-1)
    def _inner(m: int, carry):
        x, g, h = carry
        indices = n_indices[:m]

        # Compute the range of indices in a vectorized manner
        nmj_indices = n + m - (indices + 1)

        # Dot products
        x_num = -vector[m] + row_col[nmj_indices] @ x[indices]
        x_den = -row_col[n - 1] + row_col[nmj_indices] @ g[m - indices - 1]

        if x_den == 0:
            raise jnp.linalg.LinAlgError('Singular principal minor')

        x = x.at[m].set(x_num / x_den)
        x = x.at[indices].add(-(x[m] * g[m - indices - 1]))

        if m == n - 1:
            return x, g, h

        # Vectorized updates for g_num, h_num, and g_den
        g_num = -row_col[n - m - 2] + row_col[n + indices - m - 1] @ g[indices]
        h_num = -row_col[n + m] + row_col[n + m - indices - 1] @ h[indices]
        g_den = -row_col[n - 1] + row_col[n + indices - m - 1] @ h[m - indices - 1]

        #if g_den == 0.0:
        #    raise jnp.linalg.LinAlgError('Singular principal minor')

        g = g.at[m].set(g_num / g_den)
        h = h.at[m].set(h_num / x_den)

        # Update g and h
        m2 = (m + 1) // 2
        c1 = g[m]
        c2 = h[m]

        j_indices = jnp.arange(m2)  # Indices from 0 to m2-1
        k_indices = m - 1 - j_indices  # Indices from m-1 down to m2

        # Retrieve values for the current g and h
        gj = g[j_indices]
        gk = g[k_indices]
        hj = h[j_indices]
        hk = h[k_indices]

        # Vectorized update for g and h
        g = g.at[j_indices].add(-(c1 * hk))
        g = g.at[k_indices].add(-(c1 * hj))
        h = h.at[j_indices].add(-(c2 * gk))
        h = h.at[k_indices].add(-(c2 * gj))

        return x, g, h

    (x, _, _) = lax.fori_loop(1, n, _inner, (x, g, h))

    return x
