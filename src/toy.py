from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import combinations

from occ_rep import FermionAnnihilationOperator
from occ_rep import OccupationNumber

DEGENERACY = [2, 4, 2, 6, 2, 4, 8, 4, 6, 2, 10, 6, 8, 2, 4]


def _t_i(i, hw):
    return (_n(i) + 1.5) * hw


def _n(i):
    n = 0
    k = 0
    while True:
        for j in range(2 * (n + 1) ** 2):  # DEGENERACY[n]): #2 * (2 * n + 1)):
            if k >= i - 1:
                return n
            k += 1
        else:
            n += 1


# noinspection PyUnusedLocal
def _t_ij(i, j):
    return 0


# noinspection PyPep8Naming
class HamiltonianToyModel:
    """Representation of the Hamiltonian for the toy model
    """

    def __init__(self, A, v0, hw=1,
                 t_i=_t_i, t_ij=_t_ij, A_c=None, A_v=None, get_A_fn=None,
                 core_size=2):
        """Initialize the representation of the Hamiltonian
        :param A: The A value for use in calculations
        :param v0: The v0 value
        :param hw: The hw frequency level, default 1 (relative)
        :param t_i: The method for weighting the single particle energy levels
        given occupation index i
        :param t_ij: The method for weighting particle-particle interactions
        given occupation numbers i and j
        :param A_c: The A value for use in calculating core energy
        :param A_v: The A value for use in calculating valence (single particle)
        energy
        :param core_size: The size of the core
        """
        self.A = A
        self.A_c = A_c
        self.A_v = A_v
        if get_A_fn is not None:
            self.A_c, self.A_v, self.A = get_A_fn(A)
        self.v0 = v0
        self.hw = hw
        self.t_i = t_i
        self.t_ij = t_ij
        self.core_size = core_size

    def __call__(self, occ_num):
        """Apply the Hamiltonian to a state vector in OccupationNumber
        representation
        :return: The OccupationNumber representation of the results of applying
        the Hamiltonian to the given state vector
        """
        core = self.core(occ_num)
        single_particle = self.single_particle(occ_num)
        interaction = self.interactions(occ_num)
        return (core + single_particle + interaction,
                core, single_particle, interaction)

    def core(self, occ_num):
        s = 0
        for i in range(1, self.core_size + 1):
            ai = FermionAnnihilationOperator(i)
            aii = ai.adjoint()
            if s == 0:
                s = aii(ai(occ_num)) * self.t_i(i, self.hw)
            else:
                s += aii(ai(occ_num)) * self.t_i(i, self.hw)
        return s * (1 - 1 / self.A_c)

    def single_particle(self, occ_num):
        s = 0
        for i in range(self.core_size + 1, len(occ_num) + 1):
            ai = FermionAnnihilationOperator(i)
            aii = ai.adjoint()
            if s == 0:
                s = aii(ai(occ_num)) * self.t_i(i, self.hw)
            else:
                s += aii(ai(occ_num)) * self.t_i(i, self.hw)
        return s * (1 - 1 / self.A_v)

    def interactions(self, occ_num):
        s = 0
        for i, j in combinations(range(1,
                                       len(occ_num) + 1), 2):
            ai = FermionAnnihilationOperator(i)
            aj = FermionAnnihilationOperator(j)
            aii = ai.adjoint()
            ajj = aj.adjoint()
            if s == 0:
                s = (aii(ajj(aj(ai(occ_num)))) *
                     (self.v0 - self.t_ij(i, j) / self.A))
            else:
                s += (aii(ajj(aj(ai(occ_num)))) *
                      (self.v0 - self.t_ij(i, j) / self.A))
        return s


def e(n, hamiltonian, n_max, _idx=0):
    """Given a Hamiltonian (HamiltonianToyModel), and an n-value, calculate
    the ground state energy En for an n-body system
    :param _idx:
    :param n: The size of the system
    :param hamiltonian: The Hamiltonian to apply to the system
    :param n_max: The maximum size of the system
    :return: A scalar value representing the energy
    """
    state = OccupationNumber(n_max=n_max, a=n)
    eig = hamiltonian(state)[_idx]
    if eig == 0:
        return eig
    else:
        return eig.scalar


class HamiltonianEffectiveVCE:
    def __init__(self, hamiltonian, e_core=None, e_step=None, v_eff=None,
                 core_size=2):
        self.h = hamiltonian
        self.A = self.h.A
        self.core_size = core_size
        self.e_core_fn = e_core
        self.e_step_fn = e_step
        self.v_eff_fn = v_eff
        if self.e_core_fn is None:
            def e_c(n_max):
                return e(2, self.h, n_max=n_max)

            self.e_core_fn = e_c
        if self.e_step_fn is None:
            def e_st(n_max):
                return e(3, self.h, n_max=n_max) - e(2, self.h, n_max=n_max)

            self.e_step_fn = e_st
        if self.v_eff_fn is None:
            def vf(n_max):
                return (e(4, self.h, n_max=n_max) -
                        e(3, self.h, n_max=n_max) -
                        self.e_step_fn(n_max))

            self.v_eff_fn = vf

    def __call__(self, occ_num):
        core = self.core(occ_num)
        single_particle = self.single_particle(occ_num)
        interaction = self.interactions(occ_num)
        return (core + single_particle + interaction,
                core, single_particle, interaction)

    def core(self, occ_num):
        return occ_num * self.e_core_fn(occ_num.n_max)

    def single_particle(self, occ_num):
        s = 0
        for p in range(self.core_size + 1, len(occ_num) + 1):
            a_p = FermionAnnihilationOperator(p)
            a_pi = a_p.adjoint()
            sp = a_pi(a_p(occ_num)) * self.e_step_fn(n_max=occ_num.n_max)
            if s == 0:
                s = sp
            else:
                s += sp
        return s

    def interactions(self, occ_num):
        s = 0
        for p, q in combinations(range(self.core_size + 1, len(occ_num) + 1),
                                 2):
            a_p = FermionAnnihilationOperator(p)
            a_pi = a_p.adjoint()
            a_q = FermionAnnihilationOperator(q)
            a_qi = a_q.adjoint()
            sp = a_pi(a_qi(a_q(a_p(occ_num))))
            if s == 0:
                s = sp
            else:
                s += sp
        return self.v_eff_fn(n_max=occ_num.n_max) * s
