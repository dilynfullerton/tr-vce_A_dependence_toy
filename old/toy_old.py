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
                return n if n < 1 else 1
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

    def __init__(self, a, v0=1, hw=1, t_i=_t_i, t_ij=_t_ij):
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
        self.a = a
        self.v0 = v0
        self.hw = hw
        self.t_i = t_i
        self.t_ij = t_ij

    def __call__(self, occ_num):
        """Apply the Hamiltonian to a state vector in OccupationNumber
        representation
        :return: The OccupationNumber representation of the results of applying
        the Hamiltonian to the given state vector
        """
        k = len(occ_num)
        s = 0
        for i in range(1, k + 1):
            ai = FermionAnnihilationOperator(i=i)
            aii = ai.adjoint()
            s += self.t_i(i, self.hw) * aii(ai(occ_num))
        s *= 1 - 1 / self.a
        for i, j in combinations(range(1, k + 1), 2):
            ai = FermionAnnihilationOperator(i=i)
            aii = ai.adjoint()
            aj = FermionAnnihilationOperator(i=j)
            aji = aj.adjoint()
            s += ((self.v0 - (1 / self.a) * self.t_ij(i, j)) *
                  aii(aji(aj(ai(occ_num)))))
        return s


def e(k, hamiltonian, n_max=0):
    """Given a Hamiltonian (HamiltonianToyModel), and an n-value, calculate
    the ground state energy En for an n-body system
    :param _idx:
    :param k: The size of the system
    :param hamiltonian: The Hamiltonian to apply to the system
    :param n_max: The maximum size of the system
    :return: A scalar value representing the energy
    """
    state = OccupationNumber(n_max=n_max, a=k)
    eig = hamiltonian(state)
    if eig == 0:
        return eig
    else:
        return eig.scalar


class HamiltonianEffectiveVCE:
    def __init__(self, h2, h3, h4, n_max=0, core_size=2):
        self.h2 = h2
        self.h3 = h3
        self.h4 = h4
        print('a2 = {}'.format(self.h2.a))
        print('a3 = {}'.format(self.h3.a))
        print('a4 = {}'.format(self.h4.a))
        print()
        self.n_max = n_max
        self.core_size = core_size
        self.e_core = e(k=self.core_size, hamiltonian=self.h2, n_max=self.n_max)
        self.e_p = e(k=self.core_size + 1,
                     hamiltonian=self.h3,
                     n_max=self.n_max) - self.e_core
        self.v_eff = (e(k=self.core_size + 2,
                        hamiltonian=self.h4,
                        n_max=self.n_max) -
                      2 * self.e_p - self.e_core)
        print('E_core = {}'.format(self.e_core))
        print('e_p = {}'.format(self.e_p))
        print('v_eff = {}'.format(self.v_eff))

    def __call__(self, occ_num):
        k = len(occ_num)
        s = self.e_core * occ_num
        print(self.e_core)
        print(s)
        for p in range(3, k + 1):  # valence space
            ap = FermionAnnihilationOperator(i=p)
            api = ap.adjoint()
            s += self.e_p * api(ap(occ_num))
        print(s)
        for p, q in combinations(range(3, k + 1), 2):
            ap = FermionAnnihilationOperator(i=p)
            api = ap.adjoint()
            aq = FermionAnnihilationOperator(i=q)
            aqi = aq.adjoint()
            s += self.v_eff * api(aqi(aq(ap(occ_num))))
        print(s)
        return s
