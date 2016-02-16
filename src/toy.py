from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import combinations

from occ_rep import FermionAnnihilationOperator
from occ_rep import FermionOccupationNumber
from occ_rep import ModelSpaceTooSmallException


class _Hamiltonian(object):
    def __call__(self, state):
        self._operate_on(state)

    def _operate_on(self, state):
        raise NotImplemented()

    def energy(self, state):
        es = self._operate_on(state)
        if es == 0:
            return 0
        else:
            return es.scalar

    def ground_state_energy(self, k, n_max=0):
        return self.energy(
            FermionOccupationNumber(occupied=[1] * k, n_max=n_max))


def _t_i(i, n, hw):
    return (n(i) + 3 / 2) * hw


# noinspection PyUnusedLocal
def _t_ij(i, j, t_core, t_mix, t_val, valence_space):
    if j not in valence_space:
        return t_core
    elif i not in valence_space:
        return t_mix
    else:
        return t_val


def _n_i(i):
    return 0 if i <= 2 else 1  # todo figure out correct form


class HamiltonianToy(_Hamiltonian):
    """Toy model Hamiltonian to be applied to occupation state vectors
    """

    def __init__(self, a, v0, hw, valence_space, t_i=_t_i, t_ij=_t_ij, n_i=_n_i,
                 t_core=0, t_mix=0, t_val=0):
        self.a = a
        self.v0 = v0
        self.hw = hw
        self.valence = valence_space
        self._t_i = t_i
        self._t_ij = t_ij
        self._n_i = n_i
        self._t_core = t_core
        self._t_mix = t_mix
        self._t_val = t_val

    def _operate_on(self, state):
        k = len(state)
        s = 0
        for i in range(1, k + 1):
            ai = FermionAnnihilationOperator(i)
            ai_ = ai.adjoint()
            s += self._t_i(i=i, n=self._n_i, hw=self.hw) * ai_(ai(state))
        s *= (1 - 1 / self.a)
        for i, j in combinations(range(1, k + 1), 2):
            ai = FermionAnnihilationOperator(i)
            ai_ = ai.adjoint()
            aj = FermionAnnihilationOperator(j)
            aj_ = aj.adjoint()
            tij = self._t_ij(i=i, j=j,
                             t_core=self._t_core,
                             t_mix=self._t_mix,
                             t_val=self._t_val,
                             valence_space=self.valence)
            s += ((self.v0 - tij / self.a) * ai_(aj_(aj(ai(state)))))
        return s


class HamiltonianToyEffective(_Hamiltonian):
    """Effective Hamiltonian for the toy model based on an A-prescription
    """

    def __init__(self, e_core, e_p, v_eff, valence_space):
        self.e_core = e_core
        self.e_p = e_p
        self.v_eff = v_eff
        self.valence_space = valence_space

    def _operate_on(self, state):
        s0 = self.e_core * state
        s1 = 0
        for p in self.valence_space:
            ap = FermionAnnihilationOperator(p)
            ap_ = ap.adjoint()
            try:
                s1 += self.e_p * ap_(ap(state))
            except ModelSpaceTooSmallException:
                continue
        s2 = 0
        for p, q in combinations(self.valence_space, 2):
            ap = FermionAnnihilationOperator(p)
            ap_ = ap.adjoint()
            aq = FermionAnnihilationOperator(q)
            aq_ = aq.adjoint()
            try:
                s2 += ap_(aq_(aq(ap(state))))
            except ModelSpaceTooSmallException:
                continue
        s2 *= self.v_eff
        return s0 + s1 + s2


# A prescriptions
def get_a_exact(a):
    return (a,) * 3 + ('A_eff = A', '$A_{\mathrm{eff}} = A$',)


def custom_a_prescription(a, b, c):
    # noinspection PyUnusedLocal
    def get_a_custom(x):
        tup = (a, b, c)
        return tup + ('A_eff = {}'.format(tup),
                      '$A_{\mathrm{eff}} =' + ' {}$'.format(tup),)

    return get_a_custom
