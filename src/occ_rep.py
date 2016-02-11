from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class OccupationNumber:
    def __init__(self, n_max=0, a=0, occupied=None, scalar=1):
        self.n_max = n_max
        self.scalar = scalar
        self.a = a
        self.occ = occupied
        if self.occ is None:
            self.occ = [-1] + [1] * self.a + [0] * n_max  # ground state
        else:
            self.a = len(filter(lambda x: x == 1, self.occ))
            self.occ = self.occ + [0] * (self.n_max + self.a - len(self.occ))
            if self.occ[0] != -1:
                self.occ = [-1] + self.occ

    def __len__(self):
        return len(self.occ) - 1

    def __getitem__(self, item):
        return self.occ[item]

    def __add__(self, other):
        return self.__radd__(other)

    def __radd__(self, other):
        if (other == 0 or
                (isinstance(other, OccupationNumber) and other.scalar == 0)):
            return self
        elif self.scalar == 0:
            return other
        elif self.occ != other.occ:
            raise CannotAddOccupationNumbersException(
                'Cannot add occupation states {} and {}'.format(self, other))
        else:
            return OccupationNumber(n_max=self.n_max,
                                    a=self.a,
                                    occupied=self.occ,
                                    scalar=self.scalar + other.scalar)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    def __mul__(self, other):
        return self.__rmul__(other)

    def __rmul__(self, other):
        return OccupationNumber(n_max=self.n_max,
                                a=self.a,
                                occupied=self.occ,
                                scalar=self.scalar*other)

    def __str__(self):
        s = self.scalar
        if s == 0:
            return '0'
        s_str = '' if s == 1 else str(s)
        sep = '|'
        right = '>'
        occ = str(self.occ[1:]).strip('[]').replace(', ', '')
        return '{}{}{}{}'.format(s_str, sep, occ, right)

    def __iter__(self):
        return iter(list(self.occ[1:]))

    def __eq__(self, other):
        if isinstance(other, OccupationNumber):
            if self.scalar == 0:
                return 0 == other.scalar
            else:
                return self.scalar == other.scalar and self.occ == other.occ
        else:
            return self.scalar == 0 and other == 0

    def create(self, i):
        if i > len(self):
            raise ModelSpaceTooSmallException(
                'Cannot create a particle in state i = {}. Model space is '
                'limited to i in \{1, 2, ...{}\}.'.format(i, len(self.occ)))
        elif self[i] == 1:
            return OccupationNumber(n_max=self.n_max,
                                    a=self.a,
                                    occupied=self.occ,
                                    scalar=0)
        elif self[i] == 0:
            next_occ = list(self.occ)
            next_occ[i] = 1
            return OccupationNumber(n_max=self.n_max,
                                    a=self.a+1,
                                    occupied=next_occ,
                                    scalar=self.scalar)

    def annihilate(self, i):
        if i > len(self):
            raise ModelSpaceTooSmallException(
                ('Cannot create a particle in state i = {}.'.format(i) +
                 ' Model space is limited to i in {1, 2, ...' +
                 '{}'.format(len(self))) + '}.')
        elif self[i] == 0:
            return OccupationNumber(n_max=self.n_max,
                                    a=self.a,
                                    occupied=self.occ,
                                    scalar=0)
        elif self[i] == 1:
            next_occ = list(self.occ)
            next_occ[i] = 0
            return OccupationNumber(n_max=self.n_max,
                                    a=self.a-1,
                                    occupied=next_occ,
                                    scalar=self.scalar)


class CannotAddOccupationNumbersException(Exception):
    pass


class _FermionCAOperator(object):
    def __init__(self, i, adjoint_type):
        self.i = i
        self._adjoint_type = adjoint_type

    def adjoint(self):
        return self._adjoint_type(self.i)

    def anticommute(self, other):
        if isinstance(other, self._adjoint_type):
            return 1 if self.i == other.i else 0
        else:
            return 0


class FermionCreationOperator(_FermionCAOperator):
    def __init__(self, i):
        super(FermionCreationOperator,
              self).__init__(i, FermionAnnihilationOperator)

    def __call__(self, occ_num):
        return occ_num.create(self.i)


class FermionAnnihilationOperator(_FermionCAOperator):
    def __init__(self, i):
        super(FermionAnnihilationOperator,
              self).__init__(i, FermionCreationOperator)

    def __call__(self, occ_num):
        return occ_num.annihilate(self.i)


class ModelSpaceTooSmallException(Exception):
    pass
