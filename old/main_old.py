from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import combinations
from matplotlib import pyplot as plt

from toy_old import HamiltonianEffectiveVCE, HamiltonianToyModel, e

from sys import path
path.append(b'/Users/Alpha/workspace/TRIUMF/imsrg_mass_plots/src')

# noinspection PyPep8,PyPep8,PyUnresolvedReferences
from plotting import plot_the_plots

N_MAX = 0
K_RANGE = range(2, 17)
V0 = 1
HW = 1
CORE = 2

plots = list()
for a_prescription in [(4, 5, 6), (6,)*3,
                       (2, 3, 4),
                       (3, 4, 5)
                       ]:
    print('=' * 80)
    print(a_prescription)
    print('=' * 80)
    a2, a3, a4 = a_prescription
    h2 = HamiltonianToyModel(a=a2, v0=V0, hw=HW)
    h3 = HamiltonianToyModel(a=a3, v0=V0, hw=HW)
    h4 = HamiltonianToyModel(a=a4, v0=V0, hw=HW)
    h_eff = HamiltonianEffectiveVCE(h2=h2, h3=h3, h4=h4, n_max=N_MAX,
                                    core_size=CORE)
    x = list()
    y = list()
    const_list = list()
    const_dict = {'presc': str(a_prescription)}
    for k in K_RANGE:
        print('\nk = {}:'.format(k))
        print('.' * 80)
        h_exact = HamiltonianToyModel(a=k, v0=V0, hw=HW)
        x.append(k)
        eff = e(k=k, hamiltonian=h_eff, n_max=N_MAX)
        exc = e(k=k, hamiltonian=h_exact, n_max=N_MAX)
        y.append(eff - exc)
        print('E_eff = {}'.format(eff))
        print('E_exact = {}'.format(exc))
        print()
    print()
    plots.append((x, y, const_list, const_dict))
x = list()
y = list()
const_list = list()
const_dict = {'presc': 'A'}
for k in K_RANGE:
    x.append(k)
    y.append(0)
plots.append((x, y, const_list, const_dict))


plot_the_plots(plots, label='{a}',
               title=('Ground state toy model energies calculated for '
                      'different A prescriptions'),
               xlabel='Number of particles',
               ylabel='E_valence - E_exact',
               get_label_kwargs=lambda plot, i: {'a': plot[3]['presc']},
               include_legend=True)

plt.show()
