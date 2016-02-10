from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from matplotlib import pyplot as plt

from toy import HamiltonianEffectiveVCE, HamiltonianToyModel, e

from sys import path
path.append(b'/Users/Alpha/workspace/TRIUMF/imsrg_mass_plots/src')

# noinspection PyPep8,PyPep8,PyUnresolvedReferences
from plotting import plot_the_plots

V0 = 1
N_MAX = 10

a_prescriptions = list([lambda A: (4, 5, 6),
                        lambda A: (6, 6, 6),
                        lambda A: (A, A, A)])


# noinspection PyPep8Naming
def exact(A):
    return A, A, A

plots = list()
for p in a_prescriptions:
    x = list()
    y = list()
    const_list = list()
    const_dict = {'presc': p}
    for A in range(4, 17):
        h = HamiltonianToyModel(A=A, v0=V0, get_A_fn=p)
        heff = HamiltonianEffectiveVCE(hamiltonian=h)
        h_ex = HamiltonianToyModel(A=A, v0=V0, get_A_fn=exact)
        heff_ex = HamiltonianEffectiveVCE(hamiltonian=h_ex)
        x.append(A)
        y.append(e(n=A, hamiltonian=heff, n_max=N_MAX, _idx=0) -
                 e(n=A, hamiltonian=heff_ex, n_max=N_MAX, _idx=0))
    plots.append((x, y, const_list, const_dict))

plot_the_plots(plots, label='{a}',
               title=('Ground state toy model energies calculated for '
                      'different A prescriptions'),
               xlabel='Number of particles',
               ylabel='E_valence - E_exact',
               get_label_kwargs=lambda plot, i: {'a': plot[3]['presc']},
               include_legend=True)

plt.show()
