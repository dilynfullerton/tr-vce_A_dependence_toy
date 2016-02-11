from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sys import path

from matplotlib import pyplot as plt

from toy import HamiltonianToy, HamiltonianToyEffective
from toy import get_a_exact as exact
from toy import custom_a_prescription as custom

path.append(b'/Users/Alpha/workspace/TRIUMF/imsrg_mass_plots/src')
# noinspection PyPep8,PyUnresolvedReferences
from plotting import plot_the_plots

A_PRESCRIPTIONS = [exact,
                   custom(4, 5, 6),
                   custom(6, 6, 6)
                   ]
K_RANGE = range(4, 17)
VALENCE_SPACE = range(5, 18)
HW = 1
V0 = 1
T2 = 0


def plot_a_prescriptions(a_prescriptions=A_PRESCRIPTIONS,
                         k_range=K_RANGE,
                         valence_space=VALENCE_SPACE,
                         v0=V0, hw=HW, t2=T2):
    plots = list()
    for ap in a_prescriptions:
        x = list()
        y = list()
        const_list = list()
        const_dict = {'presc': ap(0)[3]}
        for k in k_range:
            a2, a3, a4, ap_name = ap(k)

            h2 = HamiltonianToy(a2, v0=v0, hw=hw, t2=t2)
            h3 = HamiltonianToy(a3, v0=v0, hw=hw, t2=t2)
            h4 = HamiltonianToy(a4, v0=v0, hw=hw, t2=t2)

            e2 = h2.ground_state_energy(valence_space[0] - 1)
            e3 = h3.ground_state_energy(valence_space[0])
            e4 = h4.ground_state_energy(valence_space[0] + 1)

            e_core = e2
            e_p = e3 - e2
            v_eff = e4 - 2 * e3 + e2
            h_eff = HamiltonianToyEffective(e_core, e_p, v_eff,
                                            valence_space=valence_space)
            h_exact = HamiltonianToy(a=k, v0=v0, hw=hw, t2=t2)
            e_eff = h_eff.ground_state_energy(k)
            e_exact = h_exact.ground_state_energy(k)

            x.append(k)
            y.append(e_eff - e_exact)
        plots.append((x, y, const_list, const_dict))

    plot_the_plots(plots, label='{a}',
                   title=('Ground state toy model energies calculated for '
                          'different A prescriptions'),
                   xlabel='Number of particles',
                   ylabel='E_valence - E_exact',
                   get_label_kwargs=lambda plot, i: {'a': plot[3]['presc']},
                   include_legend=True)

plot_a_prescriptions()
plt.show()
