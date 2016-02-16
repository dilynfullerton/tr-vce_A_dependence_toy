from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sys import path
from itertools import combinations_with_replacement, permutations

from matplotlib import pyplot as plt

from toy import HamiltonianToy as H_ex
from toy import HamiltonianToyEffective as H_eff
from toy import get_a_exact as exact
from toy import custom_a_prescription as custom

path.append(b'../../imsrg_mass_plots/src')
# noinspection PyPep8,PyUnresolvedReferences
from plotting import plot_the_plots

A_PRESCRIPTIONS = [exact,
                   custom(4, 5, 6),
                   custom(6, 6, 6)
                   # custom(2, 3, 4),
                   # custom(4, 4, 4)
                   ]
K_RANGE = range(4, 17)
# K_RANGE = range(2, 17)
VALENCE_SPACE = range(5, 18)
# VALENCE_SPACE = range(3, 18)
HW = 1
V0 = 1
T_2 = 0
T_CORE = T_2
T_MIX = T_2
T_VAL = T_2
LATEX = True


def plot_a_prescriptions(a_prescriptions=A_PRESCRIPTIONS,
                         k_range=K_RANGE,
                         valence_space=VALENCE_SPACE,
                         v0=V0, hw=HW, t_core=T_CORE, t_mix=T_MIX, t_val=T_VAL,
                         use_latex=LATEX):
    """For each A-prescription, plot the difference between the energy based on
    the effective Hamiltonian generated from the A prescription and the energy
    based on the exact Hamiltonian.
    Note: k represents the actual mass number (i.e. the number of ones in the
    state vector), while a represents that used in the Hamiltonian
    :param use_latex:
    :param t_val:
    :param t_mix:
    :param t_core:
    :param a_prescriptions: A list of functions that take an actual mass
    number and produce the first three A values from it
    :param k_range: The range of ACTUAL mass numbers for which to evaluate the
    energy difference.
    :param valence_space: The range of values over which to evaluate the
    effective Hamiltonian
    :param v0: v0
    :param hw: hw
    """
    plots = list()
    for ap in a_prescriptions:
        x = list()
        y = list()
        const_list = list()
        if not use_latex:
            const_dict = {'presc': ap(0)[3]}
        else:
            const_dict = {'presc': ap(0)[4]}
        for k in k_range:
            a2, a3, a4 = ap(k)[:3]

            h2 = H_ex(a=a2, v0=v0, hw=hw, valence_space=valence_space,
                      t_core=t_core, t_mix=t_mix, t_val=t_val)
            h3 = H_ex(a=a3, v0=v0, hw=hw, valence_space=valence_space,
                      t_core=t_core, t_mix=t_mix, t_val=t_val)
            h4 = H_ex(a=a4, v0=v0, hw=hw, valence_space=valence_space,
                      t_core=t_core, t_mix=t_mix, t_val=t_val)

            e2 = h2.ground_state_energy(k=valence_space[0]-1)
            e3 = h3.ground_state_energy(k=valence_space[0])
            e4 = h4.ground_state_energy(k=valence_space[0]+1)

            e_core = e2
            e_p = e3 - e2
            v_eff = e4 - 2 * e3 + e2
            h_eff = H_eff(e_core=e_core, e_p=e_p, v_eff=v_eff,
                          valence_space=valence_space)
            h_exact = H_ex(a=k, v0=v0, hw=hw, valence_space=valence_space,
                           t_core=t_core, t_mix=t_mix, t_val=t_val)
            e_eff = h_eff.ground_state_energy(k)
            e_exact = h_exact.ground_state_energy(k)

            x.append(k)
            if hw != 0:
                y.append((e_eff - e_exact) / hw)
            else:
                y.append((e_eff - e_exact))
        plots.append((x, y, const_list, const_dict))

    if not use_latex:
        title = ('Ground state toy model energies calculated for '
                 'different A prescriptions;'
                 '\nv_0={}, hw={}, T={}'
                 ''.format(v0, hw, (t_core, t_mix, t_val)))
        xlabel = 'Number of particles, A'
        ylabel = '(E_valence - E_exact) / hw'
    else:
        title = ('Ground state toy model energies calculated for '
                 'different A prescriptions;'
                 '\n$v_0={}, \\hbar\\omega={}, T={}$'
                 ''.format(v0, hw, (t_core, t_mix, t_val)))
        xlabel = 'Number of particles, $A$'
        ylabel = ('$(E_{\\mathrm{valence}} - E_{\\mathrm{exact}}) / '
                  '\\hbar\\omega$')

    return plot_the_plots(
        plots,
        sort_key=lambda plot: plot[3]['presc'],
        sort_reverse=True,
        label='{a}',
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        get_label_kwargs=lambda plot, i: {'a': plot[3]['presc']},
        include_legend=True,
        cmap='jet',
        dark=False)


def permutations_with_replacement(iterable, r):
    combos = combinations_with_replacement(iterable, r)
    perms = list()
    for c in combos:
        perms.extend(permutations(c, r))
    return set(perms)

for t1, t2, t3 in sorted(permutations_with_replacement(range(2), 3)):
    plot_a_prescriptions(t_core=t1, t_mix=t2, t_val=t3)
    plt.show()
