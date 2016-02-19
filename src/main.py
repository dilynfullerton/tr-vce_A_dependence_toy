from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sys import path
from itertools import combinations_with_replacement, permutations

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq

from toy import HamiltonianToy as H_ex
from toy import HamiltonianToyEffective as H_eff
from toy import get_a_exact as exact
from toy import custom_a_prescription as custom

path.extend([b'../../imsrg_mass_plots/src'])
# noinspection PyPep8,PyUnresolvedReferences
from plotting import plot_the_plots
from metafit import _meta_fit

A_PRESCRIPTIONS = [exact,
                   custom(4, 5, 6),
                   custom(6, 6, 6),
                   custom(6, 7, 8),
                   custom(7, 7, 8),
                   custom(7, 7, 7),
                   custom(6.79, 7.14, 7.54)
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
T_CORE = 0
T_MIX = -1
T_VAL = 0
LATEX = False


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
            err = _get_e_eff_error(
                ap=ap, k=k, valence_space=valence_space,
                v0=v0, hw=hw, t_core=t_core, t_mix=t_mix, t_val=t_val)
            x.append(k)
            if hw != 0:
                y.append(err / hw)
            else:
                y.append(err)
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


def e_eff_error_array(a_prescription, k_range, valence_space, v0, hw,
                      t_core, t_mix, t_val):
    x_array = np.array(k_range)
    y_array = np.empty(shape=x_array.shape)
    for x, i in zip(x_array, range(len(x_array))):
        ap = custom(*a_prescription)
        y_array[i] = _get_e_eff_error(
            ap=ap, k=x, valence_space=valence_space,
            v0=v0, hw=hw, t_core=t_core, t_mix=t_mix, t_val=t_val)
    return y_array


def e_eff_error_arrays(a_prescription, params_range):
    list_of_y_array = list()
    for params in params_range:
        list_of_y_array.append(e_eff_error_array(a_prescription, *params))
    return np.concatenate(tuple(list_of_y_array))


def _get_e_eff_error(ap, k, valence_space, v0, hw, t_core, t_mix, t_val):
    a2, a3, a4 = ap(k)[:3]

    h2 = H_ex(a=a2, v0=v0, hw=hw, valence_space=valence_space,
              t_core=t_core, t_mix=t_mix, t_val=t_val)
    h3 = H_ex(a=a3, v0=v0, hw=hw, valence_space=valence_space,
              t_core=t_core, t_mix=t_mix, t_val=t_val)
    h4 = H_ex(a=a4, v0=v0, hw=hw, valence_space=valence_space,
              t_core=t_core, t_mix=t_mix, t_val=t_val)

    e2 = h2.ground_state_energy(k=valence_space[0] - 1)
    e3 = h3.ground_state_energy(k=valence_space[0])
    e4 = h4.ground_state_energy(k=valence_space[0] + 1)

    e_core = e2
    e_p = e3 - e2
    v_eff = e4 - 2 * e3 + e2
    h_eff = H_eff(e_core=e_core, e_p=e_p, v_eff=v_eff,
                  valence_space=valence_space)
    h_exact = H_ex(a=k, v0=v0, hw=hw, valence_space=valence_space,
                   t_core=t_core, t_mix=t_mix, t_val=t_val)
    e_eff = h_eff.ground_state_energy(k)
    e_exact = h_exact.ground_state_energy(k)

    return e_eff - e_exact


def permutations_with_replacement(iterable, r):
    combos = combinations_with_replacement(iterable, r)
    perms = list()
    for c in combos:
        perms.extend(permutations(c, r))
    return set(perms)


# for t1, t2, t3 in reversed(sorted(permutations_with_replacement(range(3), 3))):
#     plot_a_prescriptions(t_core=t1, t_mix=t2, t_val=t3)
# plot_a_prescriptions(t_core=T_CORE, t_mix=T_MIX, t_val=T_VAL)
for tt2 in np.linspace(-5.0, 0.0, 11):
    plot_a_prescriptions(t_mix=tt2)
plt.show()

# params_range = list()
# for t2 in np.linspace(-5.0, 0.0, 11):
#     params_range.append((K_RANGE, VALENCE_SPACE, V0, HW, T_CORE, t2, T_VAL))
# res = leastsq(func=e_eff_error_arrays, x0=np.array([1.0, 1.0, 1.0]),
#               args=(params_range,),
#               full_output=False)
# print(res)
