from __future__ import unicode_literals

from sys import path
import os
from itertools import combinations_with_replacement, permutations

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq

from toy import HamiltonianToy as H_ex
from toy import HamiltonianToyEffective as H_eff
from toy import get_a_exact as exact
from toy import custom_a_prescription as custom
from toy import get_n_i_fn as get_ni_fn

path.extend([b'../../tr-a_dependence_plots/src'])
# noinspection PyPep8,PyUnresolvedReferences
from plotting import save_plot_figure
from plotting import save_plot_data_file
# noinspection PyPep8,PyUnresolvedReferences
from constants import LEGEND_SIZE

A_PRESCRIPTIONS = [
    exact,
    # custom(16, 17, 18),
    # custom(18, 18, 18),
    # custom(19.25066421,  19.11085927,  17.88211487),   # T2 = -1
    custom(4, 5, 6),
    custom(6, 6, 6),
    # custom(6.81052543, 7.16905406, 7.56813846),         # T2 = [-5,0]
    # custom(1., 7.11366579, 7.51313418),                 # T2 = [-5,0], no1b
    # custom(4.17371637, 4.94074871, 5.80477268),         # T2 = -0.5
    # custom(4.57072129, 4.75300152, 4.85662704),         # T2 = -1
    # custom(5.23102228, 4.60448702, 1.41439515*10**06),  # T2 = -1.5
    # custom(5.64453846, 2.39084813, 8.94982568),         # T2 = -2
    # custom(6.80793461, 7.1641873, 7.56355624),          # T2 = [-10,0]
    # custom(6.79278718, 7.1414551, 7.54129819),          # T2 = [-20,0]
    ]
N_SHELL = 1
N_COMPONENT = 2
K0 = int((N_SHELL+2) * (N_SHELL+1) * N_SHELL/3 * N_COMPONENT)
KMAX = int((N_SHELL+3) * (N_SHELL+2) * (N_SHELL+1)/3 * N_COMPONENT)
K_RANGE = range(K0, KMAX+1)
VALENCE_SPACE = range(K0+1, KMAX+1)
GET_NI_FN = get_ni_fn
INCLUDE_1BODY = False
INCLUDE_2BODY = True
HW = 1.0
V0 = 1.0
T_2 = 0.0
T_CC = 0.0
T_CV = -1
T_VV = 0.0
# SAVEDIR = b'../../3679406mjsmbs/dilyn/figures'
SAVEDIR = b'../plots'
DATA_FILE_SAVEDIR = b'../plots'
LEGEND_SIZE.space_scale = 6


def permutations_with_replacement(iterable, r):
    combos = combinations_with_replacement(iterable, r)
    perms = list()
    for c in combos:
        perms.extend(permutations(c, r))
    return set(perms)


def _get_e_eff_error(
        ap, k, valence_space, n_component, v0, hw, t_cc, t_cv, t_vv,
        get_n_i_fn, incl_1body, incl_2body
):
    """For a given A-prescription (ap) and mass number (k), gets the error
    between the ground state energy from the effective Hamiltonian
    (from the A-prescription) and the exact ground state energy from
    the toy model Hamiltonian
    :param ap: function A -> (A4, A5, A6, name) that when given a mass number
    returns a tuple of the three effective A's from which to generate the
    effective Hamiltonian and a name for the prescription.
    :param k: mass number. I called this 'k' to separate it from the
    effective A. This is consistent with my analytical derivations.
    :param valence_space: list of mass numbers that are considered to be the
    valence space
    :param n_component: 1 -> neutrons, 2 -> protons & neutrons
    :param v0: v0 term in toy model hamiltonian
    :param hw: h-bar omega for the oscillator
    :param t_cc: core-core t term for toy model Hamiltonian
    :param t_cv: core-valence t term for toy model Hamiltonian
    :param t_vv: valence-valence t term for toy model Hamiltonian
    :param get_n_i_fn: function that when given ncomponent, returns the appropriate
    function from i -> Ni. Thus this function has the form
        ni_fn: ncomponent -> (ni: i -> n)
    :param incl_1body: if true, include 1-body term
    :param incl_2body: if true, include 2-body term
    """
    a2, a3, a4 = ap(k)[:3]
    h2 = H_ex(
        a=a2, v0=v0, hw=hw, n_i=get_n_i_fn, valence_space=valence_space,
        n_component=n_component, t_cc=t_cc, t_cv=t_cv, t_vv=t_vv,
        include1body=incl_1body, include2body=incl_2body
    )
    h3 = H_ex(
        a=a3, v0=v0, hw=hw, n_i=get_n_i_fn, valence_space=valence_space,
        n_component=n_component, t_cc=t_cc, t_cv=t_cv, t_vv=t_vv,
        include1body=incl_1body, include2body=incl_2body
    )
    h4 = H_ex(
        a=a4, v0=v0, hw=hw, n_i=get_n_i_fn, valence_space=valence_space,
        n_component=n_component, t_cc=t_cc, t_cv=t_cv, t_vv=t_vv,
        include1body=incl_1body, include2body=incl_2body
    )
    e2 = h2.ground_state_energy(k=valence_space[0] - 1)
    e3 = h3.ground_state_energy(k=valence_space[0])
    e4 = h4.ground_state_energy(k=valence_space[0] + 1)
    e_core = e2
    e_p = e3 - e2
    v_eff = e4 - 2 * e3 + e2
    h_eff = H_eff(
        e_core=e_core, e_p=e_p, v_eff=v_eff, valence_space=valence_space)
    h_exact = H_ex(
        a=k, v0=v0, hw=hw, n_i=get_n_i_fn, valence_space=valence_space,
        n_component=n_component, t_cc=t_cc, t_cv=t_cv, t_vv=t_vv,
        include1body=incl_1body, include2body=incl_2body
    )
    e_eff = h_eff.ground_state_energy(k)
    e_exact = h_exact.ground_state_energy(k)
    return e_eff - e_exact


def plot_a_prescriptions(
        a_prescriptions=A_PRESCRIPTIONS, k_range=K_RANGE,
        nshell=N_SHELL, n_component=N_COMPONENT,
        valence_space=VALENCE_SPACE, v0=V0, hw=HW,
        t_tuples=list([(T_CC, T_CV, T_VV)]), get_n_i_fn=GET_NI_FN,
        incl_1body=INCLUDE_1BODY, incl_2body=INCLUDE_2BODY,
        savedir=SAVEDIR, datafile_savedir=DATA_FILE_SAVEDIR,
        savename_fmt='fig_{presc}_{ttup}_{nsh}_{v0}_{hw}-test',
        title='Toy model ground state energy error',
        legend_size=LEGEND_SIZE, label_fmt='{t:9}, {a}',
):
    """For each A-prescription, plot the difference between the energy based on
    the effective Hamiltonian generated from the A prescription and the energy
    based on the exact Hamiltonian.
    Note: k represents the actual mass number (i.e. the number of ones in the
    state vector), while a represents that used in the Hamiltonian
    :param a_prescriptions: List of functions that take an actual mass
    number and produce the first three Aeff values from it
    :param k_range: Range of ACTUAL mass numbers for which to evaluate the
    energy difference.
    :param nshell: Major harmonic oscillator shell
    :param n_component: Number of components of the system
    (2 ==> protons + neutrons)
    :param valence_space: Range of values over which to evaluate the
    effective Hamiltonian
    :param v0: v0
    :param hw: hw
    :param t_tuples: List of 3-tuples specifying the range of core-core,
    core-valence, and valence-valence interaction terms to address
    :param get_n_i_fn: Function that gives the n_i value
    :param incl_1body: Whether to include 1-body term
    :param incl_2body: Whether to include 2-body terms
    :param savedir: Directory in which the plots are to be saved
    :param datafile_savedir: Directory in which to write a data file
    representing the plot. (If None, will not write file)
    :param savename_fmt: Template filename with which to save the plots
    :param title: Title of the plot
    :param legend_size: Legend sizing object, based on the definition in
    imsrg_mass_plots project
    :param label_fmt: Label string template
    """
    plots = list()
    for ap in a_prescriptions:
        for t_cc, t_cv, t_vv in t_tuples:
            x = list()
            y = list()
            const_list = list()
            const_dict = {'presc': ap(0)[3], 't_cv': t_cv}
            for k in k_range:
                err = _get_e_eff_error(
                    ap=ap, k=k, v0=v0, hw=hw, get_n_i_fn=get_n_i_fn,
                    valence_space=valence_space, n_component=n_component,
                    t_cc=t_cc, t_cv=t_cv, t_vv=t_vv,
                    incl_1body=incl_1body, incl_2body=incl_2body
                )
                x.append(k)
                if hw != 0:
                    y.append(err / hw)
                else:
                    y.append(err)
            plots.append((x, y, const_list, const_dict))
    xlabel = 'Number of particles, A'
    ylabel = '(E_valence - E_exact) / hw'
    # Make plot labels
    data_labels = list()
    for p in plots:
        fmt_dict = {'t': 'T_cv = ' + str(p[3]['t_cv']), 'a': p[3]['presc']}
        data_labels.append(label_fmt.format(**fmt_dict))
    ap_names = [str(ap(0)[3].split('=')[1].strip())
                for ap in a_prescriptions]
    savename = savename_fmt.format(
        presc=ap_names, ttup=t_tuples, nsh=nshell, v0=v0, hw=hw)
    save_plot_figure(
        data_plots=plots, title=title,
        xlabel=xlabel, ylabel=ylabel, data_labels=data_labels,
        savepath=os.path.join(savedir, savename + '.pdf'),
        cmap_name='jet',
        legendsize=legend_size,
    )
    save_plot_data_file(
        plots=plots, title=title,
        xlabel=xlabel, ylabel=ylabel, labels=data_labels,
        savepath=os.path.join(datafile_savedir, savename + '.dat'),
    )
    # get_label_kwargs = lambda p, i: {
    #     't': 'T_cv = ' + str(p[3]['t_cv']), 'a': p[3]['presc']},
    # return plot_the_plots(
    #     plots,
    #     sort_key=lambda plot: plot[3]['presc'],
    #     sort_reverse=True,
    #     label=label_fmt,
    #     title=title,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     get_label_kwargs=get_label_kwargs,
    #     include_legend=True,
    #     legend_size=legend_size,
    #     cmap='jet',
    #     savedir=savedir,
    #     savename=savename,
    #     dark=False,
    #     extension='.pdf',
    #     data_file_savedir=datafile_savedir,
    # )


def e_eff_error_array(
        a_prescription, k_range, valence_space, n_component,
        v0, hw, t_cc, t_cv, t_vv, get_n_i_fn, incl_1body, incl_2body
):
    """Gets the error array for the given A-prescription and mass number
    range based on _get_e_eff_error.
    See docstring for _get_e_eff_error for explanation of other parameters
    """
    x_array = np.array(k_range)
    y_array = np.empty(shape=x_array.shape)
    for x, i in zip(x_array, range(len(x_array))):
        ap = custom(*a_prescription)
        y_array[i] = _get_e_eff_error(
            ap=ap, k=x, valence_space=valence_space, n_component=n_component,
            v0=v0, hw=hw, t_cc=t_cc, t_cv=t_cv, t_vv=t_vv, get_n_i_fn=get_n_i_fn,
            incl_1body=incl_1body, incl_2body=incl_2body
        )
    return y_array


def e_eff_error_arrays(a_prescription, params_list):
    """For each set of params in params_list (a list of params tuples),
    gets the error array based on e_eff_error_array and concatenates all of
    these into one for simultaneous optimization
    """
    list_of_y_array = list()
    for params in params_list:
        list_of_y_array.append(e_eff_error_array(a_prescription, *params))
    return np.concatenate(tuple(list_of_y_array))


def optimize_prescription_for_t_cv_range(
        t_cv_range,
        k_range=K_RANGE, valence_space=VALENCE_SPACE, n_component=N_COMPONENT,
        v0=V0, hw=HW, t_cc=T_CC, t_vv=T_VV, get_n_i_fn=GET_NI_FN,
        incl_1body=INCLUDE_1BODY, incl_2body=INCLUDE_2BODY,
):
    """Performs a simultaneous least-squares minimization of the error arrays
    for all of the t_cv terms in t_cv_range. Providing a larger range
    of t_cv values will ensure a more robust and universal optimization.
    See docstring for _get_e_eff_error for explanation of other parameters
    """
    params_range = list()
    for t_cv in t_cv_range:
        params_range.append(
            (k_range, valence_space, n_component,
             v0, hw, t_cc, t_cv, t_vv, get_n_i_fn, incl_1body, incl_2body))
    return leastsq(
        func=e_eff_error_arrays, x0=np.array([1, 1, 1]),
        args=(params_range,), full_output=False)

# SCRIPT

# plot_a_prescriptions(
#     t_tuples=[(T_CC, t, T_VV) for t in range(-1, 2)])

plot_a_prescriptions()

# for tt2 in np.linspace(-1.0, 0.0, 21):
#     plot_a_prescriptions(t_cv=tt2)
plt.show()
#
# t_cv_range = np.linspace(-5.0, 0.0, 11)
# t_cv_range = [0.001]
res = optimize_prescription_for_t_cv_range(t_cv_range=t_cv_range)
# print(res)
