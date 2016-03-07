#!/usr/bin/python
from __future__ import division
from os import getcwd, path, walk, mkdir, chdir, symlink, remove
from subprocess import call
from FGetSmallerInteraction import run as truncate_interaction
from FdoVCE import run as vce_calculation
import re

# CONSTANTS
Z_NAME_MAP = {
    1: 'h_', 2: 'he', 3: 'li', 4: 'be', 5: 'b_', 6: 'c_', 7: 'n_', 8: 'o_',
    9: 'f_', 10: 'ne', 11: 'na', 12: 'mg', 13: 'al', 14: 'si', 15: 'p_',
    16: 's_', 17: 'cl', 18: 'ar', 19: 'k_', 20: 'ca'
}
ZNAME_FMT_ALT = '%d-'
PATH_MAIN = getcwd()
PATH_TEMPLATES = path.join(PATH_MAIN, 'templates')
PATH_RESULTS = path.join(PATH_MAIN, 'results')
DIR_FMT_NUC = '%s%d_%d'  # name, A, Aeff
DIR_FMT_VCE = 'vce_presc%s'  # A prescription
FNAME_FMT_VCE = 'Aeff%d.int'  # Aeff
FNAME_FMT_NCSD_OUT = '%s%d_%d.out'  # name, A, Aeff
FNAME_MFDP = 'mfdp.dat'
FNAME_TRDENS_IN = 'trdens.in'
FNAME_EGV = 'mfdp.egv'
FNAME_TRDENS_OUT = 'trdens.out'
FNAME_HEFF = 'Heff_OLS.dat'
REGEX_TBME = 'TBME'
REGEX_EGV = 'mfdp_*\d+\.egv'
N_SHELL = 1
NHW = 6
N1 = 15
N2 = 6


# FUNCTIONS
def generating_a_values(nshell):
    """Based on the given major harmonic oscillator shell, gets the 3
    A values that are used to generate the effective Hamiltonian
    :param nshell: major oscillator shell
    """
    a0 = int((nshell+2)*(nshell+1)*nshell/3 * 2)
    return a0, a0+1, a0+2


def get_name(z, z_name_map=Z_NAME_MAP, alt_name=ZNAME_FMT_ALT):
    """Given the proton number, return the short element name
    :param z: number of protons
    :param z_name_map: map from proton number to abbreviated name
    :param alt_name: alternate name format if z not in z_name_map
    """
    if z in z_name_map:
        return z_name_map[z]
    else:
        return alt_name % z


def make_base_directories(a_values, a_prescription, results_path, dir_nuc):
    """Makes directories for first 3 a values if they do not exist yet
    :param a_values: Values of A for which directories are made.
    Example: If in nshell1, would make directories for He4,5,6
    :param a_prescription: Aeff prescription for VCE expansion
    :param results_path: Path to the directory into which these base
    directories are put
    :param dir_nuc: The directory name template which accepts one string and
    two integers as format parameters. (Name, A, Aeff)
    """
    if not path.exists(results_path):
        call(['ls', '~/NCSM'])
        mkdir(results_path)
    z = a_values[0]/2
    for a, aeff in zip(a_values, a_prescription):
        dirname = dir_nuc % (str(get_name(z=z)), int(a), int(aeff))
        dirpath = path.join(results_path, dirname)
        if not path.exists(dirpath):
            mkdir(dirpath)


def make_mfdp_file(z, a, aeff, nhw, n1, n2, path_elt,
                   outfile_name=FNAME_FMT_NCSD_OUT,
                   path_temp=PATH_TEMPLATES,
                   mfdp_name=FNAME_MFDP):
    """Reads the mfdp file from path_temp 
    and rewrites it into path_elt in accordance
    ith the given z, a, aeff, nhw, n1, n2, and outfile name
    :param z: Proton number
    :param a: Mass number
    :param aeff: Effective mass number for interaction
    :param nhw: Something something something dark side
    :param n1: Something something something dark side
    :param n2: Something something something dark side
    :param path_elt: The path to the directory into which the mfdp file is
    to be put
    :param outfile_name: The name of the written mfdp file
    :param path_temp: The path to the template directory
    :param mfdp_name: The name of the mfdp file
    """
    temp_mfdp_path = path.join(path_temp, mfdp_name)
    mfdp_path = path.join(path_elt, mfdp_name)
    replace_map = get_mfdp_replace_map(outfile_name=outfile_name, z=z, a=a, 
                                       nhw=nhw, n1=n1, n2=n2, aeff=aeff)
    _rewrite_file(src=temp_mfdp_path, dst=mfdp_path,
                  replace_map=replace_map)


def make_mfdp_files(z, a_range, a_presc, nhw, n1, n2,
                    results_path=PATH_RESULTS,
                    nuc_dir=DIR_FMT_NUC,
                    outfile_name=FNAME_FMT_NCSD_OUT,
                    path_temp=PATH_TEMPLATES,
                    mfdp_name=FNAME_MFDP):
    for a, aeff in zip(a_range, a_presc):
        dirname = nuc_dir % (get_name(z), a, aeff)
        dirpath = path.join(results_path, dirname)
        make_mfdp_file(z=z, a=a, aeff=aeff, nhw=nhw, n1=n1, n2=n2,
                       path_elt=dirpath, outfile_name=outfile_name,
                       path_temp=path_temp, mfdp_name=mfdp_name)


def get_mfdp_replace_map(outfile_name, z, a, nhw, n1, n2, aeff):
    outfile_name = outfile_name % (get_name(z), a, aeff)
    n = a - z
    if a % 2 == 0:
        tot2 = 0
    else:
        tot2 = 1
    return {'<<OUTFILE>>': str(outfile_name),
            '<<Z>>': str(z), '<<N>>': str(n),
            '<<NHW>>': str(nhw), '<<TOT2>>': str(tot2),
            '<<N1>>': str(n1), '<<N2>>': str(n2),
            '<<AEFF>>': str(aeff)}


def make_trdens_file(z, a, aeff,
                     nuc_dir=DIR_FMT_NUC,
                     path_results=PATH_RESULTS,
                     path_temp=PATH_TEMPLATES,
                     trdens_name=FNAME_TRDENS_IN):
    """Reads the trdens.in file from path_temp and rewrites it 
    into path_elt in accordance with the given z, a
    :param z: The proton number
    :param a: The mass number
    :param aeff: The effective mass number (to specify the interaction)
    :param nuc_dir: The directory name template
    :param path_results: The path to the results directory
    :param path_temp: The path to the templates directory
    :param trdens_name: The name of the trdens file in the templates dir
    """
    src = path.join(path_temp, trdens_name)
    nuc_dir = nuc_dir % (get_name(z), a, aeff)
    path_elt = path.join(path_results, nuc_dir)
    dst = path.join(path_elt, trdens_name)
    rep_map = get_trdens_replace_map(z=z, a=a)
    _rewrite_file(src=src, dst=dst, replace_map=rep_map)
    

def get_trdens_replace_map(z, a):
    nnn, num_states = get_num_states(z, a)
    return {'<<NNN>>': str(nnn), '<<NUMSTATES>>': str(num_states)}


def get_num_states(z, a):
    if z == 2:
        if a == 5:
            return 1, 2
        elif a == 6:
            return 2, 5
        else:
            raise UnknownNumStatesException()
    else:
        raise UnknownNumStatesException()


class UnknownNumStatesException(Exception):
    pass


def _rewrite_file(src, dst, replace_map):
    """Reads the file given by src, replaces string elements based
       on the replace map, writes the file into dst.
    """
    # read the src file
    read_lines = list()
    infile = open(src, 'r')
    for line in infile:
        read_lines.append(line)
    infile.close()
    # replace strings
    write_lines = list()
    for line in read_lines:
        for k, v in replace_map.iteritems():
            if k in line:
                line = line.replace(k, str(v))
        write_lines.append(line)
    # write to the dst file
    outfile = open(dst, 'w')
    outfile.writelines(write_lines)
    outfile.close()

    
def truncate_space(n1, n2,
                   path_elt,
                   path_temp=PATH_TEMPLATES,
                   tbme_name_regex=REGEX_TBME):
    """Run the script that truncates the space by removing extraneous
    interactions from the TBME file

    :param n1: something
    :param n2: something
    :param path_elt: Path to the directory in which the resultant TBME file
    is to be put
    :param path_temp: Path to the templates directory in which the full TBME
    files resides
    :param tbme_name_regex: Regular expression that matches only the TBME file
    in the templates directory
    """
    w = walk(path_temp)
    dirpath, dirnames, filenames = w.next()
    for f in filenames:
        if re.match(tbme_name_regex, f) is not None:
            tbme_filename = f
            break
    else:
        raise TBMEFileNotFoundException()
    src_path = path.join(dirpath, tbme_filename)
    dst_path = path.join(path_elt, tbme_filename)
    truncate_interaction(src_path, n1, n2, dst_path)


def truncate_spaces(n1, n2,
                    dirpaths, path_temp,
                    tbme_name_regex):
    """For multiple directories, perform the operation of truncate_space

    :param n1: Something
    :param n2: Something
    :param dirpaths: Paths to the destinations
    :param path_temp: Path to the source
    :param tbme_name_regex: Regex that matches the TBME file in the source
    """
    for dirpath in dirpaths:
        truncate_space(n1=n1, n2=n2, path_elt=dirpath,
                       path_temp=path_temp,
                       tbme_name_regex=tbme_name_regex)


class TBMEFileNotFoundException(Exception):
    pass


def rename_egv_file(a6_dir, egv_name_regex, next_egv_name, force):
    """Renames the egv file from its default output name to the name needed
    for running TRDENS

    :param a6_dir: Directory in which the file resides
    :param egv_name_regex: Regular expression that matches the defualt output
    name
    :param next_egv_name: Name that the file is renamed to
    :param force: If True, replaces any existing files by the name of
    next_egv_name
    """
    next_egv_path = path.join(a6_dir, next_egv_name)
    if path.lexists(next_egv_path): 
        if not force:
            return 0
        else:
            remove(next_egv_path)
    dirpath, dirnames, filenames = walk(a6_dir).next()
    for f in filenames:
        if re.match(egv_name_regex, f) is not None:
            symlink(f, next_egv_path)
            break
    else:
        raise EgvFileNotFoundException()
    return 1
    

class EgvFileNotFoundException(Exception):
    pass


def do_ncsd(a_values, a_prescription, a_dirpaths_map, a_outfile_map, force):
    """Run the NCSD calculations. For each A value, does the NCSD calculation
    in each of its corresponding directories.

    :param a_values: 3-tuple of A values which generate the effective
    Hamiltonian
    :param a_prescription: 3-tuple of Aeff values which, along with the
    A values, are used to generate the effective Hamiltonian
    :param a_dirpaths_map: Map from A values to the path to the directory in
    which NCSD is to be performed
    :param a_outfile_map: Map from A values to the .out files produced by the
    calculation. If force is False, will not do the calculation if such files
    already exist
    :param force: If true, redoes the calculations even if the output files
    already exist.
    """
    main_dir = getcwd()
    for a, aeff in zip(a_values, a_prescription):
        if force or not path.exists(path.join(a_dirpaths_map[a], 
                                              a_outfile_map[a])):
            chdir(a_dirpaths_map[a])
            call(['NCSD'])
            chdir(main_dir)


def do_trdens(a6_dir, force, outfile):
    """Run the TRDENS calculation in a6_dir

    :param a6_dir: Directory in which to run the calulation
    :param force: If True, redoes the calculation even if output files already
    exist
    :param outfile: Name of the output file generated by the TRDENS calculation.
    (If force is False, will not run if outfile already exists)
    """
    outfile_path = path.join(a6_dir, outfile)
    if path.exists(outfile_path):
        if not force:
            return 0
        else:
            remove(outfile_path)
    main_dir = getcwd()
    chdir(a6_dir)
    call(['TRDENS'])
    chdir(main_dir)
    return 1


def do_vce(a_outfile_map, a_values, a6_dir, heff_fname, aeff_range,
           vce_int_fname, vce_dirpath, force):
    """Do the VCE expansion calculation for each Aeff value in aeff_range

    :param a_outfile_map: Map from A values to their respective NCSD output
    files
    :param a_values: A values used to form the effective Hamiltonian
    :param a6_dir: Directory for the 3rd A value
    :param heff_fname: Name of the effective Hamiltonian output file
    generated by the TRDENS calculation for the 3rd A value
    :param aeff_range: Range of Aeff values to evaluate based on the effective
    Hamiltonian
    :param vce_int_fname: Filename template for generated interaction files
    :param vce_dirpath: Path to the directory in which to put generated
    interaction files
    :param force: If True, force redoing the calculation even if output files
    already exist
    """
    he4_fname = a_outfile_map[a_values[0]]
    he5_fname = a_outfile_map[a_values[1]]
    he6_fname = path.join(a6_dir, heff_fname)
    for aeff in aeff_range:
        fname = vce_int_fname % aeff
        fpath = path.join(vce_dirpath, fname)
        if force or not path.exists(fpath):
            vce_calculation(aeff, fpath, he4_fname, he5_fname, he6_fname)


def ncsd_vce_calculation(a_prescription, aeff_range=list(),
                         nshell=N_SHELL,
                         nhw=NHW, n1=N1, n2=N2,
                         path_results=PATH_RESULTS,
                         path_temp=PATH_TEMPLATES,
                         dir_fmt_nuc=DIR_FMT_NUC,
                         dir_fmt_vce=DIR_FMT_VCE,
                         fname_regex_tbme=REGEX_TBME,
                         fname_regex_egv=REGEX_EGV,
                         fname_fmt_ncsd_out=FNAME_FMT_NCSD_OUT,
                         fname_fmt_vce=FNAME_FMT_VCE,
                         fname_mfdp=FNAME_MFDP,
                         fname_trdens_in=FNAME_TRDENS_IN,
                         fname_trdens_out=FNAME_TRDENS_OUT,
                         fname_egv_final=FNAME_EGV,
                         fname_heff=FNAME_HEFF,
                         force_ncsd=False,
                         force_trdens=False,
                         force_vce=False,
                         force_all=False):
    """Valence cluster expansion calculations within NCSM

    :param a_prescription: 3-tuple containing the Aeff values for use in
    constructing the effective interaction Hamiltonian
    :param aeff_range: List of Aeff values for which to evaluate based on the
    effective Hamiltonian
    :param nshell: Major harmonic oscillator shell
    :param nhw: Something something something dark side
    :param n1: Something something something dark side
    :param n2: Something something something dark side
    :param path_results: Path to the results directory
    :param path_temp: Path to the templates directory
    :param dir_fmt_nuc: Template string for the directory for the base
    interactions. The format parameters are a string and two integers,
    representing (name, A, Aeff). For example, for Nshell1 with a (6, 6, 6)
    prescription, one would create he4_6, he5_6, he6_6.
    :param dir_fmt_vce: Template string for the directory for the interactions
    calculated based on the effective Hamiltonian. Should accept a string
    representing the A_prescription tuple as a format argument.
    :param fname_regex_tbme: Regular expression that matches the TBME file in
    the templates directory
    :param fname_regex_egv: Regular expression that matches the .egv file
    outputted by the NCSD calculation (which needs to be renamed)
    :param fname_fmt_ncsd_out: Template string for the .out file to be
    outputted by the NCSD calculation. (This should accept the same arguments
    as the dir_fmt_nuc, so a natural choice for this string would be appending
    '.out' to the dir_fmt_nuc).
    :param fname_fmt_vce: Template string for the name to be given to the
    .int file generated by running the FdoVCE script. Should accept a single
    integer as an argument (Aeff).
    :param fname_mfdp: Name of the mfdp file in the templates directory
    :param fname_trdens_in: Name of the trdens input file in the templates
    directory
    :param fname_egv_final: String with which to rename the .egv file outputted
    by the NCSD calculation in prepration for the TRDENS calculation.
    :param fname_trdens_out: Name of the trdens output file (signifying that
    this operation has been performed).
    :param fname_heff: Name of the effective OLS Hamiltonian file
    :param force_ncsd: If True, force redoing the NCSD calculations even if
    output files already exist
    :param force_trdens: If True, force redoing the TRDENS calculations even
    if output files already exist
    :param force_vce: If True, force redoing the VCE expansion calculations
    even if output files already exist
    :param force_all: If True, force redoing all calculations
    """
    # Get a values and directory names
    a_values = generating_a_values(nshell)
    z = a_values[0] / 2
    a_dirpaths_map = dict()
    for a, aeff in zip(a_values, a_prescription):
        a_dirpaths_map[a] = path.join(path_results,
                                      dir_fmt_nuc % (get_name(z), a, aeff))
    a_outfile_map = dict()
    for a, aeff in zip(a_values, a_prescription):
        a_outfile_map[a] = path.join(
            a_dirpaths_map[a], fname_fmt_ncsd_out % (get_name(z), a, aeff))

    # Make directories for base files
    make_base_directories(a_values=a_values,
                          a_prescription=a_prescription,
                          results_path=path_results,
                          dir_nuc=dir_fmt_nuc)

    # ncsd calculations: make mfdp files, perform truncation, and run NCSD
    make_mfdp_files(z=z, a_range=a_values, a_presc=a_prescription,
                    nhw=nhw, n1=n1, n2=n2,
                    outfile_name=fname_fmt_ncsd_out,
                    mfdp_name=fname_mfdp)
    truncate_spaces(n1=n1, n2=n2, dirpaths=a_dirpaths_map.values(),
                    path_temp=path_temp,
                    tbme_name_regex=fname_regex_tbme)
    do_ncsd(a_values=a_values, a_prescription=a_prescription,
            a_dirpaths_map=a_dirpaths_map, a_outfile_map=a_outfile_map,
            force=force_ncsd or force_all)

    # for the 3rd a value, make trdens file and run TRDENS
    make_trdens_file(z=z, a=a_values[2], aeff=a_prescription[2],
                     nuc_dir=dir_fmt_nuc,
                     path_results=path_results,
                     path_temp=path_temp,
                     trdens_name=fname_trdens_in)
    a6_dir = a_dirpaths_map[a_values[2]]
    rename_egv_file(a6_dir=a6_dir, egv_name_regex=fname_regex_egv,
                    next_egv_name=fname_egv_final,
                    force=force_trdens or force_all)
    do_trdens(a6_dir=a6_dir, force=force_trdens or force_all,
              outfile=fname_trdens_out)

    # do valence cluster expansion
    if len(aeff_range) == 0:
        return 1
    vce_dirpath = path.join(path_results, dir_fmt_vce % str(a_prescription))
    if not path.exists(vce_dirpath):
        mkdir(vce_dirpath)
    do_vce(a_outfile_map=a_outfile_map, a_values=a_values, a6_dir=a6_dir,
           heff_fname=fname_heff, aeff_range=aeff_range,
           vce_int_fname=fname_fmt_vce, vce_dirpath=vce_dirpath,
           force=force_vce or force_all)

    return 1


# SCRIPT
ncsd_vce_calculation(a_prescription=(4, 5, 6), aeff_range=range(4, 17), nhw=4)
