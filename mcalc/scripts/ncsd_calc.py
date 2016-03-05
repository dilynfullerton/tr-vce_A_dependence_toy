#!/usr/bin/python
from __future__ import division
from os import getcwd, path, walk, mkdir, link, chdir, symlink, remove, unlink
from shutil import copyfile
from subprocess import call
from FGetSmallerInteraction import run as truncate_interaction
from FdoVCE import run as vce_calculation

import re
import glob

# CONSTANTS
Z_NAME_MAP = {
    1: 'h_', 2: 'he',
    3: 'li', 4: 'be', 5: 'b_', 6: 'c_', 7: 'n_', 8: 'o_', 9: 'f_', 10: 'ne',
    11: 'na', 12: 'mg', 13: 'al', 14: 'si', 15: 'p_', 16: 's_', 17: 'cl', 18: 'ar',
    19: 'k_', 20: 'ca'
}
ALT_NAME = '%d-'
PATH_MAIN = getcwd()
PATH_TEMPLATES = path.join(PATH_MAIN, 'templates')
PATH_RESULTS = path.join(PATH_MAIN,'results')
DIR_NUC = '%s%d_%d'  # name, A, Aeff
DIR_VCE = 'vce_presc%s'  # A prescription
FNAME_VCE = 'Aeff%d.int'  # Aeff
FNAME_OUTFILE = '%s%d_%d.out'  # name, A, Aeff
FNAME_MFDP = 'mfdp.dat'
FNAME_TRDENS = 'trdens.in'
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
    """
    a0 = int((nshell+2)*(nshell+1)*nshell/3 * 2)
    return a0, a0+1, a0+2



def get_name(z, z_name_map=Z_NAME_MAP, alt_name=ALT_NAME):
    """Given the proton number, return the short element name"""
    if z in z_name_map:
        return z_name_map[z]
    else:
        return alt_name % z


def make_base_directories(a_values, a_prescription, results_path, dir_nuc):
    """Makes directories for first 3 a values if they do not exist yet"""
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
                   outfile_name=FNAME_OUTFILE,
                   path_temp=PATH_TEMPLATES,
                   mfdp_name=FNAME_MFDP):
    """Reads the mfdp file from path_temp 
       and rewrites it into path_elt in accordance
       with the given z, a, aeff, nhw, n1, n2, and outfile name
    """
    temp_mfdp_path = path.join(path_temp, mfdp_name)
    mfdp_path = path.join(path_elt, mfdp_name)
    replace_map = get_mfdp_replace_map(outfile_name=outfile_name, z=z, a=a, 
                                       nhw=nhw, n1=n1, n2=n2, aeff=aeff)
    _rewrite_file(src=temp_mfdp_path, dst=mfdp_path,
                  replace_map=replace_map)


def make_mfdp_files(z, a_range, a_presc, nhw, n1, n2,
                    results_path=PATH_RESULTS,
                    nuc_dir=DIR_NUC,
                    outfile_name=FNAME_OUTFILE,
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
                     nuc_dir=DIR_NUC,
                     path_results=PATH_RESULTS,
                     path_temp=PATH_TEMPLATES,
                     trdens_name=FNAME_TRDENS):
    """Reads the trdens.in file from path_temp and rewrites it 
       into path_elt in accordance with the given z, a
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
    for dirpath in dirpaths:
        truncate_space(n1=n1, n2=n2, path_elt=dirpath,
                       path_temp=path_temp,
                       tbme_name_regex=tbme_name_regex)


class TBMEFileNotFoundException(Exception):
    pass


def rename_egv_file(a6_dir, egv_name_regex, next_egv_name, force):
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
    main_dir = getcwd()
    for a, aeff in zip(a_values, a_prescription):
        if force or not path.exists(path.join(a_dirpaths_map[a], 
                                              a_outfile_map[a])):
            chdir(a_dirpaths_map[a])
            call(['NCSD'])
            chdir(main_dir)


def do_trdens(a6_dir, force, outfile):
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


def do_vce(a_outfile_map, a_values, a6_dir, heff_fname, aeff_list, 
           vce_int_fname, vce_dirpath, force):
    he4_fname = a_outfile_map[a_values[0]]
    he5_fname = a_outfile_map[a_values[1]]
    he6_fname = path.join(a6_dir, heff_fname)
    for aeff in aeff_list:
        fname = vce_int_fname % aeff
        fpath = path.join(vce_dirpath, fname)
        if force or not path.exists(fpath):
            vce_calculation(aeff, fpath, he4_fname, he5_fname, he6_fname)


def ncsd_vce_calculations(a_prescription, aeff_list=list(), 
                          nshell=N_SHELL, 
                          results_path=PATH_RESULTS,
                          path_temp=PATH_TEMPLATES,
                          dir_nuc=DIR_NUC,
                          nhw=NHW, n1=N1, n2=N2,
                          tbme_name_regex=REGEX_TBME,
                          outfile_name=FNAME_OUTFILE,
                          mfdp_name=FNAME_MFDP,
                          trdens_name=FNAME_TRDENS,
                          egv_name_regex=REGEX_EGV,
                          next_egv_name=FNAME_EGV,
                          outfile_trdens=FNAME_TRDENS_OUT,
                          heff_fname=FNAME_HEFF,
                          dir_vce=DIR_VCE,
                          vce_int_fname=FNAME_VCE,
                          force_ncsd=False,
                          force_trdens=False,
                          force_vce=False,
                          force_all=False):
    # Get a values and directory names
    a_values = generating_a_values(nshell)
    z = a_values[0] / 2
    a_dirpaths_map = dict()
    for a, aeff in zip(a_values, a_prescription):
        a_dirpaths_map[a] = path.join(results_path, 
                                      dir_nuc % (get_name(z), a, aeff))
    a_outfile_map = dict()
    for a, aeff in zip(a_values, a_prescription):
        a_outfile_map[a] = path.join(a_dirpaths_map[a],
                                     outfile_name % (get_name(z), a, aeff))
    # Make directories for base files
    make_base_directories(a_values=a_values,
                          a_prescription=a_prescription,
                          results_path=results_path,
                          dir_nuc=dir_nuc)
    # ncsd calculations: make mfdp files, perform truncation, and run NCSD
    make_mfdp_files(z=z, a_range=a_values, a_presc=a_prescription,
                    nhw=nhw, n1=n1, n2=n2,
                    outfile_name=outfile_name,
                    mfdp_name=mfdp_name)
    truncate_spaces(n1=n1, n2=n2, dirpaths=a_dirpaths_map.values(),
                    path_temp=path_temp,
                    tbme_name_regex=tbme_name_regex)
    do_ncsd(a_values=a_values, a_prescription=a_prescription,
            a_dirpaths_map=a_dirpaths_map, a_outfile_map=a_outfile_map,
            force=force_ncsd or force_all)
    # for the 3rd a value, make trdens file and run TRDENS
    make_trdens_file(z=z, a=a_values[2], aeff=a_prescription[2], 
                     nuc_dir=dir_nuc,
                     path_results=results_path,
                     path_temp=path_temp,
                     trdens_name=trdens_name)
    a6_dir = a_dirpaths_map[a_values[2]]
    rename_egv_file(a6_dir=a6_dir, egv_name_regex=egv_name_regex,
                    next_egv_name=next_egv_name, 
                    force=force_trdens or force_all)
    do_trdens(a6_dir=a6_dir, force=force_trdens or force_all,
              outfile=outfile_trdens)
    # do valence cluster expansion
    if len(aeff_list) == 0:
        return 1
    vce_dirpath = path.join(results_path, dir_vce % str(a_prescription))
    if not path.exists(vce_dirpath):
        mkdir(vce_dirpath)
    do_vce(a_outfile_map=a_outfile_map, a_values=a_values, a6_dir=a6_dir, 
           heff_fname=heff_fname, aeff_list=aeff_list, 
           vce_int_fname=vce_int_fname, vce_dirpath=vce_dirpath,
           force=force_vce or force_all)
    return 1


# SCRIPT
ncsd_vce_calculations(a_prescription=(4, 5, 6), aeff_list=range(4, 17), nhw=4)
