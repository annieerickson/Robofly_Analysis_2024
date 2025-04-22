import os
import pathlib
import numpy as np
import numpy.matlib
import math
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
import h5py
from scipy.special import binom
import sys

sys.path.insert(1, '/Users/anneerickson/git/Robofly_Analysis_2024/')
from melis_etal_robofly import robofly_analysis_ae
from melis_etal_robofly import lollipop_figure_ae

def load_avg_wb_data(data_loc, input_loc, gain_factor=2.0):
    """"
    load in avg baseline and stim wingbeat runs
    data_loc: location of results
    input_loc: location of input files fed into robofly (to extract freq etc)
    gain_factor: gain factor to apply to calibration
    """
    
    RA = robofly_analysis_ae.RoboAnalysis()

    file_name_list1 = []
    for (dirpath, dirnames, filenames) in os.walk(data_loc):
        file_name_list1.extend(filenames)
        break

    # print(file_name_list1)

    file_name_list = []
    for file_n in file_name_list1:
        if 'results' in file_n:
            file_name_list.append(file_n)

    print(file_name_list)

    #center of gravity params from Melis et al, 2024
    body_cg = np.array([ 0.03604015,0.0,-0.23981816])

    RA.set_body_cg(body_cg)

    #these should match the vals from the robofly_notebook
    cg_L = np.array([[-0.16080577],
    [ 1.30565967],
    [ 0.        ]])

    M_L  = np.array([[ 1.60788240e-09,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00, -2.09934721e-09],
    [ 0.00000000e+00,  1.60788240e-09,  0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -2.58556764e-10],
    [ 0.00000000e+00,  0.00000000e+00,  1.60788240e-09,  2.09934721e-09, 2.58556764e-10,  0.00000000e+00],
    [ 0.00000000e+00, -0.00000000e+00,  2.09934721e-09,  6.31819550e-09, 5.47450644e-10,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  2.58556764e-10,  5.47450644e-10, 3.55374010e-10,  0.00000000e+00],
    [-2.09934721e-09, -2.58556764e-10,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  6.67356951e-09]])

    cg_R = np.array([[-0.16080577],
    [-1.30565967],
    [ 0.        ]])

    M_R  = np.array([[ 1.60788240e-09,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 2.09934721e-09],
    [ 0.00000000e+00,  1.60788240e-09,  0.00000000e+00, -0.00000000e+00,  0.00000000e+00, -2.58556764e-10],
    [ 0.00000000e+00,  0.00000000e+00,  1.60788240e-09, -2.09934721e-09,  2.58556764e-10,  0.00000000e+00],
    [ 0.00000000e+00, -0.00000000e+00, -2.09934721e-09,  6.31819550e-09, -5.47450644e-10,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  2.58556764e-10, -5.47450644e-10,  3.55374010e-10,  0.00000000e+00],
    [ 2.09934721e-09, -2.58556764e-10,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  6.67356951e-09]])

    beta = -(65/180)*np.pi
    RA.set_srf_angle(beta)
    RA.set_Inertia_tensors(cg_L,M_L,cg_R,M_R)

    strk_shift = np.array([
        [0,-(13/180)*np.pi]])


    bl_stim_avg_list = RA.get_L_R_fnames_avg_bl_stim(file_name_list)

    for i in range(len(bl_stim_avg_list)):
        print(bl_stim_avg_list[i])
        # RA.load_mat_file(file_loc,bl_stim_avg_list[i], gain_factor=2.0)
        RA.load_mat_file(data_loc,bl_stim_avg_list[i], gain_factor=gain_factor)
        beta_i 		= strk_shift[0,0]
        phi_shift_i = strk_shift[0,1]
        side = bl_stim_avg_list[i].split('_')[1] #'L' or 'R'

        #get corresponding input file
        #dont need if using data file
        input_file = bl_stim_avg_list[i][:-12] + '.txt' #same name without results.mat
        input_fpath = input_loc + input_file
        input_data = np.loadtxt(input_fpath, delimiter=',')
        #get frequency, change per wb
        n_fly = input_data[0:1,4]
        print(f'input_file freq {n_fly}')
        RA.add_freq_and_scaling(freq_fly=n_fly, n_wbs=7)

        RA.convert_to_SRF(beta_i,phi_shift_i, wing_side=side)

        # order added to list is: L baseline, R baseline, L stim, R stim
        RA.add_data_to_list_means()

    return RA

data_location = '/Volumes/My Passport for Mac/robofly/coef_for_robofly_updated_eta_bounds/1070/results'
input_location = '/Volumes/My Passport for Mac/robofly/coef_for_robofly_updated_eta_bounds/1070/coef_to_robofly/'
save_location = save_directory = '/Volumes/Untitled/DNa10_paper_figs/robofly/svgs_nature/'
RA = load_avg_wb_data(data_location, input_location, gain_factor=1.0)
RA.plot_LpR_forces_baseline_stim_bilateral(save_location, 
                                                include_inertial=False, 
                                                gain_factor=1.0,)