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

from .lollipop_figure_ae import Lollipop


class RoboAnalysis():

    def __init__(self):
        self.dt = 0.1
        self.dx = 0.1
        self.dy = 0.1
        self.dz = 0.1
        self.g = 9800.0
        # self.f = 200.0
        self.mass = 1.034e-6
        self.rho_fly = 1.18e-9 #density of air
        self.rho_robo = 880.0e-9 #density oil
        self.R_fly = 2.7 #mm length of fly wing
        self.R_robo = 250.0 #length robofly wing
        # self.n_fly = 180.0 #freq of fly #define on a per trial basis
        # self.n_robo = self.n_fly*(self.R_fly**2/self.R_robo**2)*(115/15.5) #~1.4x diff from n_robo old (although calc on a per trial basis- could explain gain factor of 2?)
        # self.F_scaling = 1000*(self.rho_fly*self.n_fly**2*self.R_fly**4)/(self.rho_robo*self.n_robo**2*self.R_robo**4) #define per trial
        # print('F scaling')
        # print(self.F_scaling)
        # print('')
        # self.M_scaling = 1000*(self.rho_fly*self.n_fly**2*self.R_fly**5)/(self.rho_robo*self.n_robo**2*self.R_robo**5)
        # print('M scaling')
        # print(self.M_scaling)
        # print('')
        # time
        self.t_FT_f_list = []
        self.t_FT_s_list = []
        self.t_P_f_list = []
        self.t_P_s_list = []
        # FT
        self.FT_gravity_list = []
        self.FT_data_f_list = []
        self.FT_data_s_list = []
        self.FT_filt_f_list = []
        self.FT_filt_s_list = []
        self.FT_out_list = []
        self.FT_raw_f_list = []
        self.FT_raw_s_list = []
        self.FT_trace_f_list = []
        self.FT_trace_s_list = []
        # bias
        self.bias_data_wing_list = []
        self.bias_wing_list = []
        # filter
        self.filter_gain_f_list = []
        self.filter_gain_s_list = []
        # Wing kinematics
        self.wingkin_f_list = []
        self.wingkin_s_list = []
        # time
        self.t_FT_fast_list = []
        self.N_FT_fast_list = []
        self.t_FT_slow_list = []
        self.N_FT_slow_list = []
        # Non-dimensional time
        self.T_fast_list = []
        self.T_slow_list = []
        # SRF
        self.FT_SRF_list = []
        self.wingkin_SRF_list= []
        # FT wb
        # self.FT_wb_median_list = []
        # self.FT_wb_median_wing_list = []
        self.FT_wb_mean_list = []
        self.FT_wb_mean_wing_list = []
        self.FT_wb_std_list = []
        self.FT_wb_3_list = []
        self.FT_wb_3_mean_list = []
        self.wingkin_wb_3_list = []
        self.FT_wb_4_list = []
        self.FT_wb_4_mean_list = []
        self.wingkin_wb_4_list = []
        self.FT_wb_5_list = []
        self.FT_wb_5_mean_list = []
        self.wingkin_wb_5_list = []
        self.FT_wb_6_list = []
        self.FT_wb_6_mean_list = []
        self.wingkin_wb_6_list = []
        self.FT_wb_7_list = []
        self.FT_wb_7_mean_list = []
        self.wingkin_wb_7_list = []
        self.FT_wb_8_list = []
        self.FT_wb_8_mean_list = []
        self.wingkin_wb_8_list = []
        self.FT_wing_list = []
        # Inertia
        self.FTI_vel_w_list = []
        self.FTI_acc_w_list = []
        self.FTI_vel_b_list = []
        self.FTI_acc_b_list = []
        # FT total wing:
        self.FT_total_wing_list = []
        # mean forces
        self.FT_mean_list = []

        # just for L and R avg wbs
        self.t_FT_f_list_means = []
        self.t_FT_s_list_means = []
        self.t_P_f_list_means = []
        self.t_P_s_list_means = []
        # FT
        self.FT_gravity_list_means = []
        self.FT_data_f_list_means = []
        self.FT_data_s_list_means = []
        self.FT_filt_f_list_means = []
        self.FT_filt_s_list_means = []
        self.FT_out_list_means = []
        self.FT_raw_f_list_means = []
        self.FT_raw_s_list_means = []
        self.FT_trace_f_list_means = []
        self.FT_trace_s_list_means = []
        # bias
        self.bias_data_wing_list_means = []
        self.bias_wing_list_means = []
        # filter
        self.filter_gain_f_list_means = []
        self.filter_gain_s_list_means = []
        # Wing kinematics
        self.wingkin_f_list_means = []
        self.wingkin_s_list_means = []
        # time
        self.t_FT_fast_list_means = []
        self.N_FT_fast_list_means = []
        self.t_FT_slow_list_means = []
        self.N_FT_slow_list_means = []
        # Non-dimensional time
        self.T_fast_list_means = []
        self.T_slow_list_means = []
        # SRF
        self.FT_SRF_list_means = []
        self.wingkin_SRF_list_means= []
        # FT wb
        # self.FT_wb_median_list_means = []
        # self.FT_wb_median_wing_list_means = []
        # self.FT_wb_std_list_means = []
        # self.FT_wb_3_list_means = []
        # self.FT_wb_3_mean_list_means = []
        # self.wingkin_wb_3_list_means = []
        # self.FT_wb_4_list_means = []
        # self.FT_wb_4_mean_list_means = []
        # self.wingkin_wb_4_list_means = []
        # self.FT_wb_5_list_means= []
        # self.FT_wb_5_mean_list_means = []
        # self.wingkin_wb_5_list_means = []
        # self.FT_wb_6_list_means = []
        # self.FT_wb_6_mean_list_means = []
        # self.wingkin_wb_6_list_means = []
        # self.FT_wb_7_list = []
        # self.FT_wb_7_mean_list = []
        # self.wingkin_wb_7_list = []
        # self.FT_wb_8_list = []
        # self.FT_wb_8_mean_list = []
        # self.wingkin_wb_8_list = []
        self.FT_wing_list_means = []
        # angular velocities
        self.omega_list_means = []
        # power
        self.power_list_means = []
        self.freq_list_means = []
        self.M_scaling_list_means = []
        self.F_scaling_list_means = []


    def add_freq_and_scaling(self, freq_fly, n_wbs):
        """
        create freq variable per trial (input, robofly) and calc scaling from 
        """
        self.n_fly 		  = freq_fly
        print(self.n_fly)
        self.f = freq_fly 
        self.n_robo 	  = n_wbs/(self.t_FT_fast[-1]*(10**-6))
        print(str(self.n_robo))
        self.F_scaling 	  = 1000*(self.rho_fly*self.n_fly**2*self.R_fly**4)/(self.rho_robo*self.n_robo**2*self.R_robo**4)
        self.M_scaling 	  = 1000*(self.rho_fly*self.n_fly**2*self.R_fly**5)/(self.rho_robo*self.n_robo**2*self.R_robo**5)
        #add to list
        self.freq_list_means.append(self.n_fly)
        self.F_scaling_list_means.append(self.F_scaling)
        self.M_scaling_list_means.append(self.M_scaling)


    def load_mat_file(self,file_loc_in,file_name_in):
        file_path = pathlib.Path(file_loc_in, file_name_in)
        self.mat_file = scipy.io.loadmat(file_path)
        # time
        self.t_FT_f = np.squeeze(self.mat_file['t_FT_f'])
        self.t_FT_s = np.squeeze(self.mat_file['t_FT_s'])
        # FT
        self.FT_gravity = np.squeeze(self.mat_file['FT_G'])
        self.FT_data_f = np.squeeze(self.mat_file['FT_data_f'])
        self.FT_data_s = np.squeeze(self.mat_file['FT_data_s'])
        self.FT_filt_f = np.squeeze(self.mat_file['FT_filt_f'])
        self.FT_filt_s = np.squeeze(self.mat_file['FT_filt_s'])
        self.FT_out = np.squeeze(self.mat_file['FT_out'])
        self.FT_raw_f = np.squeeze(self.mat_file['FT_raw_f'])
        self.FT_raw_s = np.squeeze(self.mat_file['FT_raw_s'])
        self.FT_trace_f = np.squeeze(self.mat_file['FT_trace_f'])
        # bias
        self.bias_data_wing = np.squeeze(self.mat_file['bias_data_wing'])
        self.bias_wing = np.squeeze(self.mat_file['bias_wing'])
        # filter
        self.filter_gain_f = np.squeeze(self.mat_file['filter_gain_f'])
        self.filter_gain_s = np.squeeze(self.mat_file['filter_gain_s'])
        # Wing kinematics
        self.wingkin_f = np.squeeze(self.mat_file['wingkin_f'])
        self.wingkin_s = np.squeeze(self.mat_file['wingkin_s'])
        # time
        self.t_FT_fast = self.wingkin_f[:,0]
        self.N_FT_fast = self.wingkin_f.shape[0]
        #this should actually be 5.0 times slower 
        # self.t_FT_slow = self.wingkin_s[:,0]/4.0
        self.t_FT_slow = self.wingkin_s[:,0]/5.0
        self.N_FT_slow = self.wingkin_s.shape[0]
        # Non-dimensional time
        self.T_fast = (self.t_FT_fast/self.t_FT_fast[-1])*7.0
        self.T_slow = (self.t_FT_slow/self.t_FT_slow[-1])*7.0
        # FT wing
        self.FT_wing = self.compute_FT_wing(self.FT_out)

    #still do sign flip on the right wing for 1,3,5
    #add a gain factor?
    def compute_FT_wing(self,FT_in):
        L_cross = np.array([[0.0,-42.0,0.0],[42,0.0,0.0],[0.0,0.0,0.0]])
        R_wing1 = np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,-1.0,0.0]])
        R_wing2 = np.array([[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,-1.0,0.0]])
        R_mat = np.zeros((6,6))
        R_mat[0:3,0:3] = R_wing1
        R_mat[3:6,0:3] = np.dot(R_wing2,L_cross)
        R_mat[3:6,3:6] = R_wing2
        FT_neg = FT_in
        FT_neg[1,:] = -FT_neg[1,:]
        FT_neg[3,:] = -FT_neg[3,:]
        FT_neg[5,:] = -FT_neg[5,:]
        FT_w = np.dot(R_mat,FT_neg)
        return FT_w

    # for a single trial 
    def add_data_to_list(self):
        # time
        self.t_FT_f_list.append(self.t_FT_f)
        self.t_FT_s_list.append(self.t_FT_s)
        # FT
        self.FT_gravity_list.append(self.FT_gravity)
        self.FT_data_f_list.append(self.FT_data_f)
        self.FT_data_s_list.append(self.FT_data_s)
        self.FT_filt_f_list.append(self.FT_filt_f)
        self.FT_filt_s_list.append(self.FT_filt_s)
        self.FT_out_list.append(self.FT_out)
        self.FT_raw_f_list.append(self.FT_raw_f)
        self.FT_trace_f_list.append(self.FT_trace_f)
        # bias
        self.bias_data_wing_list.append(self.bias_data_wing)
        self.bias_wing_list.append(self.bias_wing)
        # filter
        self.filter_gain_f_list.append(self.filter_gain_f)
        self.filter_gain_s_list.append(self.filter_gain_s)
        # Wing kinematics
        self.wingkin_f_list.append(self.wingkin_f)
        self.wingkin_s_list.append(self.wingkin_s)
        # time
        self.t_FT_fast_list.append(self.t_FT_fast)
        self.N_FT_fast_list.append(self.N_FT_fast)
        self.t_FT_slow_list.append(self.t_FT_slow)
        self.N_FT_slow_list.append(self.N_FT_slow)
        # Non-dimensional time
        self.T_fast_list.append(self.T_fast)
        self.T_slow_list.append(self.T_slow)
        # FT_strkpln
        self.FT_SRF_list.append(self.FT_SRF)
        self.wingkin_SRF_list.append(self.wingkin_SRF)
        # FT wing:
        self.FT_wing_list.append(self.FT_wing)
        # Inertia:
        self.FTI_vel_w_list.append(self.FTI_vel_Lw)
        self.FTI_acc_w_list.append(self.FTI_acc_Lw)
        self.FTI_vel_b_list.append(self.FTI_vel_Lb)
        self.FTI_acc_b_list.append(self.FTI_acc_Lb)
        # Total forces
        #FT_wb_median needs to be scaled to fly scale *F_scaling
        #FT_total = self.FTI_vel_Lw+self.FTI_acc_Lw+self.FT_wb_median
        #scaled
        # FT_total = (self.FTI_vel_Lw+self.FTI_acc_Lw+self.FT_wb_median*self.F_scaling)
        FT_total = (self.FTI_vel_Lw+self.FTI_acc_Lw+self.FT_wb_mean*self.F_scaling)
        #FT_total = self.FT_wb_median
        self.FT_total_wing_list.append(FT_total)
        # Mean forces
        self.FT_mean_list.append(self.FT_mean)

    # for left and right trials of baseline and stim periods
    def add_data_to_list_means(self):
        # time
        self.t_FT_f_list_means.append(self.t_FT_f)
        self.t_FT_s_list_means.append(self.t_FT_s)
        # FT
        self.FT_gravity_list_means.append(self.FT_gravity)
        self.FT_data_f_list_means.append(self.FT_data_f)
        self.FT_data_s_list_means.append(self.FT_data_s)
        self.FT_filt_f_list_means.append(self.FT_filt_f)
        self.FT_filt_s_list_means.append(self.FT_filt_s)
        self.FT_out_list_means.append(self.FT_out)
        self.FT_raw_f_list_means.append(self.FT_raw_f)
        #.FT_raw_s_list.append(self.FT_raw_s)
        self.FT_trace_f_list_means.append(self.FT_trace_f)
        #self.FT_trace_s_list.append(self.FT_trace_s)
        # bias
        self.bias_data_wing_list_means.append(self.bias_data_wing)
        self.bias_wing_list_means.append(self.bias_wing)
        # filter
        self.filter_gain_f_list_means.append(self.filter_gain_f)
        self.filter_gain_s_list_means.append(self.filter_gain_s)
        # Wing kinematics
        self.wingkin_f_list_means.append(self.wingkin_f)
        self.wingkin_s_list_means.append(self.wingkin_s)
        # time
        self.t_FT_fast_list_means.append(self.t_FT_fast)
        self.N_FT_fast_list_means.append(self.N_FT_fast)
        self.t_FT_slow_list_means.append(self.t_FT_slow)
        self.N_FT_slow_list_means.append(self.N_FT_slow)
        # Non-dimensional time
        self.T_fast_list_means.append(self.T_fast)
        self.T_slow_list_means.append(self.T_slow)
        # FT_strkpln
        self.FT_SRF_list_means.append(self.FT_SRF)
        self.wingkin_SRF_list_means.append(self.wingkin_SRF)
        # FT wb
        # self.FT_wb_5_list_means.append(self.FT_wb_5)
        # self.FT_wb_5_mean_list_means.append(self.FT_wb_5_mean)
        # self.wingkin_wb_5_list_means.append(self.wingkin_wb_5)
        # self.FT_wb_6_list_means.append(self.FT_wb_6)
        # self.FT_wb_6_mean_list_means.append(self.FT_wb_6_mean)
        # self.wingkin_wb_6_list_means.append(self.wingkin_wb_6)
        
        self.FT_wb_mean_list_means.append(self.FT_wb_mean)
        self.FT_wb_mean_wing_list_means.append(self.FT_wb_mean_wing)
        # std
        # self.FT_wb_std_list_means.append(self.FT_wb_std)
        # FT wing:
        self.FT_wing_list_means.append(self.FT_wing)
        
        self.power_list_means.append(self.power)
        

    def set_srf_angle(self,srf_in):
        self.srf_angle = srf_in

    def set_body_cg(self,cg_in):
        self.body_cg = cg_in

    def euler_angle_shift(self,phi,theta,eta,xi,beta,phi_shift):
        phi_n      = -phi+phi_shift
        eta_n      = -eta
        theta_n  = theta
        xi_n      = -3.0*xi
        return phi_n,theta_n,eta_n,xi_n
    
    #if eta = eta_0 + xi/3 in 'coef_to_robofly wb' save_wb_2_txt function
    def euler_angle_shift_shift_eta(self,phi,theta,eta,xi,beta,phi_shift):
        phi_n      = -phi+phi_shift
        theta_n    = theta
        xi_n       = -3.0*xi
        eta_n      = -eta-(xi_n/3) 
        return phi_n,theta_n,eta_n,xi_n

    #need to add in right wing?, do anything different for aero forces with right wing?
    def convert_to_SRF(self,beta,phi_shift, shift_eta=True):

        wb_select = ((self.T_fast>=4.0)&(self.T_fast<5.0))

        wb_select_4 = ((self.T_fast>=3.0)&(self.T_fast<4.0))
        wb_select_5 = ((self.T_fast>=4.0)&(self.T_fast<5.0))
        wb_select_6 = ((self.T_fast>=5.0)&(self.T_fast<6.0))

        self.N_pts = np.sum(wb_select)

        self.dt = 1.0/(self.N_pts*self.f)

        self.t = np.linspace(0,1/self.f,num=self.N_pts)

        # xi_Lt      = np.pi*(self.wingkin_f[wb_select,5]/180.0)
        # theta_Lt = np.pi*(self.wingkin_f[wb_select,1]/180.0)
        # eta_Lt      = np.pi*(self.wingkin_f[wb_select,2]/180.0)
        # phi_Lt      = np.pi*(self.wingkin_f[wb_select,3]/180.0)

        #take mean of wb 4,5,6
        wb_456 = np.zeros((self.N_pts,13,3))
        wb_456[:,:,0] = np.pi*(self.wingkin_f[wb_select_4,:]/180.0)
        wb_456[:,:,1] = np.pi*(self.wingkin_f[wb_select_5,:]/180.0)
        wb_456[:,:,2] = np.pi*(self.wingkin_f[wb_select_6,:]/180.0)
        wb_mean = np.mean(wb_456, axis=2)

        xi_Lt      = wb_mean[:,5]
        theta_Lt   = wb_mean[:,1]
        eta_Lt     = wb_mean[:,2]
        phi_Lt     = wb_mean[:,3]
        


        if shift_eta==False:
            self.phi_L,self.theta_L,self.eta_L,self.xi_L = self.euler_angle_shift(phi_Lt,theta_Lt,eta_Lt,xi_Lt,-beta,-phi_shift)
        if shift_eta==True:
            self.phi_L,self.theta_L,self.eta_L,self.xi_L = self.euler_angle_shift_shift_eta(phi_Lt,theta_Lt,eta_Lt,xi_Lt,-beta,-phi_shift)

        self.theta_dot_L = np.squeeze(np.gradient(self.theta_L,self.dt,edge_order=2))
        self.eta_dot_L      = np.squeeze(np.gradient(self.eta_L,self.dt,edge_order=2))
        self.phi_dot_L      = np.squeeze(np.gradient(self.phi_L,self.dt,edge_order=2))

        self.theta_ddot_L = np.squeeze(np.gradient(self.theta_dot_L,self.dt,edge_order=2))
        self.eta_ddot_L   = np.squeeze(np.gradient(self.eta_dot_L,self.dt,edge_order=2))
        self.phi_ddot_L   = np.squeeze(np.gradient(self.phi_dot_L,self.dt,edge_order=2))

        
        #fig1, axs1 = plt.subplots(3,3)
        #fig1.set_figwidth(12)
        #fig1.set_figheight(12)
        #axs1[0,0].plot(self.t,self.theta_L*(180/np.pi))
        #axs1[0,1].plot(self.t,self.eta_L*(180/np.pi))
        #axs1[0,2].plot(self.t,self.phi_L*(180/np.pi))
        #axs1[1,0].plot(self.t,self.theta_dot_L*(180/np.pi))
        #axs1[1,1].plot(self.t,self.eta_dot_L*(180/np.pi))
        #axs1[1,2].plot(self.t,self.phi_dot_L*(180/np.pi))
        #axs1[2,0].plot(self.t,self.theta_ddot_L*(180/np.pi))
        #axs1[2,1].plot(self.t,self.eta_ddot_L*(180/np.pi))
        #axs1[2,2].plot(self.t,self.phi_ddot_L*(180/np.pi))

        cg_cross = np.array([[0,-self.body_cg[2],self.body_cg[1]],[self.body_cg[2],0,-self.body_cg[0]],[-self.body_cg[1],self.body_cg[0],0]])        

        q_beta = np.array([np.cos(self.srf_angle/2.0),0.0,np.sin(self.srf_angle/2.0),0.0])

        R_beta = self.comp_R(q_beta)

        R_90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])

        self.FT_SRF = np.zeros((6,self.N_pts))

        self.wingkin_SRF = np.zeros((4,self.N_pts))

        # Compute angular velocities:
        self.R_Lw = np.zeros((3,3,self.N_pts))
        self.w_Lw = np.zeros((3,self.N_pts))
        self.w_dot_Lw = np.zeros((3,self.N_pts))

        self.FTI_acc_Lw = np.zeros((6,self.N_pts))
        self.FTI_vel_Lw = np.zeros((6,self.N_pts))
        self.FTI_acc_Lb = np.zeros((6,self.N_pts))
        self.FTI_vel_Lb = np.zeros((6,self.N_pts))

        FT_456 = np.zeros((6,self.N_pts,3))
        FT_456[:,:,0] = self.FT_wing[:,wb_select_4]
        FT_456[:,:,1] = self.FT_wing[:,wb_select_5]
        FT_456[:,:,2] = self.FT_wing[:,wb_select_6]

        # self.FT_wb_median = np.mean(FT_456,axis=2)
        self.FT_wb_mean = np.mean(FT_456,axis=2)


        for i in range(self.N_pts):
            
            q_phi_L   = np.array([np.cos(self.phi_L[i]/2.0),np.sin(self.phi_L[i]/2.0),0.0,0.0])
            q_theta_L = np.array([np.cos(-self.theta_L[i]/2.0),0.0,0.0,np.sin(-self.theta_L[i]/2.0)])
            q_eta_L   = np.array([np.cos(self.eta_L[i]/2.0),0.0,np.sin(self.eta_L[i]/2.0),0.0])
            phi_dot_L_vec = np.array([[self.phi_dot_L[i]],[0.0],[0.0]])
            theta_dot_L_vec = np.array([[0.0],[0.0],[-self.theta_dot_L[i]]])
            eta_dot_L_vec = np.array([[0.0],[self.eta_dot_L[i]],[0.0]])
            phi_ddot_L_vec = np.array([[self.phi_ddot_L[i]],[0.0],[0.0]])
            theta_ddot_L_vec = np.array([[0.0],[0.0],[-self.theta_ddot_L[i]]])
            eta_ddot_L_vec = np.array([[0.0],[self.eta_ddot_L[i]],[0.0]])
            q_L = self.q_mult(q_phi_L,self.q_mult(q_theta_L,q_eta_L))
            R_L = np.transpose(self.comp_R(q_L))
            self.R_Lw[:,:,i] = R_L
            self.w_Lw[:,i] = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta_L,q_theta_L)),phi_dot_L_vec)+np.dot(self.comp_R(q_eta_L),theta_dot_L_vec)+eta_dot_L_vec)
            self.w_dot_Lw[:,i] = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta_L,q_theta_L)),phi_ddot_L_vec)+np.dot(self.comp_R(q_eta_L),theta_ddot_L_vec)+eta_ddot_L_vec)
            
            R_mat = np.zeros((6,6))
            R_mat[:3,:3] = np.dot(np.transpose(R_beta),R_L)
            R_mat[3:,3:] = np.dot(np.transpose(R_beta),R_L)

            # FT_i = self.FT_wb_median[:,i]
            FT_i = self.FT_wb_mean[:,i]

            self.FT_SRF[:,i] = np.dot(R_mat,FT_i)
            self.wingkin_SRF[0,i] = self.theta_L[i]
            self.wingkin_SRF[1,i] = self.eta_L[i]
            self.wingkin_SRF[2,i] = self.phi_L[i]
            self.wingkin_SRF[3,i] = self.xi_L[i]

            w_L_cross = np.array([[0.0,-self.w_Lw[2,i],self.w_Lw[1,i]],[self.w_Lw[2,i],0.0,-self.w_Lw[0,i]],[-self.w_Lw[1,i],self.w_Lw[0,i],0.0]])

            self.FTI_acc_Lw[:3,i] = -np.dot(self.MwL[:3,3:],self.w_dot_Lw[:,i])
            self.FTI_acc_Lw[3:,i] = -np.dot(self.MwL[3:,3:],self.w_dot_Lw[:,i])
            self.FTI_acc_Lb[:3,i] = np.squeeze(np.dot(np.transpose(R_beta),np.dot(R_L,self.FTI_acc_Lw[:3,i])))
            self.FTI_acc_Lb[3:,i] = np.squeeze(np.dot(np.transpose(R_beta),np.dot(R_L,self.FTI_acc_Lw[3:,i])))
            self.FTI_acc_Lb[3:,i] += np.dot(cg_cross,self.FTI_acc_Lb[:3,i])

            self.FTI_vel_Lw[:3,i] = -np.squeeze(self.wing_L_m*np.dot(w_L_cross,np.dot(w_L_cross,self.wing_L_cg)))
            self.FTI_vel_Lw[3:,i] = -np.squeeze(np.dot(w_L_cross,np.dot(self.wing_L_I,self.w_Lw[:,i])))
            self.FTI_vel_Lb[:3,i] = np.squeeze(np.dot(np.transpose(R_beta),np.dot(R_L,self.FTI_vel_Lw[:3,i])))
            self.FTI_vel_Lb[3:,i] = np.squeeze(np.dot(np.transpose(R_beta),np.dot(R_L,self.FTI_vel_Lw[3:,i])))
            self.FTI_vel_Lb[3:,i] += np.dot(cg_cross,self.FTI_vel_Lb[3:,i])

        self.FT_SRF[3:,:] += np.dot(cg_cross,self.FT_SRF[:3,:])


        t_ones = np.ones(self.N_pts)

        Fg = self.mass*self.g
        FgR = self.mass*self.g*self.R_fly

        
        fig3, axs3 = plt.subplots(2,3)
        fig3.set_figwidth(12)
        fig3.set_figheight(12)
        axs3[0,0].plot(self.t,self.FT_SRF[0,:]*self.F_scaling/Fg,color='b')
        axs3[0,1].plot(self.t,self.FT_SRF[1,:]*self.F_scaling/Fg,color='b')
        axs3[0,2].plot(self.t,self.FT_SRF[2,:]*self.F_scaling/Fg,color='b')
        axs3[1,0].plot(self.t,self.FT_SRF[3,:]*self.M_scaling/FgR,color='b')
        axs3[1,1].plot(self.t,self.FT_SRF[4,:]*self.M_scaling/FgR,color='b')
        axs3[1,2].plot(self.t,self.FT_SRF[5,:]*self.M_scaling/FgR,color='b')
        axs3[0,0].plot(self.t,self.FTI_acc_Lb[0,:]/Fg,color='g')
        axs3[0,1].plot(self.t,self.FTI_acc_Lb[1,:]/Fg,color='g')
        axs3[0,2].plot(self.t,self.FTI_acc_Lb[2,:]/Fg,color='g')
        axs3[1,0].plot(self.t,self.FTI_acc_Lb[3,:]/FgR,color='g')
        axs3[1,1].plot(self.t,self.FTI_acc_Lb[4,:]/FgR,color='g')
        axs3[1,2].plot(self.t,self.FTI_acc_Lb[5,:]/FgR,color='g')
        axs3[0,0].plot(self.t,self.FTI_vel_Lb[0,:]/Fg,color='r')
        axs3[0,1].plot(self.t,self.FTI_vel_Lb[1,:]/Fg,color='r')
        axs3[0,2].plot(self.t,self.FTI_vel_Lb[2,:]/Fg,color='r')
        axs3[1,0].plot(self.t,self.FTI_vel_Lb[3,:]/FgR,color='r')
        axs3[1,1].plot(self.t,self.FTI_vel_Lb[4,:]/FgR,color='r')
        axs3[1,2].plot(self.t,self.FTI_vel_Lb[5,:]/FgR,color='r')
        axs3[0,0].plot(self.t,(self.FT_SRF[0,:]*self.F_scaling+self.FTI_acc_Lb[0,:]+self.FTI_vel_Lb[0,:])/Fg,color='k')
        axs3[0,1].plot(self.t,(self.FT_SRF[1,:]*self.F_scaling+self.FTI_acc_Lb[1,:]+self.FTI_vel_Lb[1,:])/Fg,color='k')
        axs3[0,2].plot(self.t,(self.FT_SRF[2,:]*self.F_scaling+self.FTI_acc_Lb[2,:]+self.FTI_vel_Lb[2,:])/Fg,color='k')
        axs3[1,0].plot(self.t,(self.FT_SRF[3,:]*self.M_scaling+self.FTI_acc_Lb[3,:]+self.FTI_vel_Lb[3,:])/FgR,color='k')
        axs3[1,1].plot(self.t,(self.FT_SRF[4,:]*self.M_scaling+self.FTI_acc_Lb[4,:]+self.FTI_vel_Lb[4,:])/FgR,color='k')
        axs3[1,2].plot(self.t,(self.FT_SRF[5,:]*self.M_scaling+self.FTI_acc_Lb[5,:]+self.FTI_vel_Lb[5,:])/FgR,color='k')
        #axs3[0,0].plot(self.t,np.mean(self.FT_SRF[0,:]*self.F_scaling)*t_ones,color='b')
        #axs3[0,1].plot(self.t,np.mean(self.FT_SRF[1,:]*self.F_scaling)*t_ones,color='b')
        #axs3[0,2].plot(self.t,np.mean(self.FT_SRF[2,:]*self.F_scaling)*t_ones,color='b')
        #axs3[1,0].plot(self.t,np.mean(self.FT_SRF[3,:]*self.M_scaling)*t_ones,color='b')
        #axs3[1,1].plot(self.t,np.mean(self.FT_SRF[4,:]*self.M_scaling)*t_ones,color='b')
        #axs3[1,2].plot(self.t,np.mean(self.FT_SRF[5,:]*self.M_scaling)*t_ones,color='b')
        #axs3[0,0].plot(self.t,np.mean(self.FTI_acc_Lb[0,:])*t_ones,color='g')
        #axs3[0,1].plot(self.t,np.mean(self.FTI_acc_Lb[1,:])*t_ones,color='g')
        #axs3[0,2].plot(self.t,np.mean(self.FTI_acc_Lb[2,:])*t_ones,color='g')
        #axs3[1,0].plot(self.t,np.mean(self.FTI_acc_Lb[3,:])*t_ones,color='g')
        #axs3[1,1].plot(self.t,np.mean(self.FTI_acc_Lb[4,:])*t_ones,color='g')
        #axs3[1,2].plot(self.t,np.mean(self.FTI_acc_Lb[5,:])*t_ones,color='g')
        #axs3[0,0].plot(self.t,np.mean(self.FTI_vel_Lb[0,:])*t_ones,color='r')
        #axs3[0,1].plot(self.t,np.mean(self.FTI_vel_Lb[1,:])*t_ones,color='r')
        #axs3[0,2].plot(self.t,np.mean(self.FTI_vel_Lb[2,:])*t_ones,color='r')
        #axs3[1,0].plot(self.t,np.mean(self.FTI_vel_Lb[3,:])*t_ones,color='r')
        #axs3[1,1].plot(self.t,np.mean(self.FTI_vel_Lb[4,:])*t_ones,color='r')
        #axs3[1,2].plot(self.t,np.mean(self.FTI_vel_Lb[5,:])*t_ones,color='r')
        #axs3[0,0].plot(self.t,np.mean(self.FT_SRF[0,:]*self.F_scaling+self.FTI_acc_Lb[0,:]+self.FTI_vel_Lb[0,:])*t_ones,color='k')
        #axs3[0,1].plot(self.t,np.mean(self.FT_SRF[1,:]*self.F_scaling+self.FTI_acc_Lb[1,:]+self.FTI_vel_Lb[1,:])*t_ones,color='k')
        #axs3[0,2].plot(self.t,np.mean(self.FT_SRF[2,:]*self.F_scaling+self.FTI_acc_Lb[2,:]+self.FTI_vel_Lb[2,:])*t_ones,color='k')
        #axs3[1,0].plot(self.t,np.mean(self.FT_SRF[3,:]*self.M_scaling+self.FTI_acc_Lb[3,:]+self.FTI_vel_Lb[3,:])*t_ones,color='k')
        #axs3[1,1].plot(self.t,np.mean(self.FT_SRF[4,:]*self.M_scaling+self.FTI_acc_Lb[4,:]+self.FTI_vel_Lb[4,:])*t_ones,color='k')
        #axs3[1,2].plot(self.t,np.mean(self.FT_SRF[5,:]*self.M_scaling+self.FTI_acc_Lb[5,:]+self.FTI_vel_Lb[5,:])*t_ones,color='k')
        
        # Save FT_mean
        # not sure that I will need this 
        FT_m_array = np.zeros((6,4))
        FT_m_array[:3,0] = np.mean(self.FT_SRF[:3,:]*self.F_scaling,axis=1)
        FT_m_array[3:,0] = np.mean(self.FT_SRF[3:,:]*self.M_scaling,axis=1)
        FT_m_array[:,1] = np.mean(self.FTI_acc_Lb,axis=1)
        FT_m_array[:,2] = np.mean(self.FTI_vel_Lb,axis=1)
        FT_m_array[:,3] = FT_m_array[:,0]+FT_m_array[:,1]+FT_m_array[:,2]
        self.FT_mean = FT_m_array

    #plotting functions for testing 
    def plot_wing_kinematics_and_forces_breakdown(self):
        """"
        plot left wing (for now) total forces (FT SRF + Lb inerital) for baseline
        plot left wing forces broken down by FT SRF and Lb vel, acceleration separately
        """
        Fg = self.mass*self.g
        FgR = self.mass*self.g*self.R_fly

        
        fig, axs = plt.subplots(2,3)
    

        #aero force only (SRF)
        axs[0,0].plot(self.t,self.FT_SRF[0,:]*self.F_scaling/Fg,color='b')
        axs[0,1].plot(self.t,self.FT_SRF[1,:]*self.F_scaling/Fg,color='b')
        axs[0,2].plot(self.t,self.FT_SRF[2,:]*self.F_scaling/Fg,color='b')
        axs[1,0].plot(self.t,self.FT_SRF[3,:]*self.M_scaling/FgR,color='b')
        axs[1,1].plot(self.t,self.FT_SRF[4,:]*self.M_scaling/FgR,color='b')
        axs[1,2].plot(self.t,self.FT_SRF[5,:]*self.M_scaling/FgR,color='b')

        #acceleration
        axs[0,0].plot(self.t,self.FTI_acc_Lb[0,:]/Fg,color='g')
        axs[0,1].plot(self.t,self.FTI_acc_Lb[1,:]/Fg,color='g')
        axs[0,2].plot(self.t,self.FTI_acc_Lb[2,:]/Fg,color='g')
        axs[1,0].plot(self.t,self.FTI_acc_Lb[3,:]/FgR,color='g')
        axs[1,1].plot(self.t,self.FTI_acc_Lb[4,:]/FgR,color='g')
        axs[1,2].plot(self.t,self.FTI_acc_Lb[5,:]/FgR,color='g')

        #velocity
        axs[0,0].plot(self.t,self.FTI_vel_Lb[0,:]/Fg,color='r')
        axs[0,1].plot(self.t,self.FTI_vel_Lb[1,:]/Fg,color='r')
        axs[0,2].plot(self.t,self.FTI_vel_Lb[2,:]/Fg,color='r')
        axs[1,0].plot(self.t,self.FTI_vel_Lb[3,:]/FgR,color='r')
        axs[1,1].plot(self.t,self.FTI_vel_Lb[4,:]/FgR,color='r')
        axs[1,2].plot(self.t,self.FTI_vel_Lb[5,:]/FgR,color='r')

        #all together
        axs[0,0].plot(self.t,(self.FT_SRF[0,:]*self.F_scaling+self.FTI_acc_Lb[0,:]+self.FTI_vel_Lb[0,:])/Fg,color='k')
        axs[0,1].plot(self.t,(self.FT_SRF[1,:]*self.F_scaling+self.FTI_acc_Lb[1,:]+self.FTI_vel_Lb[1,:])/Fg,color='k')
        axs[0,2].plot(self.t,(self.FT_SRF[2,:]*self.F_scaling+self.FTI_acc_Lb[2,:]+self.FTI_vel_Lb[2,:])/Fg,color='k')
        axs[1,0].plot(self.t,(self.FT_SRF[3,:]*self.M_scaling+self.FTI_acc_Lb[3,:]+self.FTI_vel_Lb[3,:])/FgR,color='k')
        axs[1,1].plot(self.t,(self.FT_SRF[4,:]*self.M_scaling+self.FTI_acc_Lb[4,:]+self.FTI_vel_Lb[4,:])/FgR,color='k')
        axs[1,2].plot(self.t,(self.FT_SRF[5,:]*self.M_scaling+self.FTI_acc_Lb[5,:]+self.FTI_vel_Lb[5,:])/FgR,color='k')

        fig.set_size_inches(10,30)
        plt.tight_layout()
        plt.show()




    def get_L_R_fnames_avg(self, file_name_list, period='baseline', naming_scheme='new'):
        for file_n in file_name_list:

            if naming_scheme=='old':

                if file_n.split('_')[1]=='uni':
                    if (file_n.split('_')[2]=='avg') and (file_n.split('_')[6]==period) and (file_n.split('_')[4]=='I'):
                        L_fname = file_n
                    if (file_n.split('_')[2]=='avg') and (file_n.split('_')[6]==period) and (file_n.split('_')[4]=='C'):
                        R_fname = file_n 
                
                else:
                    #changed naming scheme
                    if (file_n.split('_')[1]=='avg') and (file_n.split('_')[5]==period) and (file_n.split('_')[3]=='L'):
                        L_fname = file_n
                    if (file_n.split('_')[1]=='avg') and (file_n.split('_')[5]==period) and (file_n.split('_')[3]=='R'):
                        R_fname = file_n 

            else:
                if file_n.split('_')[1]=='unilateral':
                    if (file_n.split('_')[3]!='25') and (file_n.split('_')[3]==period) and (file_n.split('_')[2]=='I'):
                        L_fname = file_n
                    if (file_n.split('_')[3]!='25') and (file_n.split('_')[3]==period) and (file_n.split('_')[2]=='C'):
                        R_fname = file_n 

                else:
                #new naming scheme
                    if (file_n.split('_')[2]!='25') and (file_n.split('_')[2]==period) and (file_n.split('_')[1]=='L'):
                        L_fname = file_n
                    if (file_n.split('_')[2]!='25') and (file_n.split('_')[2]==period) and (file_n.split('_')[1]=='R'):
                        R_fname = file_n 

            

        return [L_fname, R_fname]

    def q_mult(self,qA,qB):
        QA = np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
            [qA[1],qA[0],-qA[3],qA[2]],
            [qA[2],qA[3],qA[0],-qA[1]],
            [qA[3],-qA[2],qA[1],qA[0]]])
        qC = np.dot(QA,qB)
        qC_norm = math.sqrt(pow(qC[0],2)+pow(qC[1],2)+pow(qC[2],2)+pow(qC[3],2))
        if qC_norm>0.01:
            qC /= qC_norm
        else:
            qC = np.array([1.0,0.0,0.0,0.0])
        return qC

    def comp_R(self,q):
        R = np.array([[2*pow(q[0],2)-1+2*pow(q[1],2), 2*q[1]*q[2]+2*q[0]*q[3], 2*q[1]*q[3]-2*q[0]*q[2]],
            [2*q[1]*q[2]-2*q[0]*q[3], 2*pow(q[0],2)-1+2*pow(q[2],2), 2*q[2]*q[3]+2*q[0]*q[1]],
            [2*q[1]*q[3]+2*q[0]*q[2], 2*q[2]*q[3]-2*q[0]*q[1], 2*pow(q[0],2)-1+2*pow(q[3],2)]])
        return R

    def quat_mat(self,s_in):
        q0 = np.squeeze(s_in[0])
        q1 = np.squeeze(s_in[1])
        q2 = np.squeeze(s_in[2])
        q3 = np.squeeze(s_in[3])
        tx = np.squeeze(s_in[4])
        ty = np.squeeze(s_in[5])
        tz = np.squeeze(s_in[6])
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        return M

    def set_Inertia_tensors(self,cg_L_in,M_L_in,cg_R_in,M_R_in):
        self.wing_L_cg = cg_L_in
        self.MwL = M_L_in
        self.wing_L_m = self.MwL[0,0]
        self.wing_L_I = self.MwL[3:,3:]
        self.wing_R_cg = cg_R_in
        self.MwR = M_R_in
        self.wing_R_m = self.MwR[0,0]
        self.wing_R_I = self.MwR[3:,3:]

    def dFT_du(self,seq_inds):

        m_mean = np.array([0.5,0.1,0.5,0.35,0.35,0.1,0.6,0.5,0.35,0.35,0.4,0.4,0.5])

        FT_A_L = np.zeros((6,7))
        FT_A_R = np.zeros((6,7))
        FT_I_acc_L = np.zeros((6,7))
        FT_I_vel_L = np.zeros((6,7))
        FT_I_acc_R = np.zeros((6,7))
        FT_I_vel_R = np.zeros((6,7))
        FT_L = np.zeros((6,7))
        FT_R = np.zeros((6,7))

        dFT_du = np.zeros((6,24))

        fig1, axs1 = plt.subplots(6,12)
        fig1.set_figwidth(18)
        fig1.set_figheight(9)
        for i in range(12):

            m_range_i = np.linspace(0,1-m_mean[i],num=7)

            for j in range(7):
                # Aerodynamic forces
                FT_m_L     = np.copy(self.FT_mean_list[seq_inds[j,i]])
                FT_m_R     = np.copy(self.FT_mean_list[seq_inds[j,i]])

                FT_m_R[1,:] = -FT_m_R[1,:]
                FT_m_R[3,:] = -FT_m_R[3,:]
                FT_m_R[5,:] = -FT_m_R[5,:]

                FT_A_L[:,j]     = FT_m_L[:,0]
                FT_A_R[:,j]     = FT_m_R[:,0]
                FT_I_acc_L[:,j] = FT_m_L[:,1]
                FT_I_acc_R[:,j] = FT_m_R[:,1]
                FT_I_vel_L[:,j] = FT_m_L[:,2]
                FT_I_vel_R[:,j] = FT_m_R[:,2]
                FT_L[:,j]         = FT_m_L[:,3]
                FT_R[:,j]         = FT_m_R[:,3]

            axs1[0,i].plot(m_range_i,FT_L[0,:]-FT_L[0,0],color='k')
            axs1[1,i].plot(m_range_i,FT_L[1,:]-FT_L[1,0],color='k')
            axs1[2,i].plot(m_range_i,FT_L[2,:]-FT_L[2,0],color='k')
            axs1[3,i].plot(m_range_i,FT_L[3,:]-FT_L[3,0],color='k')
            axs1[4,i].plot(m_range_i,FT_L[4,:]-FT_L[4,0],color='k')
            axs1[5,i].plot(m_range_i,FT_L[5,:]-FT_L[5,0],color='k')
            axs1[0,i].plot(m_range_i,FT_A_L[0,:]-FT_A_L[0,0],color='b')
            axs1[1,i].plot(m_range_i,FT_A_L[1,:]-FT_A_L[1,0],color='b')
            axs1[2,i].plot(m_range_i,FT_A_L[2,:]-FT_A_L[2,0],color='b')
            axs1[3,i].plot(m_range_i,FT_A_L[3,:]-FT_A_L[3,0],color='b')
            axs1[4,i].plot(m_range_i,FT_A_L[4,:]-FT_A_L[4,0],color='b')
            axs1[5,i].plot(m_range_i,FT_A_L[5,:]-FT_A_L[5,0],color='b')
            axs1[0,i].plot(m_range_i,FT_I_vel_L[0,:]-FT_I_vel_L[0,0],color='r')
            axs1[1,i].plot(m_range_i,FT_I_vel_L[1,:]-FT_I_vel_L[1,0],color='r')
            axs1[2,i].plot(m_range_i,FT_I_vel_L[2,:]-FT_I_vel_L[2,0],color='r')
            axs1[3,i].plot(m_range_i,FT_I_vel_L[3,:]-FT_I_vel_L[3,0],color='r')
            axs1[4,i].plot(m_range_i,FT_I_vel_L[4,:]-FT_I_vel_L[4,0],color='r')
            axs1[5,i].plot(m_range_i,FT_I_vel_L[5,:]-FT_I_vel_L[5,0],color='r')
            axs1[0,i].plot(m_range_i,FT_I_acc_L[0,:]-FT_I_acc_L[0,0],color='g')
            axs1[1,i].plot(m_range_i,FT_I_acc_L[1,:]-FT_I_acc_L[1,0],color='g')
            axs1[2,i].plot(m_range_i,FT_I_acc_L[2,:]-FT_I_acc_L[2,0],color='g')
            axs1[3,i].plot(m_range_i,FT_I_acc_L[3,:]-FT_I_acc_L[3,0],color='g')
            axs1[4,i].plot(m_range_i,FT_I_acc_L[4,:]-FT_I_acc_L[4,0],color='g')
            axs1[5,i].plot(m_range_i,FT_I_acc_L[5,:]-FT_I_acc_L[5,0],color='g')


            axs1[0,i].set_ylim([-0.005,0.005])
            axs1[1,i].set_ylim([-0.005,0.005])
            axs1[2,i].set_ylim([-0.005,0.005])
            axs1[3,i].set_ylim([-0.005,0.005])
            axs1[4,i].set_ylim([-0.005,0.005])
            axs1[5,i].set_ylim([-0.005,0.005])


            # dFT_du
            dFT_du[0,i]     = (FT_L[0,6]-FT_L[0,0])/m_range_i[6]
            dFT_du[1,i]     = (FT_L[1,6]-FT_L[1,0])/m_range_i[6]
            dFT_du[2,i]     = (FT_L[2,6]-FT_L[2,0])/m_range_i[6]
            dFT_du[3,i]     = (FT_L[3,6]-FT_L[3,0])/m_range_i[6]
            dFT_du[4,i]     = (FT_L[4,6]-FT_L[4,0])/m_range_i[6]
            dFT_du[5,i]     = (FT_L[5,6]-FT_L[5,0])/m_range_i[6]
            dFT_du[0,i+12]     = (FT_R[0,6]-FT_R[0,0])/m_range_i[6]
            dFT_du[1,i+12]     = (FT_R[1,6]-FT_R[1,0])/m_range_i[6]
            dFT_du[2,i+12]     = (FT_R[2,6]-FT_R[2,0])/m_range_i[6]
            dFT_du[3,i+12]     = (FT_R[3,6]-FT_R[3,0])/m_range_i[6]
            dFT_du[4,i+12]     = (FT_R[4,6]-FT_R[4,0])/m_range_i[6]
            dFT_du[5,i+12]     = (FT_R[5,6]-FT_R[5,0])/m_range_i[6]

        return dFT_du

    def make_lollipop_figure(self,seq_inds,exp_name,save_loc,m_clr,ind_0):
        #save_loc.mkdir(exist_ok=True, parents=True)
        for i,test_ind in enumerate(seq_inds):
            print(test_ind)
            LP = Lollipop()
            LP.Renderer()
            LP.ConstructModel(True)
            s_thorax  = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0])
            s_head       = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.55,0.0,0.42])
            s_abdomen = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1])
            body_scale = [0.80,0.85,0.90]
            body_clr = [(0.7,0.7,0.7)]
            LP.SetBodyColor(body_clr)
            LP.SetBodyScale(body_scale)
            LP.SetBodyState(s_thorax,s_head,s_abdomen)
            n_pts = 100
            wing_length = 2.0
            joint_L = np.array([0.0,0.5,0.0])
            joint_R = np.array([0.0,-0.5,0.0])
            LE_pt = 0.1
            TE_pt = -0.2

            theta_L = self.wingkin_SRF_list[test_ind][0,:]
            eta_L     = self.wingkin_SRF_list[test_ind][1,:]
            phi_L     = self.wingkin_SRF_list[test_ind][2,:]
            xi_L     = self.wingkin_SRF_list[test_ind][3,:]
            theta_R = self.wingkin_SRF_list[test_ind][0,:]
            eta_R     = self.wingkin_SRF_list[test_ind][1,:]
            phi_R     = self.wingkin_SRF_list[test_ind][2,:]
            xi_R     = self.wingkin_SRF_list[test_ind][3,:]
            n_pts     = xi_R.shape[0]

            #if include forces in this way only including contributions of aerodynamic forces 
            FX_L     = self.FT_total_wing_list[test_ind][0,:]
            FY_L     = self.FT_total_wing_list[test_ind][1,:]
            FZ_L     = self.FT_total_wing_list[test_ind][2,:]
            FX_R     = self.FT_total_wing_list[test_ind][0,:]
            FY_R     = self.FT_total_wing_list[test_ind][1,:]
            FZ_R     = self.FT_total_wing_list[test_ind][2,:]


            FX_mean = 0.0
            FY_mean = 0.0
            FZ_mean = 0.0
            MX_mean = 0.0
            MY_mean = 0.0
            MZ_mean = 0.0
            FX_0     = 0.0
            FY_0     = 0.0
            FZ_0     = 0.0
            MX_0     = 0.0
            MY_0     = 0.0
            MZ_0     = 0.0

            LP.set_srf_angle(self.srf_angle)
            LP.set_wing_motion_direct(theta_L,eta_L,phi_L,xi_L,theta_R,eta_R,phi_R,xi_R,n_pts)
            LP.set_forces_direct(FX_L,FY_L,FZ_L,FX_R,FY_R,FZ_R)
            LP.set_mean_forces(FX_mean,FY_mean,FZ_mean,MX_mean,MY_mean,MZ_mean)
            LP.set_FT_0(FX_0,FY_0,FZ_0,MX_0,MY_0,MZ_0)
            Fg = np.array([0.0,0.0,-FZ_0])
            LP.set_Fg(Fg)
            FD = np.array([-FX_0,0.0,0.0])
            LP.set_FD(FD)
            LP.compute_tip_forces(wing_length,joint_L,joint_R,LE_pt,TE_pt,m_clr,m_clr,0,0.0)
            img_width = 1000
            img_height = 800
            p_scale = 2.5
            clip_range = [0,16]
            cam_pos = [12,0,0]
            view_up = [0,0,1]
            test_name = exp_name+str(i)
            img_name = test_name+'_front.jpg'
            LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)
            cam_pos = [0,12,0]
            view_up = [0,0,1]
            img_name = test_name+'_side.jpg'
            LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)
            cam_pos = [0,0,12]
            view_up = [1,0,0]
            img_name = test_name+'_top.jpg'
            LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)
            clip_range = [0,16]
            cam_pos = [-12,0,0]
            view_up = [0,0,1]
            img_name = test_name+'_back.jpg'
            LP.take_image(img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,img_name)

    #confirmed: slow interpolated to fast matches up well to fast wb for gravity subtraction
    def align_kinematics_slow_fast_for_grav_sub_viz(self, save_dir, side='L'):
        """
        apply same interpolation to wingkinematics as forces/torques for alignment and grav sub viz
        """
        if side=='L':
            ind = 0
        else:
            ind = 1
        #left 1 theta, 2 eta, 3 phi
        t_f = self.wingkin_f_list[ind][:,0]
        t_s = self.wingkin_s_list[ind][:,0]

        phi_f = self.wingkin_f_list[ind][:,3]
        theta_f = self.wingkin_f_list[ind][:,1]
        eta_f = self.wingkin_f_list[ind][:,2]
        phi_s = self.wingkin_s_list[ind][:,3]
        theta_s = self.wingkin_s_list[ind][:,1]
        eta_s = self.wingkin_s_list[ind][:,2]

        N_wb=7
        t_FT_f = N_wb*(t_f/max(t_f))
        t_FT_s = N_wb*(t_s/max(t_s))

        phi_G_interp = interpolate.interp1d(t_FT_s, phi_s)
        phi_G = phi_G_interp(t_FT_f)
        theta_G_interp = interpolate.interp1d(t_FT_s, theta_s)
        theta_G = theta_G_interp(t_FT_f)
        eta_G_interp = interpolate.interp1d(t_FT_s, eta_s)
        eta_G = eta_G_interp(t_FT_f)

        phi_G_sub = phi_f - phi_G
        theta_G_sub = theta_f - theta_G
        eta_G_sub = eta_f - eta_G

        #for 3 angles
        fig, ax = plt.subplots(3,1, sharex=True)
        ax[0].plot(t_FT_f,phi_f, label='fast not grav sub', color='green', alpha=0.5)
        ax[0].plot(t_FT_f,phi_G, label='slow interp, grav', color='red', alpha=0.5)
        ax[0].plot(t_FT_f,phi_G_sub, label='grav sub', color='blue', alpha=0.5)

        ax[1].plot(t_FT_f,theta_f, color='green', alpha=0.5)
        ax[1].plot(t_FT_f,theta_G, color='red', alpha=0.5)
        ax[1].plot(t_FT_f,theta_G_sub, color='blue', alpha=0.5)

        ax[2].plot(t_FT_f,eta_f, color='green', alpha=0.5)
        ax[2].plot(t_FT_f,eta_G, color='red', alpha=0.5)
        ax[2].plot(t_FT_f,eta_G_sub, color='blue', alpha=0.5)
        
        ax[0].legend(loc='lower left')
        ax[0].set_ylabel(r'$phi$')
        ax[1].set_ylabel(r'$theta$')
        ax[2].set_ylabel(r'$eta$')
        fig.set_size_inches(20,15)
        plt.savefig(save_dir + 'traces_grav_subtracted')

    def to_deg(self,x):
        return x*(180.0/np.pi) 
    
    def LegendrePolynomials(self, N_pts,N_pol,n_deriv):
        """
        N_const: degree stitch together subsequent wingbeats
        N_deriv: number of derivatives to computer for legendre basis
        """
        L_basis = np.zeros((N_pts,N_pol,n_deriv))
        x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
        for i in range(n_deriv):
            if i==0:
                # Legendre basis:
                for n in range(N_pol):
                    if n==0:
                        L_basis[:,n,i] = 1.0
                    elif n==1:
                        L_basis[:,n,i] = x_basis
                    else:
                        for k in range(n+1):
                            L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
            else:
                # Derivatives:
                for n in range(N_pol):
                    if n>=i:
                        L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
        return L_basis
    
    def reconstruct_trace_coef_matrix(self, cur_coeffs, deriv_num=0, n_deriv=2, N_pts=100):
        """
        return trace for vector for a coefficients (often avg coeff vector)
        if reading in multiple wbs need to transpose to coef by # wbs
        """
        L_basis = self.LegendrePolynomials(N_pts, len(cur_coeffs), n_deriv)
        return np.matmul(L_basis[:,:,deriv_num], cur_coeffs)

    #produces same results as new SRF conversion, same slight temporal offset
    def convert_to_SRF_original(self,beta,phi_shift):
        
        # #previous
        # self.wingkin_SRF = np.zeros((4,self.FT_out.shape[1]))
        # self.FT_SRF = np.zeros(self.FT_out.shape)
        #to account for new scaling 
        self.wingkin_SRF = np.zeros((4,self.FT_wing.shape[1]))
        self.FT_SRF = np.zeros(self.FT_wing.shape)
        # for i in range(self.N_FT_fast):
        for i in range(np.shape(self.FT_wing)[1]):
            # phi = np.pi*(self.wingkin_f[i,3]/180.0)-phi_shift
            # theta = np.pi*(self.wingkin_f[i,1]/180.0)
            # xi = 3.0*np.pi*(self.wingkin_f[i,5]/180.0)
            # eta = np.pi*(self.wingkin_f[i,2]/180.0)
            phi = -np.pi*(self.wingkin_f[i,3]/180.0)-phi_shift
            theta = np.pi*(self.wingkin_f[i,1]/180.0)
            xi = -3.0*np.pi*(self.wingkin_f[i,5]/180.0)
            #likely want to subtract 1/3 xi that was added on, check 
            #eta = -np.pi*(self.wingkin_f[i,2]/180.0) #current 
            #ae testing 
            eta = -np.pi*(self.wingkin_f[i,2]/180.0) - xi/3 #testing
            #a_srf = -np.pi*(55.0/180.0)+beta
            a_srf = -np.pi*(75.0/180.0)+beta #might need to change; rotates strokeplane should be around 90 deg (if forces backward or forward can rotate to get a more accurate force/body weight) 
            # a_srf = -np.pi*(100.0/180.0)+beta 
            #a_srf = beta
            R_beta  = np.array([[np.cos(a_srf),0.0,np.sin(a_srf)],[0.0,1.0,0.0],[-np.sin(a_srf),0.0,np.cos(a_srf)]])
            R_phi   = np.array([[1.0,0.0,0.0],[0.0,np.cos(phi),-np.sin(phi)],[0.0,np.sin(phi),np.cos(phi)]])
            R_theta = np.array([[np.cos(-theta),-np.sin(-theta),0.0],[np.sin(-theta),np.cos(-theta),0.0],[0.0,0.0,1.0]])
            R_eta   = np.array([[np.cos(eta),0.0,np.sin(eta)],[0.0,1.0,0.0],[-np.sin(eta),0.0,np.cos(eta)]])
            R_total = np.dot(np.dot(np.dot(R_beta,R_phi),R_theta),R_eta)
            #R_total = np.dot(R_beta,np.dot(R_phi,np.dot(R_theta,R_eta)))
            R_mat = np.zeros((6,6))
            R_mat[:3,:3] = R_total
            R_mat[3:,3:] = R_total
            FT_i = self.FT_wing[:,i]
            self.FT_SRF[:,i] = np.dot(R_mat,FT_i)

            #force sensor undergoes changes in phi, theta, eta, etc so need to "undo" the rotations to get the forces on the wing (not rotating)

            #wing kinematics remain the same 

            self.wingkin_SRF[0,i] = theta
            self.wingkin_SRF[1,i] = eta
            self.wingkin_SRF[2,i] = phi
            self.wingkin_SRF[3,i] = xi

    def get_L_R_fnames_avg_bl_stim(self, file_name_list, naming_scheme='new'):
        """
        return list of L and R filenames (if sparc maps side ipsilateral to activation to Left wing)
        """
        for file_n in file_name_list:

            if naming_scheme=='old':
                #sparc "uni" second part of name 
                #map all Ipsilateral to left wing 
                if file_n.split('_')[1]=='uni':
                    if (file_n.split('_')[2]=='avg') and (file_n.split('_')[6]=='baseline') and (file_n.split('_')[4]=='I'):
                        L_fname_baseline = file_n
                    if (file_n.split('_')[2]=='avg') and (file_n.split('_')[6]=='baseline') and (file_n.split('_')[4]=='C'):
                        R_fname_baseline = file_n 

                    if (file_n.split('_')[2]=='avg') and (file_n.split('_')[6]=='stim') and (file_n.split('_')[4]=='I'):
                        L_fname_stim = file_n
                    if (file_n.split('_')[2]=='avg') and (file_n.split('_')[6]=='stim') and (file_n.split('_')[4]=='C'):
                        R_fname_stim = file_n 

                else:
                #new naming scheme
                    if (file_n.split('_')[1]=='avg') and (file_n.split('_')[5]=='baseline') and (file_n.split('_')[3]=='L'):
                        L_fname_baseline = file_n
                    if (file_n.split('_')[1]=='avg') and (file_n.split('_')[5]=='baseline') and (file_n.split('_')[3]=='R'):
                        R_fname_baseline = file_n 

                    if (file_n.split('_')[1]=='avg') and (file_n.split('_')[5]=='stim') and (file_n.split('_')[3]=='L'):
                        L_fname_stim = file_n
                    if (file_n.split('_')[1]=='avg') and (file_n.split('_')[5]=='stim') and (file_n.split('_')[3]=='R'):
                        R_fname_stim = file_n 

            else:
                if file_n.split('_')[1]=='unilateral':
                    if (file_n.split('_')[3]!='25') and (file_n.split('_')[3]=='baseline') and (file_n.split('_')[2]=='I'):
                        L_fname_baseline = file_n
                    if (file_n.split('_')[3]!='25') and (file_n.split('_')[3]=='baseline') and (file_n.split('_')[2]=='C'):
                        R_fname_baseline = file_n 

                    if (file_n.split('_')[3]!='25') and (file_n.split('_')[3]=='stim') and (file_n.split('_')[2]=='I'):
                        L_fname_stim = file_n
                    if (file_n.split('_')[3]!='25') and (file_n.split('_')[3]=='stim') and (file_n.split('_')[2]=='C'):
                        R_fname_stim = file_n 

                else:
                #new naming scheme
                    if (file_n.split('_')[2]!='25') and (file_n.split('_')[2]=='baseline') and (file_n.split('_')[1]=='L'):
                        L_fname_baseline = file_n
                    if (file_n.split('_')[2]!='25') and (file_n.split('_')[2]=='baseline') and (file_n.split('_')[1]=='R'):
                        R_fname_baseline = file_n 

                    if (file_n.split('_')[2]!='25') and (file_n.split('_')[2]=='stim') and (file_n.split('_')[1]=='L'):
                        L_fname_stim = file_n
                    if (file_n.split('_')[2]!='25') and (file_n.split('_')[2]=='stim') and (file_n.split('_')[1]=='R'):
                        R_fname_stim = file_n


        return [L_fname_baseline, R_fname_baseline, L_fname_stim, R_fname_stim]

    def plot_flyami_wingkin_robofly_input_overlay(self, data, save_loc, save_fig=True, side='L', time_period='baseline', original_SRF=False):
        """
        plot wing kinematics from flyami and robofly inputs overlayed for left and right wing baseline and activation 
        plot wing kinematics using original wingbeats from flyami (not reconstructed traces! or shifted angles)
        time_period: plotting baseline or activation wingbeat comparison (for reading in flyami wing traces)

        data: read in hdf5 file 
        """

        data_movie = data

        # concat, mean and plot 
        angle_dict = {'phi': 0, 'theta': 1, 'eta': 2, 'xi': 3}
        angles = ['phi', 'theta', 'eta', 'xi']
        forces_dict = {'Fx': 4, 'Fy': 5, 'Fz': 6, 'Mx': 7, 'My': 8, 'Mz': 9}
        forces = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        wingkin_SRF_dict = {'phi': 2, 'theta': 0, 'eta': 1, 'xi': 3}
        force_SRF_dict = {'Fx': 0, 'Fy': 1, 'Fz': 2, 'Mx': 3, 'My': 4, 'Mz': 5}
        side_dict = {0: 'L', 1: 'R', 2: 'L', 3: 'R'}
        wing_sides = ['L', 'R']

        euler_angles_dict = {'phi': r'$\phi$', 'theta': r'$\theta$', 'eta': r'$\eta$', 'xi': r'$\xi$'}

        rows = len(angles) 
        cols = 1 # left and right wings
        fig, axs = plt.subplots(rows, cols, constrained_layout=True)


        # first value with be L, and second with be right wing
        if side=='L':
            ind = 0
        else:
            ind = 1

        # in strokeplane 
        if original_SRF==True:
            wb_select = ((self.T_fast>=4.0)&(self.T_fast<5.0)) #just take wb 4 for now
            angle_wb = self.wingkin_SRF_list[ind][:,wb_select]
    
        else:
            angle_wb = self.wingkin_SRF_list[ind][:]
        
        #self.wingkin_SRF_wb_5 = self.wingkin_SRF_list[ind][:]
        #self.wingkin_SRF_wb_6 = self.wingkin_SRF_list[ind][:]
        

        for angle in angles:
            angle_mean_wb = angle_wb[wingkin_SRF_dict[angle],:]
            #angle_wingkin_5 = self.wingkin_SRF_wb_5[wingkin_SRF_dict[angle],:]
            #angle_wingkin_6 = self.wingkin_SRF_wb_6[wingkin_SRF_dict[angle],:]
            # angle_wingkin = np.vstack((angle_wingkin_5, angle_wingkin_6))
            # angle_mean_wb = np.mean(angle_wingkin, axis=0)
            relative_t = np.linspace(0,1,len(angle_mean_wb))

            axs[angle_dict[angle]].plot(relative_t, self.to_deg(angle_mean_wb), color='#1090e6', label='robofly wingkin_SRF')

            if angle=='eta':
                print('eta min robofly: ' + str(np.min(self.to_deg(angle_mean_wb))))
        
            axs[3].set_ylim(-60, 60)
            axs[3].set_yticks([-60, 0, 60])
        
            axs[0].set_ylim(-80, 100)
            axs[0].set_yticks([-80, 0, 100])
        
            axs[1].set_ylim(-30, 60) 
            axs[1].set_yticks([-30, 0, 60])
        
            axs[2].set_ylim(-140, 80)
            axs[2].set_yticks([-140, 0, 80])

    
        # plot wingkin, adapted from flyami_plotting combined flies
            
        # get data for all angles
        for row, euler_angle in enumerate(angles):

            #all flies 
            pre_data_all_flies = []
            stim_data_all_flies = []
            pre_data_freq_all_flies = []
            stim_data_freq_all_flies = []


            for fly_n, fly in enumerate(data_movie.keys()): 

                movie_list = data_movie[fly].keys()

                #per fly 
                pre_data_fly = []
                stim_data_fly = []
                pre_data_freq_fly = []
                stim_data_freq_fly = []

                
                for movie_n, movie in enumerate(movie_list):

                    # precoef
                    pre_data = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/a/' + 'pre'][:]
                    pre_frame_starts = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/wb/' + 'wb_frame_start']['pre'][:]
                    pre_frame_ends = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/wb/' + 'wb_frame_end']['pre'][:]

                    #stim coef
                    stim_data = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/a/' + 'stim'][:]
                    stim_frame_starts = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/wb/' + 'wb_frame_start']['stim'][:]
                    stim_frame_ends = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/wb/' + 'wb_frame_end']['stim'][:]

                    # post ceof 
                    post_data = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/a/' + 'post'][:]
                    post_frame_starts = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/wb/' + 'wb_frame_start']['post'][:]
                    post_frame_ends = data_movie[fly + '/' + movie + '/angle/' + euler_angle + '/' + side + '/wb/' + 'wb_frame_end']['post'][:]

                    # get pulse start and pulse end 
                    pulse_start = data_movie[fly + '/' + movie + '/time_stamps/' + 'pulse_start'][:][0]
                    pulse_end = data_movie[fly + '/' + movie + '/time_stamps/' + 'pulse_end'][:][0]

                    

                    # check if any arrays are empty, and dont include that trial if so

                    if (len(pre_data)==0) or (len(stim_data)==0) or (len(post_data)==0):
                        continue

                    else:
                        # add to respective time frame, take mean for each fly, then mean across flies
                        # add to per fly arr
                        pre_data_fly.extend(pre_data)
                        stim_data_fly.extend(stim_data)
                        
                #avg mean of all movies/fly 
                pre_data_all_flies.append(np.mean(pre_data_fly, axis=0))
                stim_data_all_flies.append(np.mean(stim_data_fly, axis=0))
                if euler_angle=='xi':
                    pre_data_freq_all_flies.append(np.mean(pre_data_freq_fly))
                    stim_data_freq_all_flies.append(np.mean(stim_data_freq_fly))

                
            #now take mean of fly means for each angle 
            baseline_coef = np.mean(pre_data_all_flies, axis=0)
            activation_coef = np.mean(stim_data_all_flies, axis=0)

            baseline_avg = self.reconstruct_trace_coef_matrix(baseline_coef, N_pts=100)
            activation_avg = self.reconstruct_trace_coef_matrix(activation_coef, N_pts=100)

            time = np.linspace(0,1,len(baseline_avg)) # timescale in units of wingbeat
            
           
            if time_period=='baseline':
                axs[row].plot(time, self.to_deg(baseline_avg), color='black', linestyle='-', linewidth=1, label='flyami')
                
            if time_period=='activation':
                 axs[row].plot(time, self.to_deg(activation_avg), color='black', linestyle='-', linewidth=1, label='flyami')

            if euler_angle=='eta':
                print('eta min flyami: ' + str(np.min(self.to_deg(baseline_avg))))

            axs[3].set_ylim(-60, 60)
            axs[3].set_yticks([-60, 0, 60])
        
            axs[0].set_ylim(-80, 100)
            axs[0].set_yticks([-80, 0, 100])
        
            axs[1].set_ylim(-30, 60) 
            axs[1].set_yticks([-30, 0, 60])
        
            axs[2].set_ylim(-140, 80)
            axs[2].set_yticks([-140, 0, 80])
            
            axs[0].legend(loc='lower left')
                
    
        
        
        fig = plt.gcf()
        fig.set_size_inches(5,10)
        fig.align_ylabels(axs[:])
    
        plt.savefig(save_loc + 'compare_kinematics_flyami_robofly_reconstructed.pdf', dpi=200)

# if __name__ == "__main__":
#     RA = RoboAnalysis()
#     data_fpath = '/Users/anneerickson/Documents/Caltech/Dickinson/Flyami_analysis/prelim_geno_plots/NDF_removed_0.5s_combined/with_timestamps/1070_individual_flies_wb_a.hdf5'
#     data = h5py.File(data_fpath, 'r')
#     RA.plot_flyami_wingkin_robofly_input_overlay(data, None, save_fig=False, side='L', time_period='baseline')


#TODO:right wing convert to SRF, 
#load in stim and baseline files, 
#per trial n_fly and an_robo, 
#check gain factor need to be added in?