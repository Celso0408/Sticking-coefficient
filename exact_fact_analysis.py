import numpy as np
from numpy.core.records import record
import pandas as pd
import csv

import matplotlib.pyplot as plt
from plt_dynamic import *
from matplotlib.lines import Line2D
import matplotlib.animation as animation


from scipy import integrate
from scipy.integrate import simps, quad
from scipy.interpolate import CubicSpline


def read_input(file_name='Results/input_NRG.dat'):
    global M, N, leng_z, nuc_mass, dz, Tot_state
    global Re_Beta, Im_Beta, Beta, BOPE, dt, NACT1, NACT2
    global n_time_steps, time_step, omega, recorded_step, init_k0

    Im_Beta = pd.read_csv('Results/Im_Beta.dat', dtype = float, sep = '\s+', header = None)
    Re_Beta = pd.read_csv('Results/Re_Beta.dat', dtype = float, sep = '\s+', header = None)
    NACT1 = pd.read_csv('Results/NACT1.dat', dtype = float, sep = '\s+', header = None)
    NACT2 = pd.read_csv('Results/NACT2.dat', dtype = float, sep = '\s+', header = None)

    BOPE = pd.read_csv('Results/Energy_BO.dat', dtype = float, sep = '\s+', header = None)
    Im_Beta = Im_Beta.to_numpy()
    Re_Beta = Re_Beta.to_numpy()
    NACT1 = NACT1.to_numpy()
    NACT2 = NACT2.to_numpy()

    Beta = Re_Beta**2.0 + Im_Beta**2.0
    omega = 0.001
    #print("omega = ", omega)

    n_time_steps = Beta.shape # number of steps 
    
    df = pd.read_csv(file_name, header=None)
    M = int(df.loc[1][0]) # number of grid points
    
    # Ground state BOPE 
    BOPE = BOPE.to_numpy()
    BOPE[:,0] = BOPE[:,0] - BOPE[M-2,0]
    BOPE = list(BOPE[:,0])
    #BOPE.append(BOPE[M-2])
    
    N = int(df.loc[2][0]) # number of wilson chain
    leng_z = float(df.loc[3][0]) # size box
    init_k0 = float(df.loc[6][0]) # initial moment
    print('init_k0 =', init_k0)
    dz = leng_z/M
    nuc_mass = float(df.loc[7][0]) # nucclear mass
    time_step = float(df.loc[9][0]) # time step in a.u
    recorded_step = 1000
    dt = round(recorded_step*time_step,4)  # time step
    Tot_state = int(df.loc[13][0]) # Total number of electronic states
    ...

def z_vec():
    z_vec = np.linspace(0, leng_z, M)
    return z_vec


def diff_manybody_vec(manybody_vec):
    
    ''' Finite Difference, O(h^2), devirative for a manybody vector '''
    
    diff_mat = np.zeros(n_time_steps)   

    for ii in range(n_time_steps[0]):
        temp_vec = []
        temp_vec = manybody_vec[ii][:]

        for jj in range(Tot_state):
            temp_vec_jz = []
            temp_diff_jz = []  

            for kk in range(jj, n_time_steps[1], Tot_state):               
                temp_vec_jz.append(temp_vec[kk])          
            
            temp_diff_jz = np.gradient(temp_vec_jz, dz, edge_order = 2)
            i = 0
            for kk in range(jj, n_time_steps[1], Tot_state):               
                diff_mat[ii][kk] = temp_diff_jz[i]
                i += 1

    return diff_mat 

def density():

    den_mat = np.zeros((n_time_steps[0], M))
    #z_cood = z_vec()
    for ii in range(n_time_steps[0]):
        Beta1=[]
        for jz in range(M):
            Beta1.append(np.sum(Beta[ii][Tot_state*jz:Tot_state*(jz+1)]))
        den_mat[ii][:] = Beta1[:]
        #print(integrate.simps(den_mat[ii][:], z_cood, dz))
    with open('density.dat', 'w') as f: 
        write = csv.writer(f) 
        write.writerows(den_mat)
    
    return den_mat

def nuclear_phase():

    z_vec = np.linspace(0, leng_z, M)
    den_mat = np.zeros((n_time_steps[0], M))
    den_mat = density()

    S_mat = np.zeros((n_time_steps[0], M))
    dsdz_NACT_d1 = np.zeros((n_time_steps[0], M))

    diff_Re_Beta = diff_manybody_vec(Re_Beta)
    diff_Im_Beta = diff_manybody_vec(Im_Beta)

    for ii in range(n_time_steps[0]):
        
        # NACT d_1 contribution
        temp_dsdz = np.zeros(M)
        for jz in range(M):
            temp_NACT1 = np.zeros([Tot_state, Tot_state])
            temp_NACT1 = NACT1[Tot_state*jz:Tot_state*(jz+1)][:]
            
            temp_Re_jz = np.zeros(Tot_state)
            temp_Im_jz = np.zeros(Tot_state)                       
            
            temp_Re_jz = Re_Beta[ii][Tot_state*jz:Tot_state*(jz+1)]   
            temp_Im_jz = Im_Beta[ii][Tot_state*jz:Tot_state*(jz+1)]
            
            temp_d1 = 0.0
            for ll in range(Tot_state):
                for kk in range(Tot_state):
                    temp_d1 = temp_d1 + (temp_Re_jz[ll]*temp_Im_jz[kk] - temp_Im_jz[ll]*temp_Re_jz[kk])*temp_NACT1[ll][kk]  
        
            dsdz_NACT_d1[ii][jz] = temp_d1
             
            temp_diff_Re_jz = np.zeros(Tot_state)
            temp_diff_Im_jz = np.zeros(Tot_state)                       
            
            temp_diff_Re_jz = np.array(diff_Re_Beta[ii][Tot_state*jz:Tot_state*(jz+1)])   
            temp_diff_Im_jz = np.array(diff_Im_Beta[ii][Tot_state*jz:Tot_state*(jz+1)])
        
            temp_dsdz[jz] = (np.inner(temp_Re_jz, temp_diff_Im_jz) - np.inner(temp_Im_jz,temp_diff_Re_jz) + dsdz_NACT_d1[ii][jz])/( den_mat[ii,jz] + omega )
            #temp_dsdz[jz] = ( np.inner(temp_Re_jz, temp_diff_Im_jz) - np.inner(temp_Im_jz,temp_diff_Re_jz) )/( den_mat[ii][jz] + omega )
        cs = CubicSpline(z_vec, temp_dsdz)

        for jz in range(M):
            S_mat[ii][jz] = cs.integrate(z_vec[0], z_vec[jz])
            #S_mat[ii][jz] = np.trapz(temp_dsdz[0:jz], z_vec[0:jz], dz)

        S_mat[ii][:] = S_mat[ii][:] - S_mat[ii][M-1]
        #print(np.max(S_mat[ii][:]), np.min(S_mat[ii][:]))
        
    return S_mat


def chi_z():
    den_mat = density()
    S_mat = np.zeros((n_time_steps[0], M))
    S_mat = nuclear_phase()

    Re_chi_z = np.zeros((n_time_steps[0], M))
    Im_chi_z = np.zeros((n_time_steps[0], M))
    
    for ii in range(n_time_steps[0]):
        Re_chi_z[ii][:] = np.sqrt(den_mat[ii][:])*np.cos(S_mat[ii][:])
        Im_chi_z[ii][:] = np.sqrt(den_mat[ii][:])*np.sin(S_mat[ii][:])
    
    return Re_chi_z, Im_chi_z

def continuity_eq():
    Cont_eq_mat = np.zeros((n_time_steps[0], M))
    den_mat = density()
    S_mat = nuclear_phase()
    z_cood = z_vec()

    var = 0
    
    for ii in range(n_time_steps[0]):
        dsdz = np.gradient(S_mat[ii][:], dz, edge_order = 2)
        Diff_J = np.gradient( (den_mat[ii][:]*dsdz)/nuc_mass, dz, edge_order = 2)
        dt_den = np.gradient(den_mat[ii][:], dt, edge_order = 2)
        Cont_eq_mat[ii][:] = Diff_J + dt_den
        #var = var + np.sum(Cont_eq_mat[ii][:]*Cont_eq_mat[ii][:])
        var = var + integrate.simps(Cont_eq_mat[ii][:]*Cont_eq_mat[ii][:], z_cood, dz)
    print("error = ", var)
    return Cont_eq_mat

def tdpes():

    tdpes_mat = np.zeros((n_time_steps[0], M))
    den_mat = density()
    S_mat = nuclear_phase()

    tdpes_mat = np.zeros((n_time_steps[0], M))
    
    for ii in range(n_time_steps[0]):
        dt_tdpes = -np.gradient(S_mat[ii][:], dt, edge_order=2)
        dsdz = np.gradient(S_mat[ii][:], dz, edge_order=2)
        dchidz =  np.gradient(np.sqrt(den_mat[ii][:]), dz, edge_order=2)
        d2chidz =  np.gradient(dchidz, dz, edge_order=2)
        tdpes_mat[ii][:] = dt_tdpes[:] + ( d2chidz/( np.sqrt(den_mat[ii][:]) + omega) - dsdz[:]**2.0 )/(2.0*nuc_mass)
    
    with open('TDPES.dat', 'w') as f: 
        write = csv.writer(f) 
        write.writerows(tdpes_mat)
    
    return tdpes_mat

def phi_z():
    
    den_mat = np.zeros((n_time_steps[0], M))
    den_mat = density()

    Re_chi_z = np.zeros((n_time_steps[0], M))
    Im_chi_z = np.zeros((n_time_steps[0], M))
    Re_chi_z, Im_chi_z = chi_z()

    Re_phi_z = np.zeros(n_time_steps)   
    Im_phi_z = np.zeros(n_time_steps)

    for ii in range(n_time_steps[0]):
        
        # |Re(psi)> and |Im(psi)|> 
        
        temp_Re_jz = []
        temp_Im_jz = []

        i = 0
        for kk in range(0, n_time_steps[1], Tot_state):
    
            temp_Re_jz = Re_Beta[ii][kk:kk+Tot_state]
            temp_Im_jz = Im_Beta[ii][kk:kk+Tot_state]
            
            Re_phi_z[ii][kk:kk+Tot_state] = ( Re_chi_z[ii][i]*np.array(temp_Re_jz[:]) + Im_chi_z[ii][i]*np.array(temp_Im_jz[:]) )/( den_mat[ii][i] )
            Im_phi_z[ii][kk:kk+Tot_state] = ( Re_chi_z[ii][i]*np.array(temp_Im_jz[:]) - Im_chi_z[ii][i]*np.array(temp_Re_jz[:]) )/( den_mat[ii][i] )

            # Uncomment the below two lines to checking PNC
            #x = np.sum(Re_phi_z[ii][kk:kk+Tot_state]**2.0 + Im_phi_z[ii][kk:kk+Tot_state]**2.0)
            #print(x)        

            i += 1
    return Re_phi_z, Im_phi_z

def all_kinetic_energy():

    exact_data = np.zeros([n_time_steps[0],5])
    data_time = []

    for ii in range(n_time_steps[0]):
        data_time.append(round(ii*recorded_step*time_step/41,3))
    exact_data[:,0] = data_time
    

    z_cood = z_vec()
    
    # Nuclear 
    #d_geo_z = np.zeros((n_time_steps[0], M))
    Re_chi_z = np.zeros((n_time_steps[0], M))
    Im_chi_z = np.zeros((n_time_steps[0], M))
     
    Re_chi_z, Im_chi_z = chi_z()
    
    Tn = []

    # Tn,marginal
    for ii in range(n_time_steps[0]):    
        int_var = 0.0
        Re_dchi_dz = []
        Im_dchi_dz = []
        Re_d2chi_dz2 = []
        Im_d2chi_dz2 = []
        Re_int = []
        Im_int = []

        Re_dchi_dz = np.gradient(Re_chi_z[ii][:], dz, edge_order=2)
        Im_dchi_dz = np.gradient(Im_chi_z[ii][:], dz, edge_order=2)
        Re_d2chi_dz2 = np.gradient(Re_dchi_dz, dz, edge_order=2)
        Im_d2chi_dz2 = np.gradient(Im_dchi_dz, dz, edge_order=2)

        Re_int = list(np.array(Re_chi_z[ii][:])*np.array(Re_d2chi_dz2))
        Im_int = list(np.array(Im_chi_z[ii][:])*np.array(Im_d2chi_dz2))

        int_var = (integrate.simps(Re_int,z_cood, dz) + integrate.simps(Im_int,z_cood,dz))
        int_var = -int_var/(2.0*nuc_mass)
        Tn.append(int_var)
    
    exact_data[:,1] = Tn
    # plt.plot(Tn,'-o')
    # plt.show()

    # Electronic contribution
    Re_phi_z = np.zeros(n_time_steps)
    Im_phi_z = np.zeros(n_time_steps)
    Re_phi_z, Im_phi_z = phi_z()

    den_mat = np.zeros((n_time_steps[0], M))
    den_mat = density()

    diff_Re = diff_manybody_vec(Re_phi_z)
    diff_Im = diff_manybody_vec(Im_phi_z)

    diff2_Re = diff_manybody_vec(diff_Re)
    diff2_Im = diff_manybody_vec(diff_Im)

    Tn_phy = []
    d_geo = []

    for ii in range(n_time_steps[0]):
        int_var = 0.0
        temp_dphy2 = []
        temp_dgeo = []
        
        for jz in range(M):

            temp_NACT1 = np.zeros([Tot_state, Tot_state])
            temp_NACT2 = np.zeros([Tot_state, Tot_state])

            temp_NACT1 = NACT1[Tot_state*jz:Tot_state*(jz+1)][:]
            temp_NACT2 = NACT2[Tot_state*jz:Tot_state*(jz+1)][:]
                        
            temp_Re_jz = Re_phi_z[ii][Tot_state*jz:Tot_state*(jz+1)]   
            temp_Im_jz = Im_phi_z[ii][Tot_state*jz:Tot_state*(jz+1)]

            temp_diff_Re_jz = diff_Re[ii][Tot_state*jz:Tot_state*(jz+1)]   
            temp_diff_Im_jz = diff_Im[ii][Tot_state*jz:Tot_state*(jz+1)]

            temp_diff2_Re_jz = diff2_Re[ii][Tot_state*jz:Tot_state*(jz+1)]   
            temp_diff2_Im_jz = diff2_Im[ii][Tot_state*jz:Tot_state*(jz+1)]

            temp_d1 = 0.0
            temp_d2 = 0.0
            

            #print(jz, ii, temp_NACT1.shape)

            for kk in range(Tot_state):
                for ll in range(Tot_state):
                    temp_d1 = temp_d1 + (temp_Re_jz[kk]*temp_diff_Re_jz[ll] + temp_Im_jz[kk]*temp_diff_Im_jz[ll])*np.abs(temp_NACT1[kk][ll])
                    temp_d2 = temp_d2 + (temp_Re_jz[kk]*temp_Re_jz[ll] + temp_Im_jz[kk]*temp_Im_jz[ll])*temp_NACT2[kk][ll]
            
            # print(temp_d1)
            temp_d1 = 4.0*temp_d1

            # NACTs contribution
            temp_dphy2.append( np.inner( np.array(temp_Re_jz).transpose(), np.array(temp_diff2_Re_jz) ) \
                + np.inner( np.array(temp_Im_jz).transpose(), np.array(temp_diff2_Im_jz)) + temp_d1 + temp_d2 )
            #temp_dphy2.append(temp_d1 + temp_d2)

            # NACT1 contribution
            # temp_var = 0.0
            # for in range(Tot_state):
        temp_dgeo = -np.array(temp_dphy2)/(2.0*nuc_mass)
        #d_geo_z[ii][:] = temp_dgeo[:]

        temp_dphy2 = np.array(temp_dphy2)*den_mat[ii][:]
        temp_dphy2 = list(temp_dphy2)
        
        int_var = integrate.simps(temp_dphy2, z_cood, dz)
        int_var = -int_var/(2.0*nuc_mass)
        Tn_phy.append(int_var)
        
        # Geometric phase
        int_var = 0.0
        int_var = integrate.simps(np.sqrt(np.abs(temp_dgeo)), z_cood, dz)
        d_geo.append(int_var)
    
    exact_data[:,2] = Tn_phy
    exact_data[:,3] = d_geo

    # plt.plot(Tn_phy,'-o')
    # plt.show()

    # Psi contribution

    diff_Re_Beta = diff_manybody_vec(Re_Beta)
    diff_Im_Beta = diff_manybody_vec(Im_Beta)
    diff2_Re_Beta = diff_manybody_vec(diff_Re_Beta)
    diff2_Im_Beta = diff_manybody_vec(diff_Im_Beta)

    Tn_Psi = []

    for ii in range(n_time_steps[0]):
        int_var = 0.0
        temp_d2Psi = []
        
        for jz in range(M):
            temp_Re_jz = []
            temp_Im_jz = []
            temp_d2PsiRe_jz = []
            temp_d2PsiIm_jz = []

            temp_Re_jz = Re_Beta[ii][Tot_state*jz:Tot_state*(jz+1)]   
            temp_Im_jz = Im_Beta[ii][Tot_state*jz:Tot_state*(jz+1)]
            temp_d2PsiRe_jz = diff2_Re_Beta[ii][Tot_state*jz:Tot_state*(jz+1)]   
            temp_d2PsiIm_jz = diff2_Im_Beta[ii][Tot_state*jz:Tot_state*(jz+1)]

            temp_d2Psi.append(np.inner( np.array(temp_Re_jz), np.array(temp_d2PsiRe_jz) ) \
                + np.inner( np.array(temp_Im_jz),np.array(temp_d2PsiIm_jz) ) )
        
        int_var = integrate.simps(temp_d2Psi, z_cood, dz)
        int_var = -int_var/(2.0*nuc_mass)
        Tn_Psi.append(int_var) 
    
    exact_data[:,4] = Tn_Psi
    details = ['%Time',       'Tn-marg','          Tn-Phy','             d-geo','         Tn-Psi']

    with open('exact_data.dat', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(details) 
        write.writerows(exact_data)
    print("Hi Celso") 
    # plt.plot(Tn_Psi,'-o')
    # plt.show()
    #return d_geo_z
    ...    

if __name__ == '__main__':

    read_input()

    # Time propagation
   
    #S_mat = nuclear_phase()
    #print(S_mat[10][:])
    #den_mat = density() # nuclear density
    #all_kinetic_energy() # all kinetic energies contribution
    #tdpes() # Time-dependent potential energy surface

    ani = SubplotAnimation(z_vec(), BOPE, density(), nuclear_phase(), tdpes(), continuity_eq(), time_step)
    ani.save('sticking_exact.mp4')
