#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 22:15:31 2021

@author: shni2598
"""
#%%
sys_argv = 0

loop_num = -1

#param_in = [[0.08,6.5]]
repeat = 20

# for num_ee in [240]:#np.linspace(125, 297, 10,dtype=int): 
#     for num_ei in [400]:#np.linspace(204, 466, 10, dtype=int):#[320]:               
#         for num_ie in [150]:#[260]:#np.linspace(246, 300, 6,dtype=int):#[221]:#np.linspace(156, 297, 10,dtype=int):
#             for num_ii in [230]:#np.linspace(135, 170,6,dtype=int):#[129]:#np.linspace(93, 177, 10,dtype=int):
#for ie_r_e in [2.76*6.5/5.8]:# np.linspace(3.07,3.13,7):#np.arange(2.5,4.0,0.04):#[3.1]:#np.linspace(3.10,3.45,6):#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
#    for ie_r_e1 in np.array([0.96,0.98,1.02,1.04]):
#        for ie_r_e2 in [1]:#np.arange(0.96,1.041,0.02):
#            for w_ee_ in [4*5.8]:#[4]:
#               for w_ie_ in [None]:#[23.12]:#[17]:#np.linspace(17,23,7):#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
#                    for ie_r_i in [2.450*6.5/5.8]:#[2.719]:#[2.817]:#[2.786]:#np.linspace(2.5,3.0,15):#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
#                        for w_ii_ in [None]:#[25]:#np.linspace(20,27,8):#np.arange(1.4,1.81,0.1)]]:#np.linspace(1.,1.,1):
#                            for w_ei_ in [5*5.8]:#[6.35]:
tau_k_ = 60
chg_adapt_range = 7 

for w_extnl_ in [10]:
    for delta_gk_1 in [3.5]:#np.linspace(7,15,9)]]:
        for delta_gk_2 in [16]:
            for new_delta_gk_2 in [1]:#np.arange(0.15,0.31,0.05)*delta_gk_:#np.linspace(1,5,5):
                for tau_s_di_ in np.linspace(4.4,4.4,1):
                    for tau_s_de_ in np.linspace(5.,5.,1):
                        for tau_s_r_ in [1]:#np.linspace(1,1,1):
                            for scale_w_12_e in np.arange(1.4, 1.81, 0.05):#np.arange(0.8,1.21,0.05):
                                for scale_w_12_i in scale_w_12_e*np.array([1]):#np.arange(0.9,1.11,0.1):#np.arange(0.8,1.21,0.05):#[scale_w_12_e]:#0.2*np.arange(0.8,1.21,0.05):
                                    for scale_w_21_e in np.arange(0.8, 1.21, 0.05):#np.arange(0.8,1.21,0.05):#[1]:#np.arange(0.8,1.21,0.05):
                                        for scale_w_21_i in scale_w_21_e*np.array([1]):#np.arange(0.9,1.11,0.1):#np.arange(0.8,1.21,0.05):#[1]:#[scale_w_21_e]:#0.35*np.arange(0.8,1.21,0.05):
                                            for tau_p_d_e1_e2 in [5]:
                                                for tau_p_d_e1_i2 in [5]:
                                                    for tau_p_d_e2_e1 in [10]:
                                                        for tau_p_d_e2_i1 in [10]:
                                                            for peak_p_e1_e2 in [0.5]:#np.arange(0.3,0.451,0.05):
                                                                for peak_p_e1_i2 in [0.5]:#np.arange(0.25,0.401,0.05):
                                                                    for peak_p_e2_e1 in [0.15]:
                                                                        for peak_p_e2_i1 in [0.15]*repeat:
                                                                    
                                                            #for decay_p_ie_p_ii in [20]:    
                                                                #for ie_ratio_ in 3.375*np.arange(0.94, 1.21, 0.02):#(np.arange(0.7,1.56,0.05)-0.02):#np.linspace(2.4, 4.5, 20):
                                                                            loop_num += 1
                                                                            if loop_num == sys_argv:
                                                                                print('loop_num:',loop_num)
                                                                                break
                                                                            else: continue
                                                                            break
                                                                        else: continue
                                                                        break
                                                                    else: continue
                                                                    break
                                                                else: continue
                                                                break
                                                            else: continue
                                                            break
                                                        else: continue                    
                                                        break
                                                    else: continue
                                                    break
                                                else: continue
                                                break
                                            else: continue
                                            break
                                        else: continue
                                        break
                                    else: continue
                                    break
                                else: continue
                                break
                            else: continue
                            break
                        else: continue
                        break
                    else: continue
                    break
                else: continue
                break
            else: continue
            break
        else: continue
        break
    else: continue
    break
    #                             else: continue
    #                             break
    #                         else: continue
    #                         break
    #                     else: continue
    #                     break
    #                 else: continue
    #                 break
    #             else: continue
    #             break
    #         else: continue
    #         break
    #     else: continue
    #     break
    # else: continue
    # break
               
    #             else: continue
    #             break
    #         else: continue
    #         break
    #     else: continue
    #     break
    # else: continue
    # break

if loop_num != sys_argv: sys.exit("Error: wrong PBS_array_id")                    

#%%
def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               
def find_w_i(w_e, num_e, num_i, ie_ratio):
    return (w_e*num_e)*ie_ratio/num_i

w_ie_1 = 115; w_ii_1 = 100
ie_r_e = 2.76*6.5/5.8; ie_r_e1 = 1.56
ie_r_i = 2.450*6.5/5.8; ie_r_i1 = 1.482

ijwd1 = pre_process_sc.get_ijwd()
ijwd1.Ne = 64*64; ijwd1.Ni = 32*32
ijwd1.width = 64#79
#ijwd1.w_ee_mean *= 2; ijwd1.w_ei_mean *= 2; ijwd1.w_ie_mean *= 2; ijwd1.w_ii_mean *= 2;
#scale_d_p = 1 #np.sqrt(8/5) 
ijwd1.decay_p_ee = 7 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd1.decay_p_ei = 9 #* scale_d_p_ei# scale_d_p # decay constaw_ie_ in [115]nt of e to i connection probability as distance increases
ijwd1.decay_p_ie = 19 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd1.decay_p_ii = 19 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases
ijwd1.delay = [0.5,2.5]

num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd1.mean_SynNumIn_ee = num_ee     ; # p = 0.08
ijwd1.mean_SynNumIn_ei = num_ei #* 8/5     ; # p = 0.125
ijwd1.mean_SynNumIn_ie = num_ie  #scale_d_p_i    ; # p = 0.2
ijwd1.mean_SynNumIn_ii = num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

ijwd1.w_ee_mean = find_w_e(w_ie_1, num_ie, num_ee, ie_r_e*ie_r_e1)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd1.w_ei_mean = find_w_e(w_ii_1, num_ii, num_ei, ie_r_i*ie_r_i1)
ijwd1.w_ie_mean = w_ie_1 #find_w_i(w_ee_, num_ee, num_ie, ie_r_e*ie_r_e1)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
ijwd1.w_ii_mean = w_ii_1 #find_w_i(w_ei_, num_ei, num_ii, ie_r_i)#w_ii_
        
ijwd1.generate_ijw()
#ijwd1.generate_d_rand()
ijwd1.generate_d_dist()

# ijwd1.w_ee *= scale_ee_1#tau_s_de_scale_d_p_i
# ijwd1.w_ei *= scale_ei_1 #tau_s_de_ #5*nS
# ijwd1.w_ie *= scale_ie_1#tau_s_di_#25*nS
# ijwd1.w_ii *= scale_ii_1#tau_s_di_#
param_a1 = {**ijwd1.__dict__}



del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee']  
del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei'] 
del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie'] 
del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']

#w_ee_2 = 4*5.8; w_ei_2 = 5*5.8
w_ie_2 = 115; w_ii_2 = 140
ie_r_e = 2.76*6.5/5.8; ie_r_e2 = 1.03
ie_r_i = 2.450*6.5/5.8; ie_r_i2 = 1.04

ijwd2 = pre_process_sc.get_ijwd()
ijwd2.Ne = 64*64; ijwd2.Ni = 32*32
ijwd2.width = 64#79
#ijwd2.w_ee_mean *= 2; ijwd2.w_ei_mean *= 2; ijwd2.w_ie_mean *= 2; ijwd2.w_ii_mean *= 2;
#scale_d_p = 1 #np.sqrt(8/5) 
ijwd2.decay_p_ee = 7#8*0.8 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd2.decay_p_ei = 9#10*0.82 #* scale_d_p_ei# scale_d_p # decay constant of e to i connection probability as distance increases
ijwd2.decay_p_ie = 19#20*0.86 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd2.decay_p_ii = 19#20*0.86 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases
ijwd2.delay = [0.5,2.5]

num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd2.mean_SynNumIn_ee = num_ee#num_ee     ; # p = 0.08
ijwd2.mean_SynNumIn_ei = num_ei#num_ei #* 8/5     ; # p = 0.125
ijwd2.mean_SynNumIn_ie = num_ie#num_ie  #scale_d_p_i    ; # p = 0.2
ijwd2.mean_SynNumIn_ii = num_ii#num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

# ijwd2.w_ee_mean = w_ee_2#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
# ijwd2.w_ei_mean = w_ei_2#find_w_e(w_ii_, num_ii, num_ei, ie_r_i)
# ijwd2.w_ie_mean = find_w_i(w_ee_2, num_ee, num_ie, ie_r_e*ie_r_e2)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
# ijwd2.w_ii_mean = find_w_i(w_ei_2, num_ei, num_ii, ie_r_i*ie_r_i2)#w_ii_

ijwd2.w_ee_mean = find_w_e(w_ie_2, num_ie, num_ee, ie_r_e*ie_r_e2)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd2.w_ei_mean = find_w_e(w_ii_2, num_ii, num_ei, ie_r_i*ie_r_i2)
ijwd2.w_ie_mean = w_ie_2 #find_w_i(w_ee_, num_ee, num_ie, ie_r_e*ie_r_e1)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
ijwd2.w_ii_mean = w_ii_2 #find_w_i(w_ei_, num_ei, num_ii, ie_r_i)#w_ii_

   
ijwd2.generate_ijw()
#ijwd2.generate_d_rand()
ijwd2.generate_d_dist()

param_a2 = {**ijwd2.__dict__}

del param_a2['i_ee'], param_a2['j_ee'], param_a2['w_ee'], param_a2['d_ee'], param_a2['dist_ee'] 
del param_a2['i_ei'], param_a2['j_ei'], param_a2['w_ei'], param_a2['d_ei'], param_a2['dist_ei']
del param_a2['i_ie'], param_a2['j_ie'], param_a2['w_ie'], param_a2['d_ie'], param_a2['dist_ie'] 
del param_a2['i_ii'], param_a2['j_ii'], param_a2['w_ii'], param_a2['d_ii'], param_a2['dist_ii']


ijwd_inter = preprocess_2area.get_ijwd_2()

ijwd_inter.Ne1 = 64*64; ijwd_inter.Ne2 = 64*64; 
ijwd_inter.width1 = 64; ijwd_inter.width2 = 64;
ijwd_inter.p_inter_area_1 = 1/3; ijwd_inter.p_inter_area_2 = 1/3
ijwd_inter.section_width_1 = 3;  ijwd_inter.section_width_2 = 3; 
ijwd_inter.peak_p_e1_e2 = peak_p_e1_e2; ijwd_inter.tau_p_d_e1_e2 = tau_p_d_e1_e2
ijwd_inter.peak_p_e1_i2 = peak_p_e1_i2; ijwd_inter.tau_p_d_e1_i2 = tau_p_d_e1_i2        
ijwd_inter.peak_p_e2_e1 = peak_p_e2_e1; ijwd_inter.tau_p_d_e2_e1 = tau_p_d_e2_e1
ijwd_inter.peak_p_e2_i1 = peak_p_e2_i1; ijwd_inter.tau_p_d_e2_i1 = tau_p_d_e2_i1

ijwd_inter.w_e1_e2_mean = 5*scale_w_12_e; ijwd_inter.w_e1_i2_mean = 5*scale_w_12_i
ijwd_inter.w_e2_e1_mean = 5*scale_w_21_e*1; ijwd_inter.w_e2_i1_mean = 5*scale_w_21_i*1

ijwd_inter.generate_ijwd()

param_inter = {**ijwd_inter.__dict__}

del param_inter['i_e1_e2'], param_inter['j_e1_e2'], param_inter['w_e1_e2'], param_inter['d_e1_e2'] 
del param_inter['i_e1_i2'], param_inter['j_e1_i2'], param_inter['w_e1_i2'], param_inter['d_e1_i2'] 
del param_inter['i_e2_e1'], param_inter['j_e2_e1'], param_inter['w_e2_e1'], param_inter['d_e2_e1'] 
del param_inter['i_e2_i1'], param_inter['j_e2_i1'], param_inter['w_e2_i1'], param_inter['d_e2_i1']

#%%
del param_a1, param_a2, param_inter
#%%
lattice_trg_1 = ijwd_inter.e_lattice1[ijwd_inter.j_e2_e1[ijwd_inter.i_e2_e1 == 2000]] 

lattice_trg_2 = ijwd_inter.e_lattice2[ijwd_inter.j_e1_e2[ijwd_inter.i_e1_e2 == 2002]] 

#%%
plt.figure()
plt.plot(lattice_trg_1[:,0], lattice_trg_1[:,1], 'o')
plt.xlim([-32, 32])
plt.ylim([-32, 32])

plt.figure()
plt.plot(lattice_trg_2[:,0], lattice_trg_2[:,1], 'o')
plt.xlim([-32, 32])
plt.ylim([-32, 32])
#%%
lattice_trg_2_s = ijwd2.e_lattice[ijwd2.j_ee[ijwd2.i_ee == 2000]] 

#lattice_trg_2 = ijwd2.e_lattice2[ijwd2.j[ijwd2.i == 2002]] 

#%%
plt.figure()
plt.plot(lattice_trg_2_s[:,0], lattice_trg_2_s[:,1], 'o')
plt.xlim([-32, 32])
plt.ylim([-32, 32])
#%%
plt.figure()
plt.plot(lattice_trg_2[:,0], lattice_trg_2[:,1], 'o')
plt.xlim([-32, 32])
plt.ylim([-32, 32])

#%%
mat_test = np.arange(15).reshape(3,5)
mat_test_sps = scipy.sparse.csc_matrix(mat_test)


