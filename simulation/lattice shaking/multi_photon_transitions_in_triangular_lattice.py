#%%
# -*- coding: utf-8 -*-
"""
Hubbard parameters for triangular optical lattice
"""
import numpy as np
lam = 1 # Normalized wavelength
d = lam/2 # lattice constant
# k = 2*np.pi/lam # wave number
# k1 = k*np.array([0,-1]) # reciprocal lattice vector 1
# k2 = k*np.array([np.sqrt(3)/2, 1/2]) # reciprocal lattice vector 2
# k3 = k*np.array([-np.sqrt(3)/2, 1/2]) # reciprocal lattice vector 3

lambda_L = 830*10**(-9) #wave lenght of triangular lattice beam [m] 
h_bar = 1.054571596*10**(-34) #Converted Plank const
mrb = 87*1.66053873*10**(-27) #Single atom mass of 87Yb
er = h_bar**2/(2*mrb)*(2*np.pi/lambda_L)**2 #Recoil energy of 532nm lattice

n = 8 # include (2n+1)^2 plane waves in calculation <=> calculate over (2n+1)^2 bands
m = 8 # calculate over (2m(+1))^2 quasimomenta
# %%

#Quasimomentum list
q_list = [(0,0), (1/2,0), (2/3,1/3), (1/2,1/2)] #Gamma =(0,0), M = (1/2,0), K = (2/3,1/3), X = (1/2,1/2)

v_list = np.linspace(2,6,50) #Potential = 2ER - 6ER
gap_min = np.zeros(len(v_list))
gap_max = np.zeros(len(v_list))


Nsite=2*n+1
l_list = [(x, y) for x in np.linspace(-n, n, Nsite, dtype=np.int) for y in np.linspace(-n, n, Nsite, dtype=np.int)]
E = np.zeros([len(q_list), Nsite**2])
C = np.zeros([Nsite**2, len(q_list), Nsite**2])
H_tmp = np.zeros([Nsite**2, Nsite**2])

l_list_1 = np.array(l_list)[:, 0]
l_list_2 = np.array(l_list)[:, 1]
l2, l1 = np.meshgrid(l_list_1, l_list_1)
m2, m1 = np.meshgrid(l_list_2, l_list_2)

l_diffs_1 = l1 - l2
l_diffs_2 = m1 - m2
l_diffs = l_diffs_1 * l_diffs_2
condition_1 = (np.abs(l_diffs_1) == 1) * (m1 == m2)
condition_2 = (l1 == l2) * (np.abs(l_diffs_2) == 1)
condition_3 = (l_diffs == 1)

for i_v, v in enumerate(v_list):
    H_tmp = np.zeros([Nsite**2, Nsite**2])
    H_tmp[condition_1 == 1] = -v/4
    H_tmp[condition_2 == 1] += -v/4
    H_tmp[condition_3 == 1] += -v/4
    for i_q, q in enumerate(q_list):
        H = np.copy(H_tmp)
        K = 3 * ((q[0] - l1)**2 + (q[1] - m1)**2 - (q[0] - l2) * (q[1] - m2))
        H += ((l1 == l2) * (m1 == m2)) * K
        E0, P = np.linalg.eig(H)
        rearrangedEvalsVecs = sorted(zip(E0, P.T), key=lambda x: x[0].real, reverse=False)
        E[i_q, :], tmp = map(list, zip(*rearrangedEvalsVecs))
        #C[:, i_q, :] = np.array(tmp)
    gap_min[i_v] = np.min(E[:,1]-E[:,0])
    gap_max[i_v] = np.max(E[:,1]-E[:,0])
# %%
import matplotlib.pyplot as plt
gap_min *= er/(2*np.pi*h_bar)*10**(-3) # unit conversion: [E_R] => [kHz]
gap_max *= er/(2*np.pi*h_bar)*10**(-3) 

fig, ax = plt.subplots(figsize=[5,8])
for n in range(1,6):
    ax.plot(v_list,gap_min/n,v_list,gap_max/n,color = "black")
    ax.fill_between(v_list,gap_min/n,gap_max/n,label="n="+str(n))
ax.set_xlabel(r"Lattice depths $v_{12}=v_{23} = v_{31}$ $[E_R]$", fontsize=15)
ax.set_ylabel(r"$Min[E_1-E_0]/n$,$Max[E_1-E_0]/n$ [kHz]", fontsize=15)
ax.set_xlim([2,6])
ax.set_ylim([0.5,3.5])
ax.legend()
# %%
