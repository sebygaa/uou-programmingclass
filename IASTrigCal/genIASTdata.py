# %%
# Importing packages
# %%
from IASTrigCal import IAST_bi
import numpy as np
# %%
# IAST_bi example
# %%
qm1 = 2.5
b1 = 1
n1 = 0.8

qm2 = 1.5
b2 = 0.3
n2 = 1.5

def iso1(P):
    numor = qm1*b1*P
    denom = 1+ b1*P**n1
    q = numor/denom
    return q

def iso2(P):
    numor = qm2*b2*P
    denom = (1+b2*P)**n2
    q = numor/denom
    return q

P_test = 8
y1_test = 0.07
[q1,q2], fval = IAST_bi(iso1, iso2, y1_test, P_test)
print('[q1, q2] = ', q1,',', q2)
print('pi/RT error = ', fval)
# %%
# Gen 4 points and Interpolate 
# %%
y_targ = 0.006
P_targ = 1.22
y1_1 = 0.005
y1_2 = 0.01
P_1 = 1.2
P_2 = 1.3
f = lambda y1, P: IAST_bi(iso1,iso2, y1, P)

q_11, fval = f(y1_1, P_1)
print('q_11')
print(q_11)
print('fval')
print(fval)
print()
q_12, _ = f(y1_1, P_2)
print('q_12')
print(q_12)
q_21, _ = f(y1_2, P_1)
print('q_21')
print(q_21)
q_22, _ = f(y1_2, P_2)
print('q_22')
print(q_22)

x_y = (y_targ - y1_1)/(y1_2 - y1_1)
x_P = (P_targ - P_1)/(P_2-P_1)
print('x_y = ', x_y)
print('x_P = ', x_P)
q_sol_P1 = (1-x_y)*np.array(q_11) + x_y*np.array(q_21)
q_sol_P2 = (1-x_y)*np.array(q_12) + x_y*np.array(q_22)

print('q_sol_P1')
print(q_sol_P1)
print('q_sol_P2')
print(q_sol_P2)

q_sol = (1-x_P)*q_sol_P1 + x_P*q_sol_P2
print('q_sol = ')
print(q_sol)

# %%
# Functionize this interpolation algorithm
def interIAST2D(x1_targ, x2_targ, x1_minmax, x2_minmax,
                f11,f12,f21,f22):
    x_x1 = (x1_targ - x1_minmax[0])/(x1_minmax[1]-x1_minmax[0])
    x_x2 = (x2_targ - x2_minmax[0])/(x2_minmax[1]-x2_minmax[0])
    # First  dim. reduction 
    f_sol_x1 = (1-x_x1)*f11 + x_x1*f21
    f_sol_x2 = (1-x_x1)*f12 + x_x1*f22
    # second dim. reduction
    f_sol = (1-x_x2)*f_sol_x1 + x_x2*f_sol_x2
    return f_sol

# %%
# TEST InterIAST2D
y_targ = 0.006
P_targ = 1.22
y1_1 = 0.005
y1_2 = 0.01
P_1 = 1.2
P_2 = 1.3
f = lambda y1, P: IAST_bi(iso1,iso2, y1, P)

q_11, _ = f(y1_1, P_1)
q_11_arr = np.array(q_11)
print('q_11')
print(q_11)

q_12, _ = f(y1_1, P_2)
q_12_arr = np.array(q_12)
print('q_12')
print(q_12)

q_21, _ = f(y1_2, P_1)
q_21_arr = np.array(q_21)
print('q_21')
print(q_21)

q_22, _ = f(y1_2, P_2)
q_22_arr = np.array(q_22)
print('q_22')
print(q_22)

y1_ran = [y1_1, y1_2]
P_ran = [P_1, P_2]
q_inter_res = interIAST2D(y_targ, P_targ,
                          y1_ran, P_ran, 
                          q_11_arr,q_12_arr,
                          q_21_arr,q_22_arr)
print()
print('[[q_sol from interpolation]]')
print(q_inter_res)
# %%
# Temperature effect (3D input space)
# %%
R_gas = 8.3145
def Arr(P, T, dH_ad, Tref):
    P_norm = P*np.exp(-dH_ad/R_gas*(1/T - 1/Tref))
    return P_norm

dH_ad1 = -8E3
dH_ad2 = -4E3
T_ref = 300
ArrAd1 = lambda P,T: Arr(P,T,dH_ad1,T_ref)
ArrAd2 = lambda P,T: Arr(P,T,dH_ad1,T_ref)

y_targ = 0.006
P_targ = 1.22
T_targ = 300
y1_1 = 0.005
y1_2 = 0.01
P_1 = 1.2
P_2 = 1.3
T_1 = 295
T_2 = 305

# T1 isotherm
iso1T1 = lambda P: iso1(ArrAd1(P,T_1))
iso2T1 = lambda P: iso2(ArrAd2(P,T_1))

# T2 isotherm
iso1T2 = lambda P: iso1(ArrAd1(P,T_2))
iso2T2 = lambda P: iso2(ArrAd2(P,T_2))

fT1 = lambda y1, P: IAST_bi(iso1T1,iso2T1, y1, P)
fT2 = lambda y1, P: IAST_bi(iso1T2,iso2T2, y1, P)

# T1 uptekes
q_11T1, _ = fT1(y1_1, P_1)
q_11T1_arr = np.array(q_11T1)
print('q_11T1')
print(q_11T1)

q_12T1, _ = fT1(y1_1, P_2)
q_12T1_arr = np.array(q_12T1)
print('q_12T1')
print(q_12T1)

q_21T1, _ = fT1(y1_2, P_1)
q_21T1_arr = np.array(q_21T1)
print('q_21T1')
print(q_21T1)

q_22T1, _ = fT1(y1_2, P_2)
q_22T1_arr = np.array(q_22T1)
print('q_22T1')
print(q_22T1)

q_sol_T1 = interIAST2D(y_targ, P_targ,
                       [y1_1, y1_2], [P_1, P_2],
                       q_11T1_arr, q_12T1_arr,
                       q_21T1_arr, q_22T1_arr)

# T2 uptekes
q_11T2, _ = fT2(y1_1, P_1)
q_11T2_arr = np.array(q_11T2)
print('q_11T1')
print(q_11T2)

q_12T2, _ = fT2(y1_1, P_2)
q_12T2_arr = np.array(q_12T2)
print('q_12T2')
print(q_12T2)

q_21T2, _ = fT2(y1_2, P_1)
q_21T2_arr = np.array(q_21T2)
print('q_21T2')
print(q_21T2)

q_22T2, _ = fT2(y1_2, P_2)
q_22T2_arr = np.array(q_22T2)
print('q_22T2')
print(q_22T2)

q_sol_T2 = interIAST2D(y_targ, P_targ,
                       [y1_1, y1_2], [P_1, P_2],
                       q_11T2_arr, q_12T2_arr,
                       q_21T2_arr, q_22T2_arr)

x_T = (T_targ - T_1)/(T_2 - T_1)
q_sol = (1-x_T)*q_sol_T1 + x_T*q_sol_T2
print('[[q_sol from interIAST2D]]')
print(q_sol)
# %%
# functionalize this
# %% 
def interIAST3D(y_targ, P_targ,
                y1_ran, P_ran, T_ran,
                q_list_T1, q_list_T2):
    q_sol_T1 = interIAST2D(y_targ, P_targ,
                           y1_ran,P_ran,
                           q_list_T1[0],q_list_T1[1],
                           q_list_T1[2],q_list_T1[3])
    q_sol_T2 = interIAST2D(y_targ, P_targ,
                           y1_ran,P_ran,
                           q_list_T2[0],q_list_T2[1],
                           q_list_T2[2],q_list_T2[3])
    x_T = (T_targ - T_ran[0])/(T_ran[1] - T_ran[0])
    q_sol = (1-x_T)*q_sol_T1 + x_T*q_sol_T2
    return q_sol

# %%
# Test interIAST3D
R_gas = 8.3145
def Arr(P, T, dH_ad, Tref):
    P_norm = P*np.exp(-dH_ad/R_gas*(1/T - 1/Tref))
    return P_norm

dH_ad1 = -8E3
dH_ad2 = -4E3
T_ref = 300
ArrAd1 = lambda P,T: Arr(P,T,dH_ad1,T_ref)
ArrAd2 = lambda P,T: Arr(P,T,dH_ad1,T_ref)

y_targ = 0.006
P_targ = 1.22
T_targ = 300
y1_1 = 0.005
y1_2 = 0.01
P_1 = 1.2
P_2 = 1.3
T_1 = 295
T_2 = 305

# T1 isotherm
iso1T1 = lambda P: iso1(ArrAd1(P,T_1))
iso2T1 = lambda P: iso2(ArrAd2(P,T_1))

# T2 isotherm
iso1T2 = lambda P: iso1(ArrAd1(P,T_2))
iso2T2 = lambda P: iso2(ArrAd2(P,T_2))

fT1 = lambda y1, P: IAST_bi(iso1T1,iso2T1, y1, P)
fT2 = lambda y1, P: IAST_bi(iso1T2,iso2T2, y1, P)

# T1 uptekes
q_11T1, _ = fT1(y1_1, P_1)
q_11T1_arr = np.array(q_11T1)
print('q_11T1')
print(q_11T1)

q_12T1, _ = fT1(y1_1, P_2)
q_12T1_arr = np.array(q_12T1)
print('q_12T1')
print(q_12T1)

q_21T1, _ = fT1(y1_2, P_1)
q_21T1_arr = np.array(q_21T1)
print('q_21T1')
print(q_21T1)

q_22T1, _ = fT1(y1_2, P_2)
q_22T1_arr = np.array(q_22T1)
print('q_22T1')
print(q_22T1)
'''
q_sol_T1 = interIAST2D(y_targ, P_targ,
                       [y1_1, y1_2], [P_1, P_2],
                       q_11T1_arr, q_12T1_arr,
                       q_21T1_arr, q_22T1_arr)
'''
# T2 uptekes
q_11T2, _ = fT2(y1_1, P_1)
q_11T2_arr = np.array(q_11T2)
print('q_11T1')
print(q_11T2)

q_12T2, _ = fT2(y1_1, P_2)
q_12T2_arr = np.array(q_12T2)
print('q_12T2')
print(q_12T2)

q_21T2, _ = fT2(y1_2, P_1)
q_21T2_arr = np.array(q_21T2)
print('q_21T2')
print(q_21T2)

q_22T2, _ = fT2(y1_2, P_2)
q_22T2_arr = np.array(q_22T2)
print('q_22T2')
print(q_22T2)

q_T1 = [q_11T1_arr, q_12T1_arr, q_21T1_arr, q_22T1_arr]
q_T2 = [q_11T2_arr, q_12T2_arr, q_21T2_arr, q_22T2_arr]

q_sol = interIAST3D(y_targ,P_targ,
                    [y1_1, y1_2],[P_1,P_2], [T_1,T_2],
                    q_T1, q_T2)

print('[[q_sol from interIAST3D]]')
print(q_sol)
# %%
# Generate IAST data 2D
# %%
# Increase the number at last from 10 to 10000
y1_ran = np.linspace(0,0.1,20+1)
P_ran = np.linspace(0,10,20+1)

q1_list = []
q2_list = []

for yy in y1_ran:
    q1_tmp_list = []
    q2_tmp_list = []
    for PP in P_ran:
        [q1_tmp, q2_tmp], fval = IAST_bi(iso1, iso2,
                                        yy, PP)
        if fval > 1E-2:
            print('[Warnning] pi/RT err = ', fval)
            print('at y1=', yy, 'P=',PP)
        q1_tmp_list.append(q1_tmp)
        q2_tmp_list.append(q2_tmp)

    q1_list.append(q1_tmp_list)
    q2_list.append(q2_tmp_list)

q1_arr_data = np.array(q1_list)
q2_arr_data = np.array(q2_list)

# %%
# How to search?
# %%
y1_targ= 0.599
P_targ = 4.0

y_diff = y1_targ- y1_ran
P_diff = P_targ - P_ran

i_1 = np.argmin(y_diff**2)
i_2 = np.argmin(P_diff**2)

if y_diff[i_1] < 0:
    i_1 = i_1 - 1
if P_diff[i_2] < 0:
    i_2 =i_2 - 1
print('i_1 = ')
print(i_1)
print('i_2 = ')
print(i_2)

y1_1 = y1_ran[i_1]
y1_2 = y1_ran[i_1+1]
y_list = [y1_1, y1_2]

P_1 = P_ran[i_2]
P_2 = P_ran[i_2+1]
P_list = [P_1, P_2]

q11 = q1_arr_data[i_1, i_2]
q12 = q1_arr_data[i_1, i_2+1]
q21 = q1_arr_data[i_1+1, i_2]
q22 = q1_arr_data[i_1+1, i_2+1]

q1_sol = interIAST2D(y1_targ,P_targ,
                    y_list,P_list,
                    q11,q12,q21,q22)

q11 = q2_arr_data[i_1, i_2]
q12 = q2_arr_data[i_1, i_2+1]
q21 = q2_arr_data[i_1+1, i_2]
q22 = q2_arr_data[i_1+1, i_2+1]
q2_sol = interIAST2D(y1_targ,P_targ,
                    y_list,P_list,
                    q11,q12,q21,q22)
print('[[q1_sol]]')
print(q1_sol)
print()
print('[[q2_sol]]')
print(q2_sol)
# %%
