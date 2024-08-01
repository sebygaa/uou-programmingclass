# %%
# Importing
# %%
from IASTrigCal import IAST_bi
from IASTrigCal import x2err_reg
from IASTrigCal import iso2pi
import numpy as np
import matplotlib.pyplot as plt
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

#P_test = 8
#y1_test = 0.07
P_test = 7.5
y1_test = 0.005
[q1,q2], fval = IAST_bi(iso1, iso2, y1_test, P_test)
print('[q1, q2] = ', q1,',', q2)
print('pi/RT error = ', fval)
# %%
# Error graph for differe solid mole fraction
# %%
x2er = lambda x: x2err_reg(iso1, iso2, y1_test, P_test, x)
#err_test = x2er(0.2)
#print(err_test)
#x_ran = np.linspace(0.5,0.6, 1001)
x_ran = np.linspace(0.048+1E-5,0.048+100E-5, 5001)
err_list = []
for xx in x_ran:
    err_tmp = x2er(xx)
    err_list.append(err_tmp)

# %%
# Graph
# %%
fig = plt.figure()
plt.plot(x_ran, err_list, linewidth = 1.9)
plt.xlabel('Guessed solid mole fraction (x$_{guess}$) ')
plt.ylabel('Spreadin P error: ((pi/RT)$_{1}$ - (pi/RT)$_{2}$)$^{2}$')
plt.savefig('Err_vs_xguess.png',dpi=100)

# %%
pi1_list = []
pi2_list = []
for xx in x_ran:
    P_o1 = y1_test*P_test/xx
    # (1-y1)*P = (1-x1)*P_o2
    P_o2 = (1-y1_test)*P_test/(1-xx)
    pi1_tmp = iso2pi(iso1, P_o1, 25)
    pi2_tmp = iso2pi(iso2, P_o2, 25)
    pi1_list.append(pi1_tmp)
    pi2_list.append(pi2_tmp)

fig = plt.figure()
plt.plot(x_ran, pi1_list, linewidth = 1.9)
plt.plot(x_ran, pi2_list, linewidth = 1.9)
plt.xlabel('Guessed solid mole fraction (x$_{guess}$) ')
plt.ylabel('Spreadin P: (pi/RT)$_{1 or 2}$')
plt.savefig('Err_vs_xguess.png',dpi=100)


# %%
