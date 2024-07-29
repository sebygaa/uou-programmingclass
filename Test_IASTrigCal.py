# %%
# Importin Functions
# %%
from IASTrigCal import IAST_bi
import numpy as np
import matplotlib.pyplot as plt

# %% 
# TEST Isotherm Function

# %%

qm1 = 2
b1 = 0.3
n1 = 0.7
def iso1(P):
    numer = qm1*b1*P**n1
    denom = 1+b1*P**n1
    q_tmp1 = numer/denom
    return q_tmp1

qm2 = 4
b2 = 0.5
n2 = 1
def iso2(P):
    numer = qm2*b2*P**n2
    denom = 1+b2*P**n2
    q_tmp2 = numer/denom
    return q_tmp2

y1_ran = np.linspace(0,1, 81)
P_fix = 5

q1_list = []
q2_list = []
for yy in y1_ran:
    [q1, q2], err_val = IAST_bi(iso1, iso2,
                            yy, P_fix)

    q1_list.append(q1)
    q2_list.append(q2)
    
q1_arr = np.array(q1_list)
q2_arr = np.array(q2_list)


# %%
# Graph
# %%
fig, ax = plt.subplots(figsize = [5,3.7])
ax.plot(y1_ran, q1_arr)
ax.plot(y1_ran, q2_arr)
fig.show()
# %%
