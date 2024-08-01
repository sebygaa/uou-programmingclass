# %%
# importing
import numpy as np

from IASTrigCal import PredLinIAST2D
from IASTrigCal import genIASTdata2D
# 
# %%
# Generate first
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

y1_ran = np.linspace(0,1,50+1)
P_ran = np.linspace(0,25,10+1)

genIASTdata2D(iso1, iso2, y1_ran, P_ran)

# %%
# Predict
y1_targ = 0.03
P_targ = 9.2
surModel = PredLinIAST2D()
q1,q2 = surModel.predict(y1_targ, P_targ)
print('q1')
print(q1)
print('q2')
print(q2)
# %%
