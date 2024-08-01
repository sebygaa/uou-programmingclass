
# %%
# Importing
# %%
import pickle
import numpy as np
from IASTrigCal import IAST_bi

# %%
# Generating func
def genIASTdata2D(iso1, iso2, y1_ran, P_ran,
                file_name='genIASTdata2D.pkl'):
    # y1_ran = np.linspace(0,1,50+1)
    # P_ran = np.linspace(0,25,10+1)
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
    IASTdata = {'q1': q1_arr_data,
                'q2': q2_arr_data,
                'y1': y1_ran,
                'P': P_ran}
    with open(file_name, 'wb') as file:
        pickle.dump(IASTdata, file)


# %%
# interLinIAST2D
# %%
# Functionize this interpolation algorithm
def interLinIAST2D(x1_targ, x2_targ, x1_minmax, x2_minmax,
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
# interLinIAST3D
# %%
def interLinIAST3D(y_targ, P_targ,T_targ,
                y1_ran, P_ran, T_ran,
                q_list_T1, q_list_T2):
    q_sol_T1 = interLinIAST2D(y_targ, P_targ,
                           y1_ran,P_ran,
                           q_list_T1[0],q_list_T1[1],
                           q_list_T1[2],q_list_T1[3])
    q_sol_T2 = interLinIAST2D(y_targ, P_targ,
                           y1_ran,P_ran,
                           q_list_T2[0],q_list_T2[1],
                           q_list_T2[2],q_list_T2[3])
    x_T = (T_targ - T_ran[0])/(T_ran[1] - T_ran[0])
    q_sol = (1-x_T)*q_sol_T1 + x_T*q_sol_T2
    return q_sol
# %%
# PredLinIAST2D
# %%
#with open('data.pkl', 'rb') as file:
#    IASTdata = pickle.load(file)

#y1_ran = np.linspace(0,1,50+1)
#P_ran = np.linspace(0,25,10+1)

#q1_arr_data = IASTdata['q1']
#q2_arr_data = IASTdata['q2']

class PredLinIAST2D:
    def __init__(self, file_name='genIASTdata2D.pkl'):
        with open(file_name, 'rb') as file:
            IASTdata = pickle.load(file)
        y1_ran = IASTdata['y1']
        P_ran = IASTdata['P']
        q1_arr_data = IASTdata['q1']
        q2_arr_data = IASTdata['q2']
        
        self.y1 = y1_ran
        self.P = P_ran 
        self.q1 = q1_arr_data
        self.q2 = q2_arr_data
        del(IASTdata)
        
    def predict(self, y1_targ,P_targ,):
        y1_ran = self.y1
        P_ran = self.P
        q1_arr_data = self.q1
        q2_arr_data = self.q2

        y_diff = y1_targ- y1_ran
        P_diff = P_targ - P_ran

        i_1 = np.argmin(y_diff**2)
        i_2 = np.argmin(P_diff**2)

        if y_diff[i_1] < 0:
            i_1 = i_1 - 1
        if P_diff[i_2] < 0:
            i_2 =i_2 - 1
        
        y1_1 = y1_ran[i_1]
        y1_2 = y1_ran[i_1+1]
        y_list = [y1_1, y1_2]

        P_1 = P_ran[i_2]
        P_2 = P_ran[i_2+1]
        P_list = [P_1, P_2]
        # q1: interLinIAST2D
        q11 = q1_arr_data[i_1, i_2]
        q12 = q1_arr_data[i_1, i_2+1]
        q21 = q1_arr_data[i_1+1, i_2]
        q22 = q1_arr_data[i_1+1, i_2+1]

        q1_sol = interLinIAST2D(y1_targ,P_targ,
                            y_list,P_list,
                            q11,q12,q21,q22)
        
        # q2: interLinIAST2D
        q11 = q2_arr_data[i_1, i_2]
        q12 = q2_arr_data[i_1, i_2+1]
        q21 = q2_arr_data[i_1+1, i_2]
        q22 = q2_arr_data[i_1+1, i_2+1]

        q2_sol = interLinIAST2D(y1_targ,P_targ,
                            y_list,P_list,
                            q11,q12,q21,q22)
        return q1_sol, q2_sol
# %%
if __name__ == '__main__':
    print()