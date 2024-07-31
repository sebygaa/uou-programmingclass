# %%
# importing packages

# %%
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simpson

# %%
# Spreading pressure calculation

# %%
def iso2pi(fun, P_o, N_integ):
    P_ran = np.linspace(1E-7, P_o, N_integ)
    q_ran = fun(P_ran)
    q_ov_P= q_ran/P_ran
    pi_ov_RT = simpson(q_ov_P, x = P_ran)
    return pi_ov_RT

# %%
# Error estimating
# %%
def x2err_reg(fun1, fun2, y1, P, x1_est1):
    # y1*P = x1*(P_o1)
    P_o1 = y1*P/x1_est1
    # (1-y1)*P = (1-x1)*P_o2
    P_o2 = (1-y1)*P/(1-x1_est1)
    N_integ_node = 20
    pi_ov_RT1 = iso2pi(fun1, P_o1, N_integ_node)
    pi_ov_RT2 = iso2pi(fun2, P_o2, N_integ_node)
    mse = np.sqrt((pi_ov_RT1 - pi_ov_RT2)**2)
    return mse

# %%
# x to q val function x2q
# %%
def x2q(fun1, fun2, y1, P, x1):
    # y1*P = x1*(P_o1)
    P_o1 = y1*P/x1
    # (1-y1)*P = (1-x1)*P_o2
    P_o2 = (1-y1)*P/(1-x1)
    # 1/q_tot = sum( 1/q_o )
    q_o1 = fun1(P_o1)
    q_o2 = fun2(P_o2)
    q_tot = 1/(x1/q_o1 + (1-x1)/q_o2)
    q1 = x1*q_tot
    q2 = (1-x1)*q_tot
    return q1,q2

# %%
## Set Five diff cases
## Case1 : y1 >= 1-1E-6
## Case2 : y1 >= 1-0.1
## Case3 : 0.1 < y1 < 1-0.1
## Case4 : y1 <= 0.1 
## Case5 : y1 <= 1E-6

# %%

# Searching each digit (grid search)
def gridupdate(fun, x_center, mesh, N):
    # x range
    N_left = np.round(N/2)
    N_right = N-N_left
    x_left = x_center - (N_left-1)*mesh
    x_right = x_center + N_right*mesh
    x_ran = np.linspace(x_left,x_right, N)
    # y estimateion
    y_list = []
    for xx in x_ran:
        y_tmp = fun([xx])
        y_list.append(y_tmp)
    y_ran = np.array(y_list)
    # Find minimum
    arg_min = np.argmin(y_ran)
    xmin = x_ran[arg_min]
    ymin = y_ran[arg_min]
    # Activate to see the results
    #print('y_values:')
    #print(ymin)
    #print('arg_min = ', arg_min)
    return xmin, ymin
    
def gridsearch(fun, x_init, mesh_init, 
            n_iter = 10, r_shrink_mesh = 0.3,N_search=20):
    #N_search = 20
    x_update = x_init
    mesh_size = mesh_init
    for ii in range(n_iter):
        x_update, fval = gridupdate(fun,x_update, 
                            mesh_size, N_search)
        mesh_size = mesh_size*r_shrink_mesh
        # Activate to see the results
        #print('x  value = ', x_update)
        #print('Func val. =',fun([x_update]))
        #print('mesh size = ', mesh_size*5)
    return x_update, fval

# FINDING IAST 
def IAST_bi(fun1, fun2, y1, P):
    # Case1: y1 > 1-1E-7
    if y1 > 1-1E-7:
        q = fun1(P)
        return [q,0], 0

    # Case2 : y1 <= 1E-7
    if y1 < 1E-7:
        q= fun2(P)
        return [0 ,q], 0

    # OBJ function 
    def obj_err(x):
        Penalty = 0
        x_est = x[0]
        if x_est > 1-1E-7:
            Penalty = Penalty+1E4*(x_est-1)**2
            x_est = 1-1E-7
        elif x_est < 1E-7:
            Penalty = Penalty+1E4*(x_est)**2
            x_est = 1E-7
        obj_mse = x2err_reg(fun1, fun2,
                        y1, P, x_est)
        return obj_mse + Penalty
    
    # Case3: Regular case with optim solver
    is_comp_err = False
    try:
        x_est_0 = [0.5]
        opt_res = minimize(obj_err, x_est_0,)
        fval = opt_res.fun
        
        if opt_res.fun > 1E-2:
            is_comp_err = True
        
        if is_comp_err == False:
            x_sol = opt_res.x[0]
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)    
############            print('SuCCeSSS with optim solver !')
            return [q1,q2], fval
    except:
        is_comp_err = True
        #print('Except is working')
    
    # Case4: Optim solver failing case
    if is_comp_err:
        #print('Failing cases')
        if y1 > 0.5:
            x_est0 = 0.8
        else:
            x_est0 = 0.2
########        print("Failing Case")
        x_sol, fval = gridsearch(obj_err, x_est0, 0.01,
                                n_iter = 7, r_shrink_mesh = 0.4)
        if fval < 1E-5:
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)    
            return [q1,q2], fval
        else:
            x_est0 = x_sol
            x_sol, fval = gridsearch(obj_err, x_est0, 0.01,
                                n_iter = 10, r_shrink_mesh = 0.15)
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)    
            return [q1,q2], fval

# %%
# TEST CODES 
#[TEST 1] iso2pi
#[TEST 2] x2err
#[TEST 3] Optimization example
#[TEST 4] x2q
#[TEST 5] gridupdate
#[TEST 6] gridsearch
#[TEST 7] IAST_bi

# %%
## TEST CODE 1: iso2pi

# %%
# TESTING "iso2pi" function
if __name__ == '__main__':
    qm1 = 3
    b1 = 0.5
    def iso_test(P):
        numer = qm1*b1*P
        denom = 1+b1*P
        q = numer/denom
        return q

    P_o_test = 5
    pi_ov_RT_test05 = iso2pi(iso_test, P_o_test, 5)
    pi_ov_RT_test10 = iso2pi(iso_test, P_o_test, 10)
    pi_ov_RT_test20 = iso2pi(iso_test, P_o_test, 20)
    pi_ov_RT_test30 = iso2pi(iso_test, P_o_test, 30)
    print('pi05=',pi_ov_RT_test05)
    print('pi10=',pi_ov_RT_test10)
    print('pi20=',pi_ov_RT_test20)
    print('pi30=',pi_ov_RT_test30)

# %%
## TEST CODE 2: x2err
# %%
# TESTING "x2err" function
if __name__ == "__main__":
    qm1 = 2
    qm2 = 1
    b1 = 0.2
    b2 = 0.4
    def iso_test1(P):
        numer = qm1*b1*P
        denom = 1+b1*P
        q = numer/denom
        return q
        
    def iso_test2(P):
        numer = qm2*b2*P
        denom = 1+b2*P
        q = numer/denom
        return q

    y1_test = 0.5
    P_test = 2
    x_est_test = 0.3
    mse_test = x2err_reg(iso_test1, iso_test2, 
                    y1_test, P_test, x_est_test)
    print(mse_test)

# %%
## TEST CODE 3: Optimization example
# %%

# Optimizing test with x2err_reg
if __name__ == "__main__":
    y1_test = 0.5
    P_test = 2
    err_test = lambda x: x2err_reg(iso_test1, iso_test2,
                            y1_test, P_test, x[0])
    x_est0 = [0.3]
    opt_res = minimize(err_test, x_est0)
    print(opt_res)

# %%
## TEST CODE 4: x2q

# %%
## TESTing x2q
if __name__ == '__main__':
    y1_test = 0.5
    P_test = 2
    err_test = lambda x: x2err_reg(iso_test1, iso_test2,
                            y1_test, P_test, x[0])
    x_est0 = [0.3]
    opt_res = minimize(err_test, x_est0)
    x1_sol_test = opt_res.x[0]
    q1_test, q2_test = x2q(iso_test1, iso_test2, 
                        y1_test, P_test, x1_sol_test)

    print('q1 = ')
    print(q1_test)
    print('q2 = ')
    print(q2_test)

# %%
## TEST CODE 5: gridupdate
# %%
if __name__ == "__main__":
    gridupdate(err_test, 0.1, 0.01, 20)

# %%
## TEST CODE 
# %%
if __name__ == "__main__":
    x_min_grid, fun_val = gridsearch(err_test, 0.4, 0.01)
    print('[Result: gridsearch 7 iterations]')
    print('x=', '{0:.5f}'.format(x_min_grid))

# %%
# TEST 6: IAST_bi
# %%
if __name__ == '__main__':
    # TEST IAST_bi
    y1_test = 0.1
    P_test = 2
    [q1,q2], fval_test = IAST_bi(iso_test1, iso_test2,
                                y1_test, P_test)

    print('q1 = ')
    print(q1)
    print('q2 = ')
    print(q2)
    print('pi_error = ')
    print(fval_test)
# %%
