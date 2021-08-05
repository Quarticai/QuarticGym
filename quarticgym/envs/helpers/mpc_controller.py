from casadi import SX, mtimes, inf, vertcat, nlpsol
import numpy as np


# simulate model for one time step and get next states and output
def sim_model(x, u, A, B, C, D):
    # x = state
    # u =  input.
    u = u / 1000  # this has been scaled in the optimization problem. hence the division by 1000 to unscale it
    xplus = np.matmul(A, x) + np.matmul(B, u)  #
    y = np.matmul(C, x) + np.matmul(D, u)
    return xplus, y


# MPC/EMPC controller
def mpc_controller(x0, N, Nx, Nu, uss, ur, yr, A, B, C, D):
    # controller weights. different values may lead to different controller performance or cause instability
    R = np.identity(4) * 0.1  # weight on inputs
    Q = 10  # weight on output

    # start with empty NLP
    w = []  # decision variables
    w0 = []  # decision variables initial guess
    lbw = []  # lower bound on decision variables
    ubw = []  # upper bound on decision variables
    J = 0  # initial cost
    g = []  # state constraint
    lbg = []  # lower bound on state constraint
    ubg = []  # upper bound on state constraint

    # formulate NLP
    xk = x0  # fix initial state from state estimator

    for k in range(N):
        Uk = SX.sym('U_' + str(k), Nu)  # create input variable at time step k
        w.append(Uk)  # store in decision variable list
        lbw.append(- uss * 1000)  # lower bound on inputs at time k
        ubw.append(uss * 1000)  # upper bound on inputs at time k
        w0.append([0] * Nu)  # initial input guess

        # simulate model to get x(k+1) and y(k)
        xk, yk = sim_model(xk, Uk, A, B, C, D)
        # calculate the cost
        J += Q * (yk - yr) ** 2 \
             + mtimes((Uk - ur).T, mtimes(R, (Uk - ur)))
        # add state constraints. currently the states are unconstrained since the
        # identified model states have no physical meaning
        g.append(xk)
        lbg.append([-inf] * Nx)
        ubg.append([inf] * Nx)

    # Create an NLP solver
    opts = {}
    opts["verbose"] = False
    opts["ipopt.print_level"] = 0
    opts["print_time"] = 0
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(
        x0=vertcat(*w0),
        lbx=vertcat(*lbw),
        ubx=vertcat(*ubw),
        lbg=vertcat(*lbg),
        ubg=vertcat(*ubg)
    )
    # get solution
    w_opt = sol['x']
    u_opt = w_opt[0:Nu]
    return u_opt.full().ravel()
