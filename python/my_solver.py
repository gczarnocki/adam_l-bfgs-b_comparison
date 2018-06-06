import scipy.optimize
import math
import numpy as np


def l_bfgs_b(fun, x0):
    scipy.optimize.minimize(
        fun,
        x0,
        args=(),
        method='L-BFGS-B',
        jac=None,
        bounds=None,
        tol=None,
        callback=None,
        options={
            'disp': None,
            'maxls': 20,
            'iprint': -1,
            'gtol': 1e-05,
            'eps': 1e-08,
            'maxiter': 15000,
            'ftol': 2.220446049250313e-09,
            'maxcor': 10,
            'maxfun': 15000}
    )


def adam(fun, x0, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    def grad_func(x):
        return scipy.optimize.approx_fprime(np.array(x), fun, epsilon=1e-6)

    theta_0 = x0  # initialize the vector
    m_t = 0
    v_t = 0
    t = 0

    conv_epsilon=1e-8

    while 1:  # till it gets converged
        t += 1
        g_t = grad_func(theta_0)  # computes the gradient of the stochastic function
        m_t = beta_1 * m_t + (1 - beta_1) * g_t  # updates the moving averages of the gradient
        v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)  # updates the moving averages of the squared gradient
        m_cap = m_t / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
        v_cap = v_t / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates
        theta_0_prev = theta_0
        theta_0 = theta_0 - (alpha * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters

        if t % 1e6 == 0:
            conv_epsilon = conv_epsilon * 10

        if np.abs(fun(theta_0) - fun.best_observed_fvalue1).sum() < conv_epsilon:
            break

        if np.abs(theta_0 - theta_0_prev).sum() < conv_epsilon:
            break
