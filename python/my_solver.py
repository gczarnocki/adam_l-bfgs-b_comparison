import scipy.optimize


def l_bfgs_b(fun, x0):
	'''L-BFGS-B'''
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