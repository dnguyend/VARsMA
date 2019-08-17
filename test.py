import numpy as np
from numpy import zeros, eye, concatenate
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
from types import SimpleNamespace

from utils import VARMA_sim, gen_stable_model, gen_rc_roots, HSRC, jac_HSRC
from utils import gen_random_pos_symmetric, to_invertible
from VARsMA import VARsMA_Estimator, optimize_model


def _gen_test():
    np.random.seed(0)    
    if False:
        k = 2
        T = 5000
        q_in = 3
        p_in = 1
    else:
        k = 10
        T = 5000
        q_in = 3
        p_in = 2

    # sigma = np.array([4, 0.8, 0.8, 1]).reshape(k, k)
    sigma = gen_random_pos_symmetric(k)

    phi2 = -gen_stable_model(p_in, k)
    theta_ = to_invertible(np.random.randn(q_in))
    theta2 = zeros((k, q_in*k))
    for i in range(q_in):
        theta2[:, i*k:(i+1)*k] = eye(k) * theta_[i]
    theta2 = concatenate([eye(k) * theta_[i] for i in range(q_in)], axis=1)
    
    vs = VARMA_sim(
        nobs=T, arlags=None, malags=None, cnst=None,
        phi=phi2, theta=theta2, skip=2000, sigma=sigma)
    X = concatenate([vs.init_series, vs.series])
    ve = VARsMA_Estimator(X)
    p = p_in
    q = q_in
    trend = False
    ve.setEstimationStructure(p, trend)
    theta0 = to_invertible(np.random.randn(q))
    
    A = np.array([-1., -1., -1., -1, 1, 3, 1, -1, 1]).reshape(3, 3)
    ub = np.array([1., 3., 1.])
    lc = LinearConstraint(A, -np.inf, ub, keep_feasible=True)
    
    def cons_f(x):
        return -x[0] * x[2] + x[1] + x[2]*x[2]

    def cons_J(x):
        return np.array([-x[2], 1, -x[0] + 2*x[2]])

    def cons_H(x, v):
        return v[0] * np.array(
            [0, 0, -1, 0, 0, 0, -1, 0, 2]).reshape(3, 3)

    nlc = NonlinearConstraint(
        cons_f, -np.inf, 1,
        jac=cons_J, hess=cons_H)

    def f(theta):
        ve.calc(theta)
        return ve.LLK

    def jf(theta):
        return ve.grLLK
    
    ret = minimize(f, theta0, method='trust-constr',
                   jac=jf, constraints=[lc, nlc])

    print(ret)
    ret_opt = optimize_model(
        X, theta0, p, trend, constraints=[lc, nlc])
    print(ret_opt)

    ret_opt_no_constr = optimize_model(X, theta0, p, trend)
    print(ret_opt_no_constr)

    """
    This failed but hopefully we can make to work
    with more research:
    """
    from utils import _optimize_model_HS
    ret_hs, x_opt = _optimize_model_HS(X, theta0, p, trend)
    print(ret_hs)


def test_HS_relation():
    np.random.seed(0)    
    
    k = 2
    T = 32
    q_in = 3
    p_in = 1

    # sigma = np.array([4, 0.8, 0.8, 1]).reshape(k, k)
    sigma = gen_random_pos_symmetric(k)

    phi2 = -gen_stable_model(p_in, k)
    theta_ = to_invertible(np.random.randn(q_in))
    theta2 = zeros((k, q_in*k))
    for i in range(q_in):
        theta2[:, i*k:(i+1)*k] = eye(k) * theta_[i]
    theta2 = concatenate([eye(k) * theta_[i] for i in range(q_in)], axis=1)
    
    vs = VARMA_sim(
        nobs=T, arlags=None, malags=None, cnst=None,
        phi=phi2, theta=theta2, skip=2000, sigma=sigma)
    X = concatenate([vs.init_series, vs.series])
    ve = VARsMA_Estimator(X)
    p = p_in
    q = q_in
    trend = False
    ve.setEstimationStructure(p, trend)

    n_test = 1
    sim_val = SimpleNamespace(llk=None, gr_llk=None)
    sim_val_inv = SimpleNamespace(llk=None, gr_llk=None)
    
    for ii in range(n_test):
        theta = np.random.randn(q)
        ve.calc(theta, check_stable=False)
        sim_val.llk = -ve.LLK
        sim_val.gr_llk = -ve.grLLK
        rts, stable_map, all_stable = gen_rc_roots(theta)
        
        if not all_stable:
            stable_theta = HSRC(rts, stable_map)
            j_hs = np.real(jac_HSRC(rts, stable_map))
            ve.calc(stable_theta, check_stable=False)
            sim_val_inv.llk = -ve.LLK,
            sim_val_inv.gr_llk = -ve.grLLK @ j_hs
            print('diff llk %f' % (sim_val.llk - sim_val_inv.llk))
            print('diff gr_llk %s' % str(sim_val.gr_llk - sim_val_inv.gr_llk))
        else:
            print('stable %s' % str(theta))


def _test_HS_1():
    use_R_data = False
    k = 2
    if use_R_data:
        """ read data dump from R to compare
        """
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        readRDS = robjects.r['readRDS']
        X = np.array(readRDS('/tmp/wenx2.rds')).reshape(-1, k)
    else:
        np.random.seed(0)
        X = np.random.randn(32, k)
        # X = matrix(rep(1,2*20), 20,2)
    p = 2
    rc_roots = SimpleNamespace(real=np.array([1.01, .8]),
                               cplx=np.array([0.1+1.00j, -0.76+0.62j]))

    hsmap = SimpleNamespace(
        real=np.array([0], dtype=int), cplx=np.array([0], dtype=int))
    tt1 = HSRC(rc_roots, hsmap)
    r2, h1, _ = gen_rc_roots(tt1)
    tt = HSRC(r2, hsmap)

    ve = VARsMA_Estimator(X=X)
    ve.setEstimationStructure(p=p, trend=False)
    ve.calc(tt, check_stable=False)
    j_hs = np.real(jac_HSRC(r2, hsmap))
    # trH = grIRsq(IRMap, j.v, rootRC)

    # q = Theta.shape[0]
    
    # tt2 = HSRC(rc_roots, hsmap)
    ve_inv = VARsMA_Estimator(X=X)
    ve_inv.setEstimationStructure(p=p, trend=False)
    ve_inv.calc(tt1, check_stable=False)
    # test case: 1
    # print(ve.grLLK - ve_inv.grLLK @ j_hs)
    # print(ve.grLLK - j_hs @ ve_inv.grLLK)
    # print(j_hs @ ve.grLLK - ve_inv.grLLK)
    print(ve.grLLK @ j_hs - ve_inv.grLLK)
    hh = 1e-6
    q = tt.shape[0]
    llks = zeros(q)
    base_llk = ve.LLK
    base_gr_llk = ve.grLLK
    for ii in range(tt.shape[0]):
        ttx = tt.copy()
        ttx[ii] += hh
        ve.calc(ttx, check_stable=False)
        llks[ii] = (ve.LLK - base_llk) / hh
    print(base_gr_llk)
    print(llks)

    llks_inv = zeros(q)
    base_llk_inv = ve_inv.LLK
    base_gr_llk_inv = ve_inv.grLLK
    for ii in range(tt.shape[0]):
        ttx = tt1.copy()
        ttx[ii] += hh
        ve_inv.calc(ttx, check_stable=False)
        llks_inv[ii] = (ve_inv.LLK - base_llk_inv) / hh
    print(base_gr_llk_inv)
    print(llks_inv)

    
def _testSimpleGradientTest():
    from types import SimpleNamespace
    from utils import HSRC, gen_rc_roots, jac_HSRC
    use_R_data = False
    k = 2
    if use_R_data:
        """ read data dump from R to compare
        """
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        readRDS = robjects.r['readRDS']
        X = np.array(readRDS('/tmp/x1.rds')).reshape(-1, k)
    else:
        np.random.seed(0)
        X = np.random.randn(32, k)
        # X = matrix(rep(1,2*20), 20,2)
    p = 2

    # ve = VARsMA_Estimator(X=X)

    # Since the growith in XTheta is exponential in T
    # it is better to test with roots that is not too far
    # away, at least root^T or (1/root)^T is not too big
    # After that is r_i^2
    Theta = np.array([1.40, 0.98, 0.40])
    rc_roots, _, _ = gen_rc_roots(Theta)

    ve = VARsMA_Estimator(X=X)
    ve.setEstimationStructure(p=p, trend=False)
    ve.calc(Theta, check_stable=False)
    hsmap = SimpleNamespace(real=np.array([0], dtype=int),
                            cplx=np.array([0], dtype=int))
    
    j_hs = np.real(jac_HSRC(rc_roots, hsmap))
    
    Theta2 = HSRC(rc_roots, hsmap)
    ve_inv = VARsMA_Estimator(X=X)
    ve_inv.setEstimationStructure(p=p, trend=False)
    ve_inv.calc(Theta2, check_stable=False)
    # test case: 1
    print(ve.grLLK - ve_inv.grLLK @ j_hs)
    # check gr of covXTheta
    print(ve_inv.grCovThetaTheta)
    

