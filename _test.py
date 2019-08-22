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
        q_in = 1
        p_in = 0
        mu = np.random.randn(k) * 2
        # mu = None
    else:
        k = 8
        T = 5000
        q_in = 3
        p_in = 1
        mu = np.random.randn(k) * 2
        # mu = None

    # sigma = np.array([4, 0.8, 0.8, 1]).reshape(k, k)
    sigma = gen_random_pos_symmetric(k)

    phi2 = -gen_stable_model(p_in, k)
    theta_ = to_invertible(np.random.randn(q_in))
    """
    theta2 = zeros((k, q_in*k))
    for i in range(q_in):
        theta2[:, i*k:(i+1)*k] = eye(k) * theta_[i]
    """
    theta2 = concatenate([eye(k) * theta_[i] for i in range(q_in)], axis=1)
    
    vs = VARMA_sim(
        nobs=T, arlags=None, malags=None, cnst=mu,
        phi=phi2, theta=theta2, skip=2000, sigma=sigma)
    X = vs.series

    p = p_in
    q = q_in
    
    trend = mu is not None
    ve = VARsMA_Estimator(X)
    ve.setEstimationStructure(p, trend)
    t0 = np.random.randn(q)
    theta0 = to_invertible(t0)
   
    A = np.array([-1., -1., -1., -1, 1, 3, 1, -1, 1]).reshape(3, 3)
    ub = np.array([1., 3., 1.])
    lc = LinearConstraint(A, -np.inf, ub, keep_feasible=True)

    """
    svz = {'X': X, 'theta0': theta0, 'theta2': theta_, 'mu': mu, 'phi2': phi2}
    np.savez_compressed('/tmp/trends.npz', **svz)
    """
    def cons_f(x):
        return -x[0] * x[2] + x[1] + x[2]*x[2]

    def cons_J(x):
        return np.array([-x[2], 1, -x[0] + 2*x[2]])

    def cons_H(x, v):
        return v[0] * np.array(
            [0, 0, -1, 0, 0, 0, -1, 0, 2]).reshape(3, 3)

    nlc = NonlinearConstraint(
        cons_f, -np.inf, 1,
        jac=cons_J, hess=cons_H, keep_feasible=True)

    def f(theta):
        ve.calc(theta, check_stable=False)
        return ve.LLK

    def jf(theta):
        return ve.grLLK
    
    from VARsMA import VARsMA_Estimator_adjust_convol
    old_ve = VARsMA_Estimator_adjust_convol(X)
    old_ve.setEstimationStructure(p, trend)
    old_ve.calc(theta0)
    print(old_ve.LLK)
    ve.calc(theta0)
    print(ve.LLK)
    print(old_ve.grLLK)
    print(ve.grLLK)
    success = False
    while not success:
        try:
            theta0 = to_invertible(np.random.randn(q))
            ret = minimize(f, theta0, method='trust-constr',
                           jac=jf, constraints=[lc, nlc])
            success = True
        except Exception as e:
            print(e)
            pass

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


def _test_predict():
    np.random.seed(0)
    k = 4
    T = 5000
    T_F = 1000
    q_in = 3
    p_in = 1
    mu = np.random.randn(k) * 2

    sigma = gen_random_pos_symmetric(k)

    phi2 = -gen_stable_model(p_in, k)
    theta_ = to_invertible(np.random.randn(q_in))
    theta2 = zeros((k, q_in*k))
    for i in range(q_in):
        theta2[:, i*k:(i+1)*k] = eye(k) * theta_[i]
    theta2 = concatenate([eye(k) * theta_[i] for i in range(q_in)], axis=1)
    # train on rolling windows:
    vs = VARMA_sim(
        nobs=T+T_F, arlags=None, malags=None, cnst=mu,
        phi=phi2, theta=theta2, skip=2000, sigma=sigma)
    X = vs.series[:T, :]

    p = p_in
    q = q_in
    
    trend = mu is not None
    ve = VARsMA_Estimator(X)
    ve.setEstimationStructure(p, trend)
    t0 = np.random.randn(q)
    theta0 = to_invertible(t0)
    ve.fit(p, q, theta0, trend)
    print(sigma)
    print(ve.Omega)
    print((ve.residuals.T @ ve.residuals) / (ve.n - ve.p))
    ve.predict(2)

    
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
    

def fit_one(k, p, q, with_trend, X):
    ve = VARsMA_Estimator(X)
    ve.setEstimationStructure(p, with_trend)
    theta0 = to_invertible(np.random.randn(q))
    ret = ve.fit(p, q, theta0, with_trend)
    if (ret is not None) and ret['success']:
        return ve
    else:
        print("failed")
        print(ret)
        return None


def predict_one(k, p, q, model, with_trend, X, X_o, t_ahead):
    """
    forecasting t_ahead periods out
    appending and do analysis
    """
    T_o = X_o.shape[0]
    k = X_o.shape[1]
    n_sims = T_o - t_ahead
    pred = np.full((n_sims, t_ahead, k), np.nan)
    for i in range(n_sims):
        X_n = concatenate([X, X_o[:i, :]])
        pred[i, :, :] = model.predict(t_ahead, X_n)
    return pred


def sim_one_set(n_obs, k, p, q, with_trend):
    sigma = gen_random_pos_symmetric(k)

    phi = -gen_stable_model(p, k)
    theta_ = to_invertible(np.random.randn(q))

    """
    for i in range(q):
        theta = zeros((k, q*k))
        theta[:, i*k:(i+1)*k] = eye(k) * theta_[i]
    """
    theta = concatenate([eye(k) * theta_[i] for i in range(q)], axis=1)
    if with_trend:
        mu = np.random.randn(k) * 3
        return VARMA_sim(
            nobs=n_obs, arlags=None, malags=None,
            cnst=mu, phi=phi, theta=theta,
            skip=2000, sigma=sigma), phi, theta_, mu, sigma
    else:
        return VARMA_sim(
            nobs=n_obs, arlags=None, malags=None,
            cnst=None, phi=phi, theta=theta,
            skip=2000, sigma=sigma), phi, theta_, None, sigma


def _double_check(ve):
    grids = 100
    x_ = (np.arange(2*grids+1) / float(grids) - 1)
    y_ = np.full_like(x_, np.nan)
    z_ = np.full_like(x_, np.nan)
    for i in range(x_.shape[0]):
        ve.calc(np.array([x_[i]]))
        if ve.LLK < 1000000.:
            y_[i] = ve.LLK
            z_[i] = ve.grLLK
    import pandas as pd
    adict = {'x': x_, 'y': y_, 'z': z_}
    np.savez_compressed('/data/share/results/VARsMA/a.npz', **adict)
    df = pd.DataFrame(adict)
    good_df = df.loc[~pd.isnull(df.z)]
    

def dump_test_dict():
    from os.path import join
    np.random.randn(0)
    T = 5000
    T_OUT = 20
    ks = [2, 8, 12, 15]
    ps = [0, 1, 2, 3, 4]
    qs = [1, 2, 3, 4]
    # qs = [4]
    t_ahead = 5
    rpts = 4

    # T_OUT = 20
    # ks = [2]
    # ps = [0]
    # qs = [2]
    # t_ahead = 5
    # rpts = 5

    result_dir = '/data/share/results/VARsMA.1'
    outs = {}
    tpl = '%s_%s_%s_%s'
    for k in ks:
        for p in ps:
            for q in qs:
                for i in range(rpts):
                    good_X = False
                    kk = tpl % (k, p, q, i)
                    good_fit = False
                    cnt_fit = 0
                    print("doing %s" % kk)
                    while (cnt_fit < 20) and (not good_fit):
                        cnt = 0
                        while (cnt < 100) and (not good_X):
                            with_trend = i > 0
                            vs, Phi, theta_, mu, sigma = sim_one_set(
                                T+T_OUT, k, p, q, with_trend)
                            X = vs.series[:T, :]
                            X_o = vs.series[T:, :]
                            max_coeff = np.max(X.T @ X / T)
                            good_X = max_coeff < 3e7
                            print(max_coeff)
                            print(good_X)
                            cnt += 1
                        cnt_fit += 1
                        fit = fit_one(
                            k, p, q, with_trend, X)
                        good_fit = fit is not None
                        
                    out_file = join(result_dir, 'out_%s' % kk)
                    if fit is not None:

                        preds = predict_one(
                            k, p, q, fit, with_trend, X, X_o, t_ahead)
                        outs = {'Phi': Phi, 'theta': theta_, 'mu': mu,
                                'sigma': sigma,
                                'Phi_out': fit.Phi, 'theta_out': fit.Theta,
                                'mu_out': fit.mu, 'sigma_out': fit.Omega,
                                'residuals': fit.residuals,
                                'X': X, 'X_o': X_o, 'preds': preds}
                    else:
                        print('Failed for %s' % kk)
                        outs = {'Phi': Phi, 'theta': theta_, 'mu': mu,
                                'sigma': sigma,
                                'Phi_out': None, 'theta_out': None,
                                'mu_out': None, 'sigma_out': None,
                                'residuals': None,
                                'X': None, 'X_o': None, 'preds': None}
                    np.savez_compressed(out_file, **outs)


def rerun_predicts():
    """ Fix a bug in predict. Now rerun
    """
    import os
    data_dir = '/data/share/results/VARsMA'
    files = [a for a in os.listdir(data_dir)
             if a.startswith('out_') and a.endswith('.npz')]
    files = sorted(files, key=lambda f: [int(jj) for jj in f[4:-4].split('_')])
    t_ahead = 5
    # found = False
    for f in files:
        # print("Doing %s" % f)
        done_list = ['out_8_3_4_0.npz',
                     'out_8_3_4_2.npz',
                     'out_12_1_1_1.npz',
                     'out_12_2_3_0.npz',
                     'out_12_3_4_2.npz',
                     'out_15_1_4_3.npz',
                     'out_15_2_3_3.npz']
        if f in done_list:
            continue
        # if f.startswith('out_2') or f.startswith('out_8')\
            # or f.startswith('out_12_0') or f.startswith('out_12_1'):
        if f.startswith('out_2') or f.startswith('out_8')\
           or f.startswith('out_12') or f.startswith('out_15_0') or f.startswith('out_15_1'):
            continue
        else:
            print('Doing %s' % f)
        a = np.load(os.path.join(data_dir, f))
        theta_ = a['theta_out']
        if (theta_ is None) or (len(theta_) == 0):
            print("No fit - reruns ? %" % f)
            continue
        k, p, q, i = tuple([int(jj) for jj in f[4:-4].split('_')])
        with_trend = i > 0
        if p != 0:
            Phi = a['Phi_out']
        else:
            Phi = None

        X = a['X']
        X_o = a['X_o']

        sigma = a['sigma']

        fit = VARsMA_Estimator(X)

        fit.setEstimationStructure(p, with_trend)
        fit.calc(theta_)
        fit.calc_residuals()
        if with_trend:
            try:
                mu = a['mu_out']
            except Exception:
                mu = None
        else:
            mu = None

        # Phi = fit.Phi
        # mu = fit.mu
        preds = predict_one(
            k, p, q, fit, with_trend, X, X_o, t_ahead)
        outs = {'Phi': Phi, 'theta': theta_, 'mu': mu,
                'sigma': sigma,
                'Phi_out': fit.Phi, 'theta_out': fit.Theta,
                'mu_out': fit.mu, 'sigma_out': fit.Omega,
                'residuals': fit.residuals,
                'X': X, 'X_o': X_o, 'preds': preds}
        out_file = os.path.join(data_dir, '%s' % f)
        np.savez_compressed(out_file, **outs)


def mv_files():
    import os
    data_dir = '/data/share/results/VARsMA'
    files = [a for a in os.listdir(data_dir)
             if a.startswith('out_out_') and a.endswith('.npz')]

    # files = sorted(files, key=lambda f: [int(jj) for jj in f[4:-4].split('_')])
    for f in files:
        s = 'mv %s/%s %s/%s' % (data_dir, f, data_dir, f[4:])
        print(s)
    
                    
if __name__ == '__main__':
    dump_test_dict()
