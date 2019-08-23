import numpy as np
from numpy import eye, concatenate
# from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
# from types import SimpleNamespace

from .utils import VARMA_sim, gen_stable_model
from .utils import gen_random_pos_symmetric, to_invertible
from .VARsMA import VARsMA_Estimator
    

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

    
def dump_tests(result_dir):
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

                    
if __name__ == '__main__':
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    result_dir = '/data/share/results/VARsMA.2'
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    dump_tests(result_dir)
