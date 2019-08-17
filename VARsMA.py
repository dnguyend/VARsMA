import numpy as np
from numpy import log, pi, zeros, ones, concatenate, roots
from numpy import diag, flip, eye, diagonal, full, nan
from numpy import poly1d, prod
from numpy.linalg import solve, det, cholesky

from scipy.linalg import solve_triangular
from utils import inverse_poly, mat_convol, to_invertible


m_BIG_M_LLK = 1e6
m_SMALL_SQRT = np.finfo(float).eps
m_SMALL_ERR = 1e-6


def mult_by_K1(lbd, cholUp, v1):
    """this is a "twisted" version of t(v1)v1
    so result should be close to t(v1)v1 if
    if theta is small
    """
    p1 = v1.T @ v1
    p2 = v1.T @ lbd
    p3 = solve_triangular(cholUp, p2.T, trans=1)
    return p1 - p3.T @ p3


def mult_by_K2(lbd, cholUp, v1, v2):
    """this is a "twisted" version of t(v1)v2
    """
    p1 = v1.T @ v2
    p2 = v2.T @ lbd
    p2a = v1.T @ lbd
    p3 = solve_triangular(cholUp, p2.T, trans=1)
    p3a = solve_triangular(cholUp, p2a.T, trans=1)
    return p1 - p3a.T @ p3


def generate_Toeplitz(q, n, trans=True):
    """Generate Toeplitz matrix for polynomial q
    q is without constant term
    """
    ret = zeros((n, n), dtype=float)
    for i in range(n):
        ml = min(q.shape[0], n-i)
        ret[i, i:ml+i] = q[:ml]

    if trans:
        return ret
    return ret.T


def AdjustConvol(xconvfull, x, vec, T, off):
    out = xconvfull[off:off+T, :].copy()
    if off > 0:
        endI = min(T, vec.shape[0])
        for i in range(endI):
            if i+off >= vec.shape[0]:
                if i < vec.shape[0] - 1:
                    out[i, :] -= flip(vec[i+1:]) @\
                        x[i+off-vec.shape[0]:off-1, :]
            else:
                out[i, :] -= flip(vec[i+1:i+1+off]).transpose() @ x[:off, :]
    return out


class VARsMA_Estimator(object):
    """ object to calculate likelihood function
    """
    def __init__(self, X):
        """ Constructor. X is the data matrix
        """
        self.X = X
        self.LLK = None
        self.Omega = None
        self.Phi = nan
        self.Tklog = None
        self.Theta = None
        self.XThetaLag = None
        self.grOmega = None
        self.grLogDetOmega = None
        self.grLLK = None
        self.grLogKbar = None
        self.Kbar = None
        self.grKbar = None
        self.grDetKbar = None
        self.has_lag = None

    def setEstimationStructure(self, p, trend):
        """ Decide if we have lag and trend
        """
        self.p = p
        self.trend = trend

    def calcGrQuad(self, A, grA, B, grB, q):
        """gr of t(A)B^[-1]A knowing grA, grB
        grA and grB are list of matrices the same length
        """
        px1 = solve(B, A)

        ret = full((px1.shape[1], px1.shape[1], q), nan)
        for i in range(q):
            px1a = px1.T @ (grB[:, :, i] @ px1)
            px2 = A.T @ solve(B, grA[:, :, i])
            ret[:, :, i] = px2 + px2.T - px1a

        return ret

    def gr_mult_by_K1(self, A, grA, T, p, q):
        """gr of t(A)KA. K is not given explicitly but through lbd and Kbar
        grA is not given explicitly but through
        a derivatives of theta applied on large X
        K = I_T - lamda K_bar lbd prime
        parts:
        p1: t(A) K grA  and t(p1)
        p2: t(A) gr(A) and transpose
        p3a : - t(A lbd) Kbar^-1 grKbar  Kbar^-1 (t(lbd) A)
        small part:
           p3s: t(lbd) A
           p3: Kbar^-1 (t(lbd) A) = Kbar^{-1} p3s

        p4a: (t(A) lbd) Kbar^-1 (t(grlbd) A) and transposed
        small part:
           p4=(t(grlbd) A)
        p1+t(p1)
       
        Usage note: if A is XTheta we should pass
        data is between p and T+p, see
        gr_Mult_By_K1(self, self.XTheta[(p+1):n]
        grA data: for the ith derivative: first i rows are zero
        the next T+i rows are between p and T+p -i
        use Kbar, grKbar, lbd, grLbd
        """
        ret = full((A.shape[1], A.shape[1], q), nan)
        adim = A.shape[1]
        for i in range(q):
            # this is prototype - the Cpp code is more optimal
            # just interested in getting the number correct.
            matGrA = concatenate(
                [np.zeros((i+1, adim)),
                 grA[:(T-i-1), :]], axis=0)
            p1 = mult_by_K2(self.Lbd, self.cholKb, A, matGrA)
            p3s = self.Lbd.T @ A
            p3 = solve(self.Kbar, p3s)
            p3a = -p3.T @ (self.grKbar[:, :, i] @ p3)
            p4 = self.grLbd[:, :, i].T @ A
            p4a = p3s.T @ solve(self.Kbar, p4)
            ret[:, :, i] = p1 + p1.T - p3a - p4a - p4a.T
        return ret

    def gr_mult_by_K2(self, A, grA, B, grB, T, p, q):
        """ gr of t(A)KB. K is not given explicitly but through lbd and Kbar
        K = I_T - lamda K_bar lbd prime
        p1:  t(A)K grB
        p1a: t(grA) K B
        Middle part: A (grK)B = A(- gr(Lbd Kbar Lamda') B. 2 sub parts:
        p3b: - t(A) lbd  Kbar^-1 grKbar  Kbar^-1 (t(lbd) B)
        p4a  t(A) lbd Kbar^-1 (t(grlbd) B)
        p5 t(grlbd A ) Kbar^-1 (t(lbd) B)
        Use: Kbar, grKbar, lbd, grLbd ,
        dim: cols A times col B.
        Case: XthetaLag K Xtheta: expect pk+(trend?) times pk
        """

        ret = full((A.shape[1], B.shape[1], q), nan)
        # n = A.shape[0]
        adim = A.shape[1]
        dimb2 = B.shape[1]
        if dimb2 is None:
            dimb2 = 1
        for i in range(q):
            matGrA = concatenate([
                zeros((i+1, adim), dtype=float),
                grA[:T-i-1, :].reshape(T-i-1, adim)], axis=0)
            matGrB = concatenate(
                [zeros((i+1, dimb2), dtype=float),
                 grB[:(T-i-1), :].reshape(T-i-1, dimb2)], axis=0)

            p1 = mult_by_K2(self.Lbd, self.cholKb, A,  matGrB)
            p1a = mult_by_K2(self.Lbd, self.cholKb, matGrA, B)

            p3l = A.T @ self.Lbd
            p3r = self.Lbd.T @ B
            p3 = solve(self.Kbar, p3r)
            p3a = solve(self.Kbar, p3l.T)  # result would be Kbar^{-1}Lbd^t A
            # -t(A)Lbd Kbar^-1 grKbar Kbar Lbd^t B:
            p3b = -p3a.T @ (self.grKbar[:, :, i] @ p3)

            p4 = self.grLbd[:, :, i].T @ B
            p4a = p3l @ solve(self.Kbar, p4)  # t(A)Lbd Kbar^{-1} grLbd' B
            p5 = self.grLbd[:, :, i].T @ A
            p5a = solve(self.Kbar, p5).T @ p3r
            ret[:, :, i] = p1 + p1a - p3b - p4a - p5a
        return ret

    def setThetaLbd(self, theta, T, q, p):
        """ Set Theta and Lbd in the environment
        """
        q = theta.shape[0]
        self.inv_Theta = inverse_poly(theta, T+p)

        self.Theta_St = generate_Toeplitz(flip(theta), q)
        self.Lbd = mat_convol(
            concatenate(
                [self.Theta_St,
                 zeros((T-q, q))]), self.inv_Theta)
        self.Kbar = self.Lbd.T @ self.Lbd + eye(q)

    def setGrThetaLbd(self, q, T):
        """Gradient of Theta and Lbd
        """
        Theta_1 = np.concatenate(
            [np.array([1.]), self.Theta])
        theta2 = (poly1d(Theta_1) * poly1d(Theta_1)).coeffs[1:]

        if (theta2.shape[0] < 2):
            self.inv_Theta2 = zeros((T+self.p))
        else:
            self.inv_Theta2 = inverse_poly(
                theta2, T+self.p)

        self.grLbd = full((self.Lbd.shape[0], self.Lbd.shape[1], q), nan)
        self.grKbar = full(
            (self.Kbar.shape[0], self.Kbar.shape[1], q), nan)
        for i in range(q):
            # This is just d_i Theta^{-1} Theta_St
            self.grLbd[:, :, i] = zeros((T, q))
            self.grLbd[:, :, i][i+1:T, :] = mat_convol(
                concatenate(
                    [self.Theta_St,
                     zeros((T-q, q))]), -self.inv_Theta2)[:T-i-1, :]
            for j in range(i+1):
                self.grLbd[j:T, q+j-i-1, i] += self.inv_Theta[:T - j]

            self.grKbar[:, :, i] = self.grLbd[:, :, i].T @ self.Lbd
            self.grKbar[:, :, i] += self.grKbar[:, :, i].T

        self.grLogKbar = np.vectorize(
            lambda i: np.sum(diag(
                solve(self.Kbar, self.grKbar[:, :, i]))))(np.arange(q))

    def calcGrLLK(self, T, k, p, q):
        """
        Gradients and sizes:
        gradient is tensor with an additional dimension
        so grDet are vectors
        """

        # compute gr for Theta, Lbd, Kbar, logKbar ( which then gives detKbar
        self.setGrThetaLbd(q, T)
        self.XThetaFull2 = mat_convol(self.XThetaFull, -self.inv_Theta)
        self.XThetaT2 = AdjustConvol(
            self.XThetaFull2, self.X, -self.inv_Theta2, T, p)
        self.XThetaLag2 = full((T, self.hSize), nan)

        if (self.trend):
            self.XThetaLag2[:, 0] = mat_convol(
                self.XThetaLag[:, 0].reshape(-1, 1), - self.inv_Theta)
        elif (self.p == 0):
            self.XThetaLag2[:, 0] = zeros((T, self.hSize))

        for i in range(self.p):
            self.XThetaLag2[:, self.bhidx+i*k:self.bhidx+(1+i)*k] =\
                AdjustConvol(
                    self.XThetaFull2, self.X, -self.inv_Theta2, T, p-i-1)
        
        # self.lbd, self.cholKb, self.XThetaLag
        # Kbar, cholKb, lbd, grlbd
        self.grCovThetaTheta = self.gr_mult_by_K1(
            self.XThetaT, self.XThetaT2, T, p, q)
        self.grOmega = full(
            (self.Omega.shape[0], self.Omega.shape[1], q), nan)

        if (self.has_lag):
            self.grCovXLagXLag = self.gr_mult_by_K1(
                self.XThetaLag, self.XThetaLag2, T, p, q)
            self.grCovXLagXTheta = self.gr_mult_by_K2(
                self.XThetaLag, self.XThetaLag2,
                self.XThetaT,
                self.XThetaT2, T, p, q)

            prj = self.calcGrQuad(
                self.covXLagXTheta,
                self.grCovXLagXTheta, self.covXLag, self.grCovXLagXLag, q)
            for i in range(q):
                self.grOmega[:, :, i] =\
                    (self.grCovThetaTheta[:, :, i] - prj[:, :, i]) / T
        else:
            for i in range(q):
                self.grOmega[:, :, i] = self.grCovThetaTheta[:, :, i] / T
        # grLogDetOmeta
        self.grLogDetOmega = zeros((q))
        for i in range(q):
            self.grLogDetOmega[i] = np.trace(
                solve(self.Omega, self.grOmega[:, :, i]))

        self.grLLK = T/2 * self.grLogDetOmega + k/2 * self.grLogKbar

    def calc(self, theta, with_gradient=True, check_stable=True, debug=False):
        """
        Model X = mu + Phi Xlags + e + Theta elags
        data is n *k dim, n = T+p. k = k
        """
        k = self.X.shape[1]
        if debug:
            print('theta=%s' % ','.join(theta.astype(str).tolist()))
        q = theta.shape[0]
        theta_1 = concatenate([np.array([1.]), theta])
        t_roots = roots(flip(theta_1))

        if (np.sum(np.abs(theta)) != 0) and\
           (check_stable and (np.where(np.abs(t_roots) < 1)[0].shape[0] > 0)):
            self.LLK = m_BIG_M_LLK
            self.Omega = full((k, k), nan)
            self.Phi = nan
            return self.LLK

        n = self.X.shape[0]
        p = self.p
        T = n-p
        self.n = n
        self.k = self.X.shape[1]
        self.Tklog = T*k/2.0*log(2*pi)
        self.Theta = theta
        self.setThetaLbd(theta, T, q, p)
        # compute for the whole length of $X, but use only from p+1 to n
        self.XThetaFull = mat_convol(self.X, self.inv_Theta)
        self.XThetaT = AdjustConvol(
            self.XThetaFull, self.X, self.inv_Theta, T, p)
        self.cholKb = cholesky(self.Kbar).T
        # compute for the whole length of $X, but use only from p+1 to n
        if debug:
            print("kbar=")
            print(self.Kbar)
            print("cholesky=")
            print(self.cholKb)
        smallE = np.where(np.abs(diag(self.cholKb)) < m_SMALL_ERR)[0]
        di = np.diag_indices(self.cholKb.shape[0])[0][smallE]
        self.cholKb[(di, di)] = m_SMALL_ERR
        self.covXTheta = mult_by_K1(
            self.Lbd, self.cholKb, self.XThetaT)
        self.detChol = prod(diagonal(self.cholKb))
        if self.trend:
            self.hSize = k*p+1  # size of lag matrix
            self.bhidx = 1  # begin index of the X, as opposed to the trend
        else:
            self.hSize = k*p
            self.bhidx = 0
        self.has_lag = self.trend or (p != 0)
        if not self.has_lag:
            self.Omega = self.covXTheta/T
            self.detOmega = det(self.Omega)
            if (self.detOmega <= 0):
                self.LLK = m_BIG_M_LLK
            else:
                self.LLK = self.Tklog + T/2.*log(self.detOmega) +\
                    k/2.*log(self.detChol)
            if debug:
                print(self.Omega)
                print(self.LLK)
            if with_gradient:
                self.calcGrLLK(T, k, p, q)
            # attr(wEnv$LLK, 'gradient') = wEnv$grLLK
            return self.LLK

        self.XThetaLag = full((T, self.hSize), nan)
        if self.trend:
            self.XThetaLag[:, 0] = mat_convol(ones(T, 1), self.inv_Theta)
        elif p == 0:
            self.XThetaLag[:, 0] = zeros(T, self.hSize)

        if p > 0:
            for i in range(p):
                self.XThetaLag[:, self.bhidx+i*k:i*k+k] =\
                    AdjustConvol(self.XThetaFull,
                                 self.X, self.inv_Theta, T, p-i-1)

                """
                self.XThetaLag[:, bhidx+i*k:i*k+k] =\
                    self.XTheta[p-i-1:n-i-1, :]
                self.XThetaLag2[:, bhidx+i*k:i*k+k] =\
                    self.XTheta2[p-i-1:n-i-1, :]
                self.XThetaLag3[:, bhidx+i*k:i*k+k] =\
                    self.XTheta3[p-i-1:n-i-1, :]
                """

        self.covXLag = mult_by_K1(
            self.Lbd, self.cholKb, self.XThetaLag)
        self.covXLagXTheta = mult_by_K2(
            self.Lbd, self.cholKb,
            self.XThetaLag, self.XThetaT)

        # same as
        # MM = lm.fit(XThetaLag,Xq[(p+1):n,])
        # should recover back sigma.
        self.Phi = solve(self.covXLag, self.covXLagXTheta)
        self.Omega = (self.covXTheta - (self.covXLagXTheta.T @ self.Phi)) / T
        self.detChol = prod(diag(self.cholKb))
        self.detOmega = det(self.Omega)
        if self.detOmega <= 0:
            self.LLK = 10000
        else:
            self.LLK = self.Tklog+T/2*log(self.detOmega)+k*log(self.detChol)
        if debug:
            print(self.Omega)
            print(self.Phi)
            print(self.LLK)
        if with_gradient:
            self.calcGrLLK(T, k, p, q)

        return self.LLK


def optimize_model(
        X, theta0, p, trend, debug=False,
        constraints=(), bounds=None):
    """optimize LLK
    """
    from scipy.optimize import minimize

    ve = VARsMA_Estimator(X)
    ve.setEstimationStructure(p, trend)
    
    def f(theta):
        ve.calc(theta)
        return ve.LLK
        
    def jf(theta):
        return ve.grLLK

    ret = minimize(f, theta0, method='trust-constr',
                   jac=jf, constraints=constraints, bounds=bounds)
    return ret


def _optimize_model_HS(X, theta0, p, trend, debug=False):
    """ A failed attempt to use the Hansen-Sargent
    root inversion map to avoid constrained optimization
    problem
    """
    from utils import HSRC, gen_rc_roots, jac_HSRC
    from scipy.optimize import minimize
    from types import SimpleNamespace

    ve = VARsMA_Estimator(X)
    ve.setEstimationStructure(p, trend)
    sim_val = SimpleNamespace(llk=m_BIG_M_LLK,
                              gr_llk=ones(theta0.shape[0]))
                              
    def f(theta):
        rts, stable_map, all_stable = gen_rc_roots(theta)
        if all_stable:
            ve.calc(theta)
            sim_val.llk = ve.LLK
            sim_val.gr_llk = ve.grLLK
        else:
            stable_theta = HSRC(rts, stable_map)
            j_hs = np.real(jac_HSRC(rts, stable_map))
            ve.calc(stable_theta)
            sim_val.llk = ve.LLK
            sim_val.gr_llk = ve.grLLK @ j_hs
        if sim_val.gr_llk is None:
            print('None gradient theta=%s' % str(theta))
        elif debug:
            print('Theta=%s' % str(theta))
            print(sim_val)
        return sim_val.llk
                              
    def jf(theta):
        if debug:
            print('gr=%s' % str(sim_val.gr_llk))
        return sim_val.gr_llk

    try:
        ret = minimize(f, x0=theta0.copy(), method='L-BFGS-B', jac=jf)
        theta_opt = to_invertible(ret['x'])
    except Exception:
        import traceback
        import pdb
        import sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    return ret, theta_opt
    """
    n_test = 10
    x_res = full((n_test, theta0.shape[0]+1), nan)
    message_res = full((n_test), '', dtype=object)
    
    for i in range(n_test):
        theta = np.random.randn(theta0.shape[0])
        print("before %f " % f(theta))
        good = True
        try:
            ret = minimize(f, x0=theta, method='CG', jac=jf)
        except Exception as e:
            good = False
            print("not good %s" % str(theta))
            print(e)
            pass
        if good:
            print("after %s" % str(ret))
            rts, stable_map, all_stable = gen_rc_roots(ret['x'])
            if all_stable:
                x_res[i, :-1] = ret['x']
            else:
                x_res[i, :-1] = HSRC(rts, stable_map)
            x_res[i, -1] = ret['fun']
            message_res[i] = ret
            print('after %str' % str(ret))
    return x_res, message_res
    """

