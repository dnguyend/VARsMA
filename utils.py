import cmath
import numpy as np
from numpy import pi, sqrt, zeros, eye, diag, flip, ones
from numpy import full, real, imag, nan, exp, poly1d
from numpy import conj, prod, isin, concatenate, roots
from numpy.linalg import solve, det, eigh, eig, inv
from numpy.random import randn, randint, uniform, choice
from numpy.fft import ifft
from types import SimpleNamespace
from numba import jit
# from scipy.interpolate import lagrange

m_SMALL_SQRT = np.finfo(float).eps

HCN = np.array([1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180,
                240, 360, 720, 840, 1260, 1680, 2520, 5040,
                7560, 10080, 15120, 20160, 25200, 27720, 45360,
                50400, 55440, 83160, 110880, 166320, 221760,
                277200, 332640, 498960, 554400, 665280, 720720,
                1081080, 1441440, 2162160], dtype=int)


def nextn(i):
    return HCN[np.searchsorted(HCN, [i])[0]]


@jit(nopython=True)
def inverse_poly(qpol, n):
    """inverse polynomial. Chopped after n terms
    including the first term
    """
    ret = zeros(n)
    ret[0] = 1
    qrev = -qpol[::-1].copy()
    for i in range(1, n):
        if i >= qpol.shape[0]:
            ret[i] = ret[i-qpol.shape[0]:i] @ qrev
        else:
            ret[i] = -(ret[:i] @ qpol[:i][::-1])
    return ret


@jit(nopython=True)
def mat_convol(mat, vec):
    """ Matrix convolution
    """
    ret = zeros(mat.shape)
    N = mat.shape[0]
    for i in range(N):
        ret[i:, :] = ret[i:, :] + vec[i]*mat[:N-i, :]
    return ret


@jit(nopython=True)
def mat_inv_convol(mat, theta):
    """ convolution with inverse time series.
    the inverse will be chopped after n terms
    theta will have no constant term (assume to be 1)
    """
    ret = zeros(mat.shape)
    N = mat.shape[0]
    ret[0, :] = mat[0, :]
    q = theta.shape[0]
    theta_rev = theta[::-1].copy()
    for i in range(1, N):
        if i <= q:
            ret[i, :] = mat[i, :] - theta[:i][::-1] @ ret[:i]
            # ret[i, :] = mat[i, :] - theta_rev[q-i:] @ ret[:i]
        else:
            ret[i, :] = mat[i, :] - theta_rev @ ret[i-q:i, :]
    return ret


def gen_norm_noise_mean_cov(
        n, mean=None,
        sigma=None,
        method="eigen"):
    """generate time series with given mean
    and covariant matrix
    """
    if mean is None:
        mean = zeros(sigma.shape[0])
    if sigma is None:
        sigma = eye(mean.shape[0])
        
    if not np.allclose(
            sigma, sigma.T,
            atol=m_SMALL_SQRT):
        raise ValueError("sigma must be a symmetric matrix")
    
    if mean.shape[0] != sigma.shape[0]:
        raise ValueError("mean and sigma have non-conforming size")

    if method == "eigen":
        e, v = eigh(sigma)
        if (not np.all(e >= -m_SMALL_SQRT * np.abs(e[0]))):
            print("sigma is numerically not positive definite")
        R = v @ (sqrt(e)[:, None] * v.T)
    retval = randn(n, sigma.shape[1]) @ R
    retval += mean[None, :]
    return retval


def gen_random_pos_symmetric(k):
    dd = randint(1, 5, k).astype(float)
    i = np.triu_indices(k)
    v = zeros((k, k))
    v[i] = randn(i[0].shape[0])
    ret = v.T @ (diag(dd) @ v)
    ret[i] = ret.T[i]
    return ret


def VARMA_sim(nobs, arlags=None, malags=None, cnst=None,
              phi=None, theta=None,
              skip=200, sigma=None):
    """ Utitity to generate VARMA(p,q) time series using Gaussian innovations.
    p: ar order (lags can be skipped)
    q: ma order (lags can be skipped)
    nobs: sample size
    cnst: constant vector
    phi: store AR coefficient matrices [phi1,phi2,...]
    theta: store MA coefficient matrices [theta1,theta2,...]
    arlags: order for each AR coefficient matrix
    malags: order for each MA coefficient matrix.
    """

    k = sigma.shape[0]
    nT = nobs+skip

    if arlags is None:
        p = phi.shape[1] // phi.shape[0]
        nar = p
        arlags = np.arange(1, p+1)
    if malags is None:
        q = theta.shape[1] // theta.shape[0]
        nma = q
        malags = np.arange(1, q+1)
    
    ist = max(p, q) + 1
    at = gen_norm_noise_mean_cov(nT, zeros((k)), sigma)
    zt = at.copy()
   
    if nma > 0:
        for j in range(nma):
            zt[ist:, :] += at[ist-malags[j]:-malags[j], :] @\
                theta[:, j*k:(j+1)*k]
    if(nar > 0):
        for it in range(ist, nT):
            for i in range(nar):
                zt[it, :] += phi[:, i*k:(i+1)*k] @\
                    zt[it-arlags[i], :]
            if cnst is not None:
                zt[it, :] += cnst
    init_series = full((0, k), nan)
    init_noises = full((0, k), nan)
   
    if (p > 0):
        init_series = zt[(skip-p):skip, :]
    if (q > 0):
        init_noises = at[(skip-q):skip, :]
    return SimpleNamespace(
        series=zt[skip:nT, :],
        noises=at[skip:nT, :],
        init_noises=init_noises,
        init_series=init_series)


def eval_arma(x, max_deg, n_ins, ph0, phi, theta):
    """Eval phi and theta at x
    """
    ma = theta[:, (max_deg-1)*n_ins:max_deg*n_ins]
    ar = phi[:, (max_deg-1)*n_ins:max_deg*n_ins]
    if max_deg > 1:
        for a in flip(np.arange(max_deg-1)):
            ma = ma * x + theta[:, a*n_ins:(a+1)*n_ins]
            ar = ar * x + phi[:, a*n_ins:(a+1)*n_ins]
    ma = ma * x + ph0
    ar = ar * x + ph0
    return SimpleNamespace(ma=ma, ar=ar)


def eval_poly_matrix(x, theta):
    """Eval polynomial matrix at points in x
    """
    n_ins = theta.shape[0]
    n_blk = theta.shape[1] // theta.shape[0]
    ma = np.zeros((n_ins, n_ins, x.shape[0]), dtype=np.complex)
    ma += theta[:, (n_blk-1)*n_ins:n_blk*n_ins][:, :, None]

    for a in flip(np.arange(n_blk-1)):
        ma = ma * x[None, None, :] + theta[:, a*n_ins:(a+1)*n_ins][:, :, None]

    return ma


def _eval_mat(x, B, A):
    d = A.shape[1]
    k = B.shape[0]
    if (A.shape[0] != k*k) or (B.shape[2] != d+1):
        raise ValueError("bad dimension for A and B")
    
    ret = full((k, k, x.shape[0]), nan, dtype=np.complex)
    for i in range(x.shape[0]):
        ret[:, :, i] = B[:, :, 0]
        for j in range(d):
            ret[:, :, i] = (
                ret[:, :, i] @
                (eye(k)-x[i]*A[:, j].reshape(k, k))) @ B[:, :, (j+1)]
    return ret


def make_real_diag(x, n_pair):
    """x is a vector with first lx - 2 n_pair
    entries representing real roots
    and last 2 n_pair entries representing
    pair of conjugate complex roots (pair containing
    real and complex part)
    Returning a block diagonal matrix
    with roots given by x
    real roots are by 1-block and complex roots
    are by 2-blocks
    """
    lx = x.shape[0]
    ret = zeros((lx, lx))
    if lx > 2*n_pair:
        ret[:(lx-2*n_pair), :(lx-2*n_pair)] =\
            diag(x[:lx-2*n_pair])
        
    if n_pair != 0:
        for i in range(n_pair):
            a_root = cmath.rect(x[lx-2*n_pair+2*i], x[lx-2*n_pair+2*i+1])
            ret[lx-2*n_pair+2*i:lx-2*n_pair+2*i+2,
                lx-2*n_pair+2*i:lx-2*n_pair+2*i+2] =\
                np.array([real(a_root), imag(a_root),
                          -imag(a_root), real(a_root)]).reshape(2, 2)
    return ret


def gen_stable_model(d, k, long_ret=False):
    """generate a monic stable polynomial matrix of a certain degree d
    algorithm: For d=1, we just need A have eigenvalue in the unit circle
    then I-AL will be stable.
    for high degree a matrix of form
    B_1(I-A_1 L)B_2(I-A_2 L)...B_d(I-A_d L)B_{d+1}
    where A_1, ..., A_d all have eigen values in unit circle
    and B_1j...B_(d+1)j = 1
    to multiply out we evaluate the matrix at d+1 points
    then apply Lagrange interpolation.
    we can of course choose A_1,...A_d
    to be diagonal, just need to adjust the B's
    """
    
    B = full((k, k, d+1), nan)
    B[:, :, :d] = randn(k, k, d)
    for i in range(d):
        e, v = eig(B[:, :, i])
        smallE = np.where(np.abs(e) < 1e-6)[0]
        while smallE.shape[0] > 0:
            B[:, :, i] = randn(k, k)
            e, v = eig(B[:, :, i])
            smallE = np.where(np.abs(e) < 1e-6)[0]
                
    B[:, :, d] = inv(B[:, :, 0])
    for i in range(1, d):
        B[:, :, d] = solve(B[:, :, i], B[:, :, d])
    
    AC = uniform(0.1, .8, (k, d))
    n_pair = choice(np.arange(k // 2), d, True)
    A = np.full((k*k, d), nan)
    for iid in range(d):
        if (k > 2*n_pair[iid]):
            sig_list = choice([-1, 1], k-2*n_pair[iid], True)
            AC[:(k-2*n_pair[iid]), iid] = AC[:(k-2*n_pair[iid]), iid]*sig_list
        A[:, iid] = make_real_diag(AC[:, iid], n_pair[iid]).reshape(-1)
    x = np.vectorize(
        lambda i: cmath.rect(1., -2*i*pi/(d+1)))(np.arange(d+1))
    evals = _eval_mat(x, B, A)

    out_array = np.full((k, k*d), nan)
    ix = np.arange(d) * k
    for ir in range(k):
        for ic in range(k):
            n_pol = ifft(evals[ir, ic, :])
            out_array[ir, ic + ix] = np.real(
                    n_pol[1:])
    if long_ret:
        return SimpleNamespace(A=A.reshape(k, k, d), B=B, pol=out_array)
    else:
        return out_array


def poly_determinant(pmat, real=True, without_constant=True):
    """Evaluate the determinant of a polynomial matrix
    the return is a scalar polynomial.
    We return the full polynomial of degree
    k * p. In case of low McMillan degree the higher degree
    terms could be zeros. The caller should use cut off to
    reduce terms.
    The algorithm is to evaluate the determinant
    at kp+1 points (using fft) then use lagrange interpolation
    """

    if pmat.shape[1] % pmat.shape[0] != 0:
        raise(
            ValueError("not a square matrix sizes are {}{}".format(
                pmat.shape[:2])))
    k = pmat.shape[0]
    d = pmat.shape[1] // pmat.shape[0]
    if not without_constant:
        d -= 1
        
    dd = k * d
    x = np.vectorize(lambda i: exp(-2j*pi*i/(dd+1)))(
        np.arange(dd+1))
    evals = eval_poly_matrix(x, pmat)
    if without_constant:
        evals = evals * x[None, None, :] + eye(k)[:, :, None]
    det_evals = [det(evals[:, :, j])
                 for j in range(evals.shape[2])]
    det_pol = ifft(det_evals)
    if real:
        det_pol = det_pol.real
    return poly1d(np.flip(det_pol, 0))


def _misc_check_1():
    d = 3
    k = 5
    s = gen_stable_model(d, k)
    s_det = poly_determinant(s.pol, real=True)
    print(np.roots(s_det))
    
    # check s.pol
    y = .5
    print(s.B[:, :, 0] @ (
        (eye(k) - y * s.A[:, 0].reshape(k, k)) @ (
            s.B[:, :, 1] @
            ((eye(k) - y * s.A[:, 1].reshape(k, k)) @ (
                s.B[:, :, 2] @ (
                    (eye(k) - y * s.A[:, 2].reshape(k, k)) @
                    s.B[:, :, 3]))))))
    print(_eval_mat(np.array([y]), s.B, s.A))

    yy = np.vectorize(
        lambda i: cmath.rect(1., -2*i*pi/(d+1)))(np.arange(d+1))
    evals = _eval_mat(yy, s.B, s.A)
    newp = np.zeros((k, k * (d+1)))
    ix = k * np.arange(d+1)
    for ii in range(k):
        for jj in range(k):
            newp[ii, jj + ix] = np.real(
                ifft(evals[ii, jj, :]))

    print(eval_poly_matrix(np.array([y]), newp))
            
    print(eval_poly_matrix(np.array([y]),
                           concatenate([
                               np.eye(k), s.pol.reshape(k, -1)], axis=1)))

    print(newp[:, :k] + y*newp[:, k:2*k] +
          (y*y)*newp[:, 2*k:3*k] + (y*y*y)*newp[:, 3*k:])


def jac_Vieta(root_list):
    """Jacobian of the Vieta map
    """
    lq = root_list.shape[0]
    jac = full((lq, lq), nan, dtype=np.complex)
    qnext = nextn(lq+1)
    """
    x = sapply(0:qnext,function(i) {
    complex(modulus=1,argument=-2*pi*i/(qnext))})
    """
    x = np.vectorize(
        lambda i: cmath.rect(1., -2*i*pi/(qnext)))(np.arange(qnext))

    vals = 1 - x[:, None] @ root_list.reshape(1, -1)

    for i in range(lq):
        y = np.vectorize(lambda j: prod(
            vals[j, isin(np.arange(lq), [i], invert=True)]))(np.arange(qnext))
        jac[:, i] = -ifft(y)[:lq]
    return jac
    

def HSRC(rc_roots, HS_map):
    """rc_roots is a list of roots of a polynomial
    with constant term 1, both real and complex
    some of the roots may be inside a unit disc so
    the resulting MA system is not invertible.
    Roots are a range such as real first
    then complex.

    Hansen Sargent map is root inversion map.

    HS map has 2 lists with real and complex components
    denoting the roots that need to be inverted
    The function returns the polynomial obtained
    by inverting those roots
    """
    # d = rc_roots.real.shape[0] + 2 * rc_roots.cplx.shape[0]
    y = poly1d([1.])
    for i in range(rc_roots.real.shape[0]):
        if (i in HS_map.real) and (rc_roots.real[i] != 0):
            y = y * poly1d([- 1. / rc_roots.real[i], 1.])
        else:
            y = y * poly1d([- rc_roots.real[i], 1.])
    for i in range(rc_roots.cplx.shape[0]):
        if i in HS_map.cplx:
            inv_root = 1 / rc_roots.cplx[i]
            p2 = real(inv_root)*real(inv_root) + imag(inv_root)*imag(inv_root)
            y = y * poly1d([p2, -2*real(inv_root), 1])
        else:
            r = rc_roots.cplx[i]
            p2 = real(r)*real(r) + imag(r)*imag(r)
            y = y * poly1d([p2, -2*real(r), 1])

    return real(flip(y.coeffs[:-1]))


def HSRC_fft(rc_roots, HS_map):
    """rc_roots is a list of roots of a polynomial
    with constant term 1, both real and complex
    some of the roots may be inside a unit disc so
    the resulting MA system is not invertible.
    Roots are a range such as real first
    then complex.

    Hansen Sargent map is root inversion map.

    HS map is a list with real and complex components
    denoting the roots that need to be inverted
    The function returns the polynomial obtained
    by inverting those roots
    """
    d = rc_roots.real.shape[0] + 2 * rc_roots.cplx.shape[0]
    x = np.vectorize(
        lambda i: cmath.rect(1., -2*i*pi/(d+1)))(np.arange(d+1))

    y = ones((d+1), dtype=np.complex)
    for ix in range(d+1):
        for i in range(rc_roots.real.shape[0]):
            if (i in HS_map.real) and (rc_roots.real[i] != 0):
                y[ix] = y[ix] * (1 - x[ix] / rc_roots.real[i])
            else:
                y[ix] = y[ix] * (1 - x[ix] * rc_roots.real[i])
        for i in range(rc_roots.cplx.shape[0]):
            if i in HS_map.cplx:
                y[ix] = y[ix] * (1 - x[ix] / rc_roots.cplx[i]) *\
                        (1-x[ix]/conj(rc_roots.cplx[i]))
            else:
                y[ix] = y[ix] * (1 - x[ix] * rc_roots.cplx[i]) *\
                        (1-x[ix]*conj(rc_roots.cplx[i]))

    return real(ifft(y))[1:(d+1)]


def jac_HSRC(rc_roots, HS_map):
    """Jacobian of the HS map
    """
    lreal = rc_roots.real.shape[0]
    lcplx = rc_roots.cplx.shape[0]
    q = lreal + 2 * lcplx
    jac_diag = ones((q), dtype=np.complex)
    jac_diag[HS_map.real] = -1 /\
        (rc_roots.real[HS_map.real]*rc_roots.real[HS_map.real])
    jac_diag[lreal + HS_map.cplx] = -1 /\
        (rc_roots.cplx[HS_map.cplx]*rc_roots.cplx[HS_map.cplx])
    jac_diag[lreal+lcplx+HS_map.cplx] = conj(jac_diag[lreal + HS_map.cplx])
    jac = diag(jac_diag)
    root_list = concatenate(
        [rc_roots.real, rc_roots.cplx, conj(rc_roots.cplx)])
    inv_root_list = root_list.copy()
    if HS_map.real.shape[0] > 0:
        inv_root_list[HS_map.real] = 1 / rc_roots.real[HS_map.real]
    if HS_map.cplx.shape[0] > 0:
        inv_root_list[lreal + HS_map.cplx] = 1 / rc_roots.cplx[HS_map.cplx]
        inv_root_list[lreal + lcplx + HS_map.cplx] =\
            conj(inv_root_list[lreal + HS_map.cplx])
    
    return jac_Vieta(inv_root_list) @\
        solve(jac_Vieta(root_list).T, jac.T).T


def gen_rc_roots(theta):
    """ get the list of roots
    grouping to real and complex roots.
    # generate all the possible maps
    mark the one that is stable
    order root list to terms
    the whole set of roots are:
    rc_roots.real
    rc_roots.cplx & conj(rc_roots.cplx)
    Also return the map which is
    needed to map unstable roots to stable roots
    """
    root_list = roots(to_monic(theta, False))
    ept = np.array([])
    rc_roots = SimpleNamespace(real=ept, cplx=ept)

    small_v = 1e-6
    cnt = 0
    while (cnt < 5) and (rc_roots.real.shape[0] +
                         2*rc_roots.cplx.shape[0] < root_list.shape[0]):
        rc_roots.real = real(root_list[
            np.where(abs(imag(root_list)) < small_v)])
        rc_roots.cplx = np.sort(root_list[
            np.where(imag(root_list) > small_v)])
        cnt += 1
        small_v /= 8.
    rc_roots.real = np.sort(rc_roots.real)
    rc_roots.cplx = rc_roots.cplx[np.argsort(np.abs(rc_roots.cplx))]
    
    us_real = np.where(np.abs(rc_roots.real) > 1)[0]  # unstable roots
    us_cplx = np.where(np.abs(rc_roots.cplx) > 1)[0]
    if rc_roots.real.shape[0] + 2*rc_roots.cplx.shape[0] < root_list.shape[0]:
        raise(ValueError(
            "real and complex roots do not add up to full list"))
    all_stable = us_real.shape[0] + us_cplx.shape[0] == 0
    return rc_roots, SimpleNamespace(real=us_real, cplx=us_cplx), all_stable


def gen_ran_HS(rc_roots,  nr, nc):
    """generate a random combination of HS maps
    """
    lr = choice(np.arange(rc_roots.real.shape[0]), nr)
    lc = choice(np.arange(rc_roots.cplx.shape[0]), nc)
    return SimpleNamespace(real=lr, cplx=lc)


def to_monic(theta, flip_root=True):
    if flip_root:
        return flip(concatenate([np.array([1.]), theta]))
    else:
        return concatenate([np.array([1.]), theta])


def roots_to_theta(rc_roots):
    HS_map = SimpleNamespace(
        real=np.array([], dtype=int), cplx=np.array([], dtype=int))
    return HSRC(rc_roots, HS_map)


def to_invertible(theta):
    """
    """
    rts, stable_map, all_stable = gen_rc_roots(theta)
    if all_stable:
        return theta
    else:
        return HSRC(rts, stable_map)


def numerical_derivative(ve, theta, p):
    hh = 1e-6
    q = theta.shape[0]
    llks = zeros(q)
    base_llk = ve.LLK
    base_gr_llk = ve.grLLK
    for ii in range(q):
        ttx = theta.copy()
        ttx[ii] += hh
        ve.calc(ttx, check_stable=False)
        llks[ii] = (ve.LLK - base_llk) / hh
    print(base_gr_llk)
    print(llks)


def _test_VARMA():
    d = 4
    k = 5
    phi_s = gen_stable_model(d, k, long_ret=True)
    theta_s = gen_stable_model(d, k, long_ret=True)
            
    phi = -phi_s.pol
    theta = theta_s.pol
    phi_det = poly_determinant(-phi)
    theta_det = poly_determinant(theta)
    phi_roots = np.roots(phi_det)
    theta_roots = np.roots(theta_det)
    print(np.abs(phi_roots))
    print(np.abs(theta_roots))
    nobs = 100
    malags = None
    cnst = None
    arlags = None
    sigma = gen_random_pos_symmetric(k)
    skip = 200
    cnst = None
    ret = VARMA_sim(
        nobs, arlags=arlags,
        malags=malags,
        cnst=cnst, phi=phi, theta=theta,
        skip=skip, sigma=sigma)

    import pandas as pd
    a1 = pd.read_csv('/tmp/a.csv', index_col=[0], header=0)
    theta2 = a1.values
    sigma2 = pd.read_csv('/tmp/sig.csv', index_col=[0], header=0)
    
    phin = np.zeros_like(theta2)
    nobs2 = 10000
    ret2 = VARMA_sim(
        nobs2, arlags=arlags,
        malags=malags,
        cnst=cnst, phi=phin, theta=-theta2,
        skip=skip, sigma=sigma2)

    theta2 = a1.values
    sigma2 = pd.read_csv('/tmp/sig.csv', index_col=[0], header=0)
    
    nobs3 = 10000
    d3 = 3
    k3 = 10
    
    theta3_s = gen_stable_model(d3, k3, long_ret=True)
    theta3 = theta3_s.pol
    phin = np.zeros_like(theta3)
    sigma3 = gen_random_pos_symmetric(k3)
    if False:
        ret3 = VARMA_sim(
            nobs3, arlags=arlags,
            malags=malags,
            cnst=cnst, phi=phin, theta=theta3,
            skip=skip, sigma=sigma3)
        ret4 = VARMA_sim(
            nobs3, arlags=arlags,
            malags=malags,
            cnst=cnst, phi=phin, theta=-theta3,
            skip=skip, sigma=sigma3)
    else:

        ret3 = VARMA_sim(
            nobs3, arlags=arlags,
            malags=malags,
            cnst=cnst, phi=theta3, theta=phin,
            skip=skip, sigma=sigma3)
        ret4 = VARMA_sim(
            nobs3, arlags=arlags,
            malags=malags,
            cnst=cnst, phi=-theta3, theta=phin,
            skip=skip, sigma=sigma3)
        
    print(np.dot(ret3.series.T, ret3.series) / nobs3)
    print('-----')
    print(np.dot(ret4.series.T, ret4.series) / nobs3)

    
if __name__ == '__main__':
    d = 4
    k = 6
    phi_s = gen_stable_model(d, k, long_ret=True)
    theta_s = gen_stable_model(d, k, long_ret=True)
        
    phi = -phi_s.pol
    theta = theta_s.pol
    phi_det = poly_determinant(-phi)
    theta_det = poly_determinant(theta)
    phi_roots = np.roots(phi_det)
    theta_roots = np.roots(theta_det)
    print(np.abs(phi_roots))
    print(np.abs(theta_roots))
    nobs = 1000
    VARMA_sim(nobs, arlags=None,
              malags=None,
              cnst=None, phi=phi, theta=theta,
              skip=200, sigma=None)

    def testJacobianfunction():
        theta = np.array([1, 2, 3.])
        lq = theta.shape[0]

        # root_list = 1. / roots(to_monic(theta))

        l1 = 1+2j
        l2 = 1-2j
        print(jac_Vieta(np.array([l1, l2]).reshape(-1, 1)))

        rc_roots, to_stable_map, all_stable = gen_rc_roots(
            theta)
        
        ahs = HSRC(rc_roots, to_stable_map)

        # jt = jac_Vieta(rc_roots)
        # print(jt)
        h = 1e-5
        HS_map = SimpleNamespace(
            cplx=np.array([0], dtype=int),
            real=np.array([], dtype=int))
        hh = jac_HSRC(rc_roots, HS_map)
        print(hh)
        ahs0 = HSRC(rc_roots, HS_map)
        for i in range(lq):
            dh = zeros(lq)
            dh[i] = h
            theta1 = theta + dh
            rc_roots1, to_stable_map1, all_stable1 = gen_rc_roots(theta1)

            ahs1 = HSRC(rc_roots1, HS_map)
            print((ahs1 - ahs0) / h)

        # mat1 = np.array([-1, l2, -1, l1]).reshape(2, 2)
        # j1 = diag(np.array([-1/(l1*l1), -1/(l2*l2)]))
        # mat2 = matrix(c(-1,1/l2,-1,1/l1),2,2)
        # mat2 %*% j1 %*% solve(mat1)
        # polyroot(c(1,ahs))


def calc_residuals(Phi, theta, X, trend=None):
    """residuals from at fitted model.
    """
    assert Phi.shape[1] % Phi.shape[0] == 0
    p = Phi.shape[1] // Phi.shape[0]
    q = theta.shape[0]
    n = X.shape[0]
    T = n - p
    k = X.shape[1]
    residuals = zeros((T+q, k))
    residuals[q:, :] = X[p:, :]
    for i in range(p):
        residuals[q:, :] -= X[i:i+T, :] @ Phi[(p-i-1)*k:(p-i)*k, :]
    if trend is not None:
        residuals[q:, :] -= trend
    for it in range(T):
        for i in range(q):
            residuals[q+it, :] -= theta[i] * residuals[q+it-i-1, :]
    return residuals[-T:, :]

        
