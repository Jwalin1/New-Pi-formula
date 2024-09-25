import json
import numpy as np
import sympy as sp
import scipy.special as spc
from mpmath import mp
from tqdm.auto import tqdm


# Pi approximation function evaluation.
rf = lambda z, m : spc.gamma(z+m) / spc.gamma(z)
def pi_appr(z, num_terms=10, axis=0, cum=False):
    # Returns the function evaluation for float or np array.
    i = np.arange(num_terms)
    t1 = 1/(i+z) - 4/(2*i+1)
    t2 = ((2*i+1)*(2*i+1))/(4*(i+z))
    seq = t1 * rf(t2-i, i-1) / spc.factorial(i)
    return seq.sum(axis=axis) if not cum else seq.cumsum(axis=axis)


def pi_appr_mp(
    z: complex | mp.mpc, num_terms: int=10, dps: None | int=None
) -> mp.mpc:
    if dps is not None:  mp.dps = dps
    sum_ = 0
    for i in range(num_terms):
        t1 = 1/(i+z) - 4/(2*i+1)
        t2 = ((2*i+1)*(2*i+1))/(4*(i+z)) - i
        sum_ += t1 * mp.rf(t2, i-1) / mp.factorial(i)
    return sum_


# Root computation:
def expand_pi_series(num_terms: int):
    # Symbolic simplification.
    n, z = sp.symbols('n lambda')
    summand = (
        (1 / sp.factorial(n))
        * ((1 / (n + z)) - (4 / (2 * n + 1)))
        * sp.rf(((((2 * n + 1) ** 2) / (4 * (n + z))) - n), n - 1)
    )
    series = sp.Sum(summand, (n, 0, num_terms-1))
    return series.doit().cancel()  # Cancel to avoid poles.

def series_to_poly(series):
    z = sp.Symbol('lambda')
    expression = series - sp.pi
    numer, denom = expression.as_numer_denom()
    numer = sp.expand(numer)
    return sp.Poly(numer, z)

def compute_roots(num_terms: int, solver='numpy') -> np.ndarray:
    """Numerical root computation. Sympy is a bit slower than numpy.
    Sypmy fails for num_terms >= 10."""
    series = expand_pi_series(num_terms)
    sp_poly = series_to_poly(series)
    if solver == 'numpy':
        np_poly = np.poly1d(sp_poly.all_coeffs())
        roots = np.roots(np_poly)
    elif solver == 'sympy':
        roots = np.array(sp.nroots(sp_poly), dtype=complex)
    else:
        raise ValueError(f'Invalid solver: `{solver}`')
    return np.sort_complex(roots)


def refine_roots(all_roots: list[np.ndarray], tol=1e-10):
    '''Refine the roots using Newton's method. Not a good strategy to refine
    the roots this way since some roots will end up converging to a completely different
    root because of the fractal nature of the Newton's method.'''
    for num_terms, roots in enumerate(tqdm(all_roots), start=1):
        for root_index, root in enumerate(roots):
            roots[root_index] = newt_pi_appr(root, num_terms, 100, tol)


def save_roots(all_roots: list[np.ndarray], save_path: str):
    complex_to_tuple = lambda c: (c.real, c.imag)
    all_roots = [list(map(complex_to_tuple, roots)) for roots in all_roots]

    with open(save_path, 'w') as f:
        json.dump(all_roots, f, indent=2)

def load_roots(path_to_roots: str) -> list[np.ndarray]:
    with open(path_to_roots, 'r') as file:
        all_roots = json.load(file)

    tuple_to_complex = lambda c: complex(*c)
    return [np.array(list(map(tuple_to_complex, roots))) for roots in all_roots]



# Functions for the Newton's method.
def pi_appr_deriv(z, n, axis=0):
    i = np.arange(n)
    a = 2*i + 1
    b = i + z
    a2, b2 = a*a, b*b
    c = a2 / (4*b)

    t1 = rf(c - i, i-1) / b2
    t2 = a2 * (1/b - 4/a) * rf(c - i, i-1)
    t3 = spc.digamma(c -1) - spc.digamma(c - i)
    f = t1 + (t2 * t3) / (4 * b2)
    seq_deriv = f / spc.factorial(i)
    return seq_deriv.sum(axis=axis)

def pi_appr_deriv(z, n, axis=0):
    i = np.arange(n)
    a = i + z
    a4 = 4 * a
    b = 1 - 4*i*(z-1)
    c = 2*i + 1
    d = c * (2*i+4*z-1)

    t1 = rf(b / a4, i-1)
    t2 = d * spc.digamma(b/a4)
    t3 = d * spc.digamma((c*c)/a4 - 1)
    f = (t1 * (a4+t2-t3)) / (a4*a*a)
    seq_deriv = f / spc.factorial(i)
    return seq_deriv.sum(axis=axis)


def newt_pi_appr(z, num_terms, num_iters, tol=0):
    for _ in range(num_iters):
        err = np.pi - pi_appr(z, num_terms)
        if abs(err) <= tol:
            return z
        z_next = z - err / pi_appr_deriv(z, num_terms)
        if not np.isfinite(z_next):
            return z
        z = z_next
    return z
