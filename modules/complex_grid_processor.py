# Modules to hold functions over a 2d complex array.
import numpy as np
import scipy.special as spc
from tqdm.auto import tqdm

from pi_formula_core import rf, pi_appr, pi_appr_deriv



def compute_convergence_2d_pi(
    extent: tuple[int, int, int, int], num_terms: int = 201, res: int = 64
) -> np.ndarray:
    '''Computes abs distance to pi using the pi apprxomation formula
    for a given 2d extent [x_start, x_end, y_start, y_end] in the complex plane.
    Returns a 2d int array.'''
    x = np.linspace(*extent[:2], res)
    y = np.linspace(*extent[2:], res)
    x, y = np.meshgrid(x, y)
    z = x + y*1j

    valid_mask = np.ones_like(z, dtype=bool)
    pi_appr_img = np.zeros_like(z, dtype=complex)
    pi_appr_next = pi_appr_img.copy()

    for i in range(num_terms):
        t1 = 1/(i+z[valid_mask]) - 4/(2*i+1)
        t2 = ((2*i+1)*(2*i+1))/(4*(i+z[valid_mask])) - i
        summand = (1/spc.factorial(i)) * t1 * rf(t2, i-1)
        pi_appr_next[valid_mask] += summand

        valid_mask = np.isfinite(pi_appr_next)
        if not valid_mask.any():
            return pi_appr_img
        pi_appr_img[valid_mask] = pi_appr_next[valid_mask]
        yield pi_appr_img


def compute_terms_to_converge_pi(extent: tuple[int, int, int, int], num_terms: int = 201, res: int = 64,
                              tol: float = 1e-3):
    pi_appr_generator = compute_convergence_2d_pi(extent, num_terms, res)
    pi_appr_img = next(pi_appr_generator)  # Used to get shape, all values in the img would be 4.
    terms_to_converge = np.zeros_like(pi_appr_img, dtype=int)
    converged_mask = abs(pi_appr_img - np.pi) < tol
    terms_to_converge[~converged_mask] += 1

    for pi_appr_img in pi_appr_generator:
        converged_mask = abs(pi_appr_img - np.pi) < tol
        if converged_mask.all():
            return pi_appr, terms_to_converge
        terms_to_converge[~converged_mask] += 1
        yield pi_appr, terms_to_converge



def create_newton_fractal(extent, num_terms=11, num_iters=15, tol=1e-8, res=64):
    x = np.linspace(*extent[:2], res)
    y = np.linspace(*extent[2:], res)
    x, y = np.meshgrid(x, y)
    roots = x + y*1j
    roots_next = roots.copy()
    iters_to_converge = np.ones_like(roots, dtype=int)
    valid_mask = np.ones_like(roots, dtype=bool)
    converged_mask = np.zeros_like(roots, dtype=bool)

    for i in range(num_iters):
        pi_appr_err = np.pi - pi_appr(roots[valid_mask][:, None], num_terms, axis=1)
        pi_appr_err_deriv = pi_appr_deriv(roots[valid_mask][:, None], num_terms, axis=1)
        update_step = pi_appr_err / pi_appr_err_deriv
        roots_next[valid_mask] -= update_step

        converged_mask[valid_mask] = abs(update_step) < tol
        iters_to_converge[~np.isfinite(roots_next) | ~converged_mask] += 1
        valid_mask = np.isfinite(roots_next) & ~converged_mask

        if not valid_mask.any():
            return roots, iters_to_converge
        roots[valid_mask] = roots_next[valid_mask]
        yield roots, iters_to_converge
