import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import signal
from tqdm.auto import tqdm

import celluloid

from pi_formula_core import pi_appr
from modules.utils import interpolate_grids, generate_mapping, get_rounded_unique_roots, make_fractal_from_roots_img
from modules.complex_grid_processor import compute_convergence_2d_pi, create_newton_fractal

# _CMAP = matplotlib.colormaps['Greys']
_CMAP = matplotlib.colormaps['inferno_r']
_adjust_cmap = lambda cmap: cmap.reversed() if cmap.name.endswith('_r') else cmap
normalize_01 = lambda x: (x-x.min()) / (x.max()-x.min())

_split = lambda z: (z.real, z.imag)
def scatter_roots(ax, all_roots: list[complex | np.ndarray], label=False, scatter_kwargs = {}):
    # Accepts roots or a list of roots.
    if isinstance(all_roots[0], np.ndarray):
        for i, roots in enumerate(all_roots):
            label_value = f'{i+1} terms' if label else None
            ax.scatter(*_split(roots), label=label_value, **scatter_kwargs)
    elif isinstance(all_roots[0], (float, complex)):
        for root in all_roots:
            root = root.real if np.isreal(root) else root
            label_value = f'{root:.3f}' if label else None
            ax.scatter(*_split(root), label=label_value, **scatter_kwargs)
    else:
        raise ValueError

def make_yticks_imaginary(ax):
    ylim = ax.get_ylim()
    ax.set_yticks(ax.get_yticks())
    # Also `set_yticks` To avoid `UserWarning`: FixedFormatter should only be used together with FixedLocator
    ax.set_yticklabels([f'{round(tick, 15)}i' for tick in ax.get_yticks()])
    ax.set_ylim(ylim)



def plot_convergence_1d_pi(ax, x: np.ndarray, y: np.ndarray, find_peaks_kwargs: dict = {}):
    ax.plot(x, y)
    minimas, _ = signal.find_peaks(-y, **find_peaks_kwargs)
    for minima in minimas:
        min_x, min_y = x[minima], y[minima]
        ax.scatter(min_x, min_y)
        ax.annotate(f'{min_x:.3f}', (min_x, min_y), fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel(r'log( |$\pi - \pi_{{appr}}(x, n)$| )')

def animate_convergence_1d_pi(x: np.ndarray, num_terms: int):
    appr_errs = abs(np.pi - pi_appr(x[:, None], num_terms, axis=1, cum=True))
    ys = np.log(appr_errs)
    fig, ax = plt.subplots(figsize=(18, 12))
    camera = celluloid.Camera(fig)
    for n, y in tqdm(zip(range(1, 1+num_terms), ys.T), total=num_terms):
        ax.set_prop_cycle(None)  # Reset colors.
        plot_convergence_1d_pi(ax, x, y, find_peaks_kwargs={'prominence': 2})
        # ax.set_title(rf'log plot of distance from $\pi$ for $n={n}$')
        ax.text(0.5, 1.02, rf'log plot of distance from $\pi$ for $n={n}$',
                transform=ax.transAxes, ha='center', fontsize=15)
        camera.snap()
    anim = camera.animate()
    plt.close(fig)
    return anim



def plot_convergence_2d_pi(ax, extent: tuple[int, int, int, int],
                        pi_appr_err: np.ndarray,
                        fig=None, cax=None,
                        vmin: float = None, vmax: float = None,
                        roots = [], scatter_kwargs={}, cyclic_color_overlay=False):
    '''Plots the output from `complex_grid_processor.compute_converge_2d`.
    Use the same `extent` for both functions.'''
    img = ax.imshow(pi_appr_err, cmap=_CMAP, extent=extent, vmin=vmin, vmax=vmax)
    if cyclic_color_overlay:
        ax.imshow(pi_appr_err, cmap='prism', extent=extent, vmin=vmin, vmax=vmax, alpha=0.1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    make_yticks_imaginary(ax)
    if fig is not None:
        fig.colorbar(img, ax=ax, cax=cax, fraction=0.046, pad=0.04,
            label=r'log( |$\pi - \pi_{{appr}}(x, n)$| )')
    if len(roots) > 0:
        scatter_roots(ax, roots, scatter_kwargs=scatter_kwargs)
    ax.set_xlim(xlim), ax.set_ylim(ylim)
    return img

def animate_convergence_2d_pi(extent: tuple[int, int, int, int], num_terms: int = 200, res: int = 64,
                           all_roots = [], vmin: float = None, vmax: float = None, cyclic_color_overlay=False):
    # Set `all_roots` if you want to scatter the roots along with the convergence image.
    # Set `vmin` and `vmax` if you want the color intensities to represent the same values throughout the animation.
    fig, ax = plt.subplots(figsize=(12, 12))
    camera = celluloid.Camera(fig)
    img_generator = compute_convergence_2d_pi(extent, num_terms, res=res)
    fig.colorbar(plt.cm.ScalarMappable(cmap=_CMAP), ax=ax, fraction=0.046, pad=0.04, ticks=[])
    _, cax = fig.get_axes()  # 1st axis is the initial one, and the 2nd one is the one added by the colorbar.

    for n, pi_appr_img in enumerate(tqdm(img_generator, total=num_terms), start=1):
        pi_appr_err = np.log(abs(np.pi - pi_appr_img))
        plot_convergence_2d_pi(ax, extent, pi_appr_err,
                            fig=fig, cax=cax, vmin=vmin, vmax=vmax,
                            roots = all_roots[n-1] if n<len(all_roots) else [],
                            scatter_kwargs={'s':50},  # Size should be adjusted according to figsize.
                            cyclic_color_overlay=cyclic_color_overlay)
        # ax.set_title(rf'log plot of distance from $\pi$ for $n={n}$')
        ax.text(0.5, 1.02, rf'log plot of distance from $\pi$ for $n={n}$',
                transform=ax.transAxes, ha='center', fontsize=15)
        camera.snap()
    anim = camera.animate()
    plt.close(fig)
    return anim



def animate_convergence_2d_roots_iters(
        extent: tuple[int, int, int, int],
        num_terms: int = 20, num_iters: int = 10, 
        tol=1e-5, res: int = 64,
):
    fig, ax = plt.subplots(figsize=(12, 12))
    camera = celluloid.Camera(fig)
    img_generator = create_newton_fractal(extent, num_terms=num_terms, num_iters=num_iters, res=res, tol=tol)

    for n, (roots_appr_img, iters_to_converge) in enumerate(tqdm(img_generator, total=num_iters), start=1):
        if np.all(iters_to_converge == iters_to_converge[0, 0]):  # If all same, then all black.
            ax.imshow(iters_to_converge, cmap=_adjust_cmap(_CMAP), extent=extent)  # reversed cmap would give all bright.
        else:
            ax.imshow(np.log(iters_to_converge), cmap=_CMAP, extent=extent)
        make_yticks_imaginary(ax)
        # ax.set_title(f'log plot of {n} newton's method iterations')
        ax.text(0.5, 1.02, f"log plot of {n} newton's method iterations",
                transform=ax.transAxes, ha='center', fontsize=15)
        camera.snap()
    anim = camera.animate()
    plt.close(fig)
    return anim


def animate_convergence_2d_roots_iters_interp(results_path: str, extent: tuple[int, int, int, int], num_terms: int, num_iters_total: int):
    results_path = os.path.join(results_path, f'{num_terms}_terms')
    assert all(os.path.exists(os.path.join(results_path, f'{n}_iters')) for n in range(1, num_iters_total+1))
    fig, ax = plt.subplots(figsize=(12, 12))
    camera = celluloid.Camera(fig)

    iters_to_converge_prev = np.zeros_like(np.load(os.path.join(results_path, '1_iters', 'iters.npy')))
    num_interps_total = np.geomspace(50, 1, num_iters_total).astype(int)

    for num_iters, num_interps in tqdm(
        zip(range(1, 1+num_iters_total), num_interps_total), total=num_iters_total
    ):
        iters_to_converge = np.load(os.path.join(results_path, f'{num_iters}_iters', 'iters.npy'))
        
        if np.all(iters_to_converge == iters_to_converge[0, 0]):  # If all same, then all black.
            ax.imshow(iters_to_converge, cmap=_adjust_cmap(_CMAP), extent=extent)  # reversed cmap would give all bright.
            ax.text(0.5, 1.02, f"log plot of {num_iters} newton's method iterations",
                    transform=ax.transAxes, ha='center', fontsize=15)
            camera.snap()
        else:
            interp_grids = interpolate_grids(iters_to_converge_prev, iters_to_converge, num_interps)
            # vmin, vmax = np.log(interp_grids[-1]).min(), np.log(interp_grids[-1]).max()
            for iters_to_converge_interp in interp_grids:
                ax.imshow(np.log(iters_to_converge_interp), cmap=_CMAP, extent=extent) #, vmin=np.log(3), vmax=np.log(num_iters+1))
                make_yticks_imaginary(ax)
                # ax.set_title(f'log plot of {num_iters} newton's method iterations')
                ax.text(0.5, 1.02, f"log plot of {num_iters-1} newton's method iterations",
                        transform=ax.transAxes, ha='center', fontsize=15)
                camera.snap()
        iters_to_converge_prev = iters_to_converge.copy()

    # Take a snap with the last frame.
    iters_to_converge = np.load(os.path.join(results_path, f'{num_iters_total}_iters', 'iters.npy'))
    ax.imshow(np.log(iters_to_converge_interp), cmap=_CMAP, extent=extent)
    ax.text(0.5, 1.02, f"log plot of {num_iters} newton's method iterations",
        transform=ax.transAxes, ha='center', fontsize=15)
    camera.snap()

    anim = camera.animate()
    plt.close(fig)
    return anim


def animate_convergence_2d_roots_terms(
    results_path: str,
    extent: tuple[int, int, int, int],
    num_iters: int,
    num_terms_total: int,
    all_roots: list[np.ndarray[complex]]
):
    assert all(os.path.exists(os.path.join(results_path, f'{n}_terms', f'{num_iters}_iters')) for n in range(1, num_terms_total+1))
    fig, ax = plt.subplots(figsize=(12, 12))
    camera = celluloid.Camera(fig)

    cmap = plt.get_cmap('prism')
    # colors = cmap(np.linspace(0, 1, 200))
    # cmap = ListedColormap(colors)
    # # `roots_to_ids[num_terms-1]` can be paassed to `make_fractal_from_roots_img`
    # roots_to_ids = generate_mapping(all_roots)

    for num_terms in tqdm(range(1, 1+num_terms_total)):
        term_iter_path = os.path.join(results_path, f'{num_terms}_terms', f'{num_iters}_iters')
        iters_to_converge = np.load(os.path.join(term_iter_path, 'iters.npy'))
        converged_roots = np.load(os.path.join(term_iter_path, 'roots.npy'))
        if np.all(iters_to_converge == iters_to_converge[0, 0]):  # If all same, then all black.
            ax.imshow(iters_to_converge, cmap=_adjust_cmap(_CMAP), extent=extent)  # reversed cmap would give all bright.
        else:
            # Colorless.
            # ax.imshow(np.log(iters_to_converge), extent=extent, cmap=_CMAP)

            # Colored.
            # Use either theoretical or empirical roots to get the fractal.
            # For the empirical approach use the make_fractals function with the same `tol` as `root_prec`.
            empirical_roots = get_rounded_unique_roots(converged_roots, iters_to_converge, root_prec=4)
            theoretical_roots = all_roots[num_terms - 1]
            fractal = make_fractal_from_roots_img(empirical_roots, converged_roots, tol=1e-4)
            alpha = 1 - normalize_01(np.log(iters_to_converge))
            ax.imshow(fractal, cmap=cmap, alpha=alpha, extent=extent)
            make_yticks_imaginary(ax)

        # ax.set_title(f"log plot using {num_terms-1} terms for {num_iters} newton's method iterations")
        ax.text(0.5, 1.02, f"log plot using {num_terms-1} terms for {num_iters} newton's method iterations",
                transform=ax.transAxes, ha="center", fontsize=15)
        camera.snap()

    anim = camera.animate()
    plt.close(fig)
    return anim
