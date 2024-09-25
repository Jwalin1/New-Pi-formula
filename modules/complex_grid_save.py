import os
import numpy as np
from tqdm.auto import tqdm

from modules.complex_grid_processor import create_newton_fractal



def save_convergence_2d_roots_iters(
        save_path: str,
        extent: tuple[int, int, int, int],
        num_terms: int = 21, num_iters: int = 10, 
        tol=1e-5, res: int = 64,
        save_only_last: bool = False,
):
    save_path = os.path.join(save_path, f'{num_terms}_terms')
    os.makedirs(save_path, exist_ok=True)
    img_generator = create_newton_fractal(extent, num_terms=num_terms, num_iters=num_iters, res=res, tol=tol)
    for n, (roots_appr_img, iters_to_converge) in enumerate(tqdm(img_generator, total=num_iters), start=1):
        if (not save_only_last) or (n==num_iters):
            iters_dir = os.path.join(save_path, f'{n}_iters')
            roots_path = os.path.join(iters_dir, 'roots.npy')
            iters_path = os.path.join(iters_dir, 'iters.npy')
            os.makedirs(iters_dir, exist_ok=True)
            np.save(roots_path, roots_appr_img)
            np.save(iters_path, iters_to_converge)


def save_convergence_2d_roots_terms_iters(
        save_path: str,
        extent: tuple[int, int, int, int],
        num_terms: int = 21, num_iters: int = 10, 
        tol=1e-5, res: int = 64,
):
    for n in tqdm(range(1, 1+num_terms)):
        terms_dir = os.path.join(save_path, f'{n}_terms')
        last_roots_path = os.path.join(terms_dir, f'{num_iters}_iters', 'roots.npy')
        last_iters_path = os.path.join(terms_dir, f'{num_iters}_iters','iters.npy')

        if not (os.path.exists(last_roots_path) and os.path.exists(last_iters_path)):
            save_convergence_2d_roots_iters(save_path=save_path,
                extent=extent, num_terms=n, num_iters=num_iters, tol=tol, res=res,
                save_only_last=True)
