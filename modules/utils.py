import numpy as np
from scipy import signal, interpolate



def interpolate_grids(start, end, n):
    interp_func = interpolate.interp1d([0, 1], [start, end], axis=0)
    t = np.linspace(0, 1, 1+n)[1:]  # Skip the first one as it would be covered by the previous last.
    return interp_func(t)


def make_fractal_from_roots_img(roots, roots_img, tol):
    """Assigns each root a unique number in increasing order."""
    fractal = np.zeros_like(roots_img, dtype=int)
    for i, root in enumerate(roots, start=1):
        mask = np.isclose(roots_img, root, atol=tol)
        fractal[mask] = i
    return fractal

# def make_fractal_from_roots_img(roots, roots_img, root_to_id=None):
#     roots = np.array(roots)[:, None, None]
#     dists = np.abs(roots_img - roots)
#     root_id_arr = dists.argmin(axis=0)
#     if root_to_id is not None:
#         fractal = np.zeros_like(root_id_arr)
#         roots = list(roots.squeeze()) if len(roots)>1 else [roots.squeeze().item()]
#         for root in roots:
#             fractal[root_id_arr == roots.index(root)] = root_to_id[root]
#     else:
#         fractal = 1 + root_id_arr
#     return fractal

def get_rounded_unique_roots(converged_roots, iters_to_converge, root_prec=4, min_count_present=1):
    """Returns the unique roots from the roots img after rounding to a certain precision and 
    having a min number of counts in the image."""
    mask = iters_to_converge < iters_to_converge.max()
    roots, counts = np.unique(converged_roots[mask].round(root_prec), return_counts=True)
    return roots[counts > min_count_present]

def relabel_based_on_counts(arr):
    unique, counts = np.unique(arr, return_counts=True)
    sorted_indices = np.argsort(counts)
    mapping = {unique[i]: sorted_indices[i] for i in range(len(unique))}
    return np.vectorize(mapping.get)(arr)

def generate_mapping(all_roots: list[np.ndarray[complex]]):
    roots_to_ids = [{}, {all_roots[1].item(): 1}]
    for roots in all_roots[2:]:
        root_to_id = {}
        for prev_root, prev_id in roots_to_ids[-1].items():
            matching_root_index = abs(roots-prev_root).argmin()
            root_to_id[roots[matching_root_index]] = prev_id
        current_id = prev_id + 1
        for current_root in roots:
            if current_root not in root_to_id:
                root_to_id[current_root] = current_id
                current_id += 1
        roots_to_ids.append(root_to_id)
    return roots_to_ids
