import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fmin_l_bfgs_b


EPS = 1e-12
def compute_sigma_perplexity(distances, perplexity):
    n = distances.shape[0]
    entropy_target = np.log(perplexity)
    sigmas = np.ones(n)

    for i in range(n):
        d_i = distances[i]

        def perp_diff(sigma):
            sigma = max(sigma, 1e-10)
            p_i = np.exp(-d_i**2 / (2.0 * sigma**2))
            p_i[i] = 0.0
            p_i_sum = p_i.sum()
            if p_i_sum == 0.0:
                p_i += EPS
                p_i_sum = p_i.sum()
            p_i /= p_i_sum
            entropy = -np.sum(p_i * np.log(p_i + EPS))
            return (entropy - entropy_target) ** 2

        sigma_opt, *_ = fmin_l_bfgs_b(
            perp_diff,
            x0=[sigmas[i]],
            bounds=[(1e-10, None)],
            approx_grad=True
        )
        sigmas[i] = sigma_opt[0]

    return sigmas


def pairwise_distances(data):
    return squareform(pdist(data, metric='euclidean'))      # (n, n)


def compute_sigmas(data, perplexity=30.0, distances=None):
    if distances is None:
        distances = pairwise_distances(data)
    return compute_sigma_perplexity(distances, perplexity)


def compute_rbf_adjacency_matrix(sigmas, distances):
    sigma_sum   = sigmas[:, None] + sigmas[None, :]          # ¦Ò_i + ¦Ò_j
    sigma_ij_sq = 0.25 * sigma_sum**2                        # ((¦Ò_i+¦Ò_j)/2)^2
    denom       = 2.0 * np.maximum(sigma_ij_sq, EPS)
    adjacency   = np.exp(- distances**2 / denom)
    return adjacency

def build_rbf_graph(data, perplexity=30.0):
    distances = pairwise_distances(data)
    sigmas    = compute_sigma_perplexity(distances, perplexity)
    adjacency = compute_rbf_adjacency_matrix(sigmas, distances)
    return adjacency, sigmas, distances

