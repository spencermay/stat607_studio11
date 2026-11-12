"""Starter for Studio 11.

Replace each _todo(...) call with your code. 
1. For Exercise 1, vectorize the computations over observations. 
2. For Exercise 2, revise the component terms, E-step, and log 
   likelihood to use the log-sum-exp trick.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from naive_em_algorithm import GaussianMixtureModel


def _todo(task):
    raise NotImplementedError(task)


class GaussianMixtureModelVectorized(GaussianMixtureModel):
    """Use the EM loop, initialization, and sampling from GaussianMixtureModel."""

    def _component_terms(self, X):
        """Build the (n_samples, n_components) array shared by the E-step and score."""
        terms = np.empty((len(X), self.n_components))
        for k in range(self.n_components):
            # In Exercise 1, evaluate the weighted Gaussian density for all
            # rows of X using self.pi_[k], self.mu_[k], and self.cov_[k].
            # In Exercise 2, compute the log-weighted density using logpdf.
            terms[:, k] = self.pi_[k] * multivariate_normal.pdf(X, mean = self.mu_[k], cov = self.cov_[k])
        return terms

    def _e_step(self, X):
        """Return responsibilities of shape (n_samples, n_components)."""
        terms = self._component_terms(X)
        # In Exercise 1, normalize each observation's row to sum to one.
        # In Exercise 2, use logsumexp and return probabilities on the original scale.
        return terms / terms.sum(axis=1, keepdims=True)

    def _m_step(self, X, responsibilities):
        """Update the weights, means, and covariance matrices of all components."""
        n_samples, n_features = X.shape
        n_k = np.sum(responsibilities, axis=0)  # (K,)
        weights = n_k / n_samples # (K,)
        means = np.einsum("ik,id->kd",responsibilities, X) # (K, d)

        covariances = np.empty((self.n_components, n_features, n_features))
        ridge = 1e-6 * np.eye(n_features)  # Matches the covariance update in the starter.
        for k in range(self.n_components):
            centered = X - means[k]
            covariances[k] = np.einsum("i,ji->ji", responsibilities[:,k], np.outer(centered, centered))
            covariances[k] += ridge

        self._set_parameters(means, covariances, weights)

    def _compute_log_likelihood(self, X):
        """Return the total observed-data log likelihood as a scalar."""
        terms = self._component_terms(X)
        # In Exercise 1, combine the densities for each observation, then take logs.
        # In Exercise 2, combine the log-weighted densities using logsumexp.
        return _todo("Sum the per-observation log likelihoods.")
