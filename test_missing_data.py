import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from LinearReg_MissingData import *

"""
Testing for linear regression implementation for missing input data.

P.L.Green
"""


def test_no_missing_data():
    """ Check that our code works with no missing data.

    """

    # Define data
    N = 50
    D = 2
    U = np.random.uniform(size=(N, D))
    theta = np.vstack(np.array([1., 2.]))
    sigma = 0.05
    Y = U @ theta + sigma * np.vstack(np.random.randn(N))

    # Prior
    mu_z = np.array([0.])
    sigma_z = np.array([1.])

    # Initial guess
    theta_init = np.array([1., 1.])

    lr_not_missing = LinearReg_MissingData(U, Y, N, D, theta_init,
                                           mu_z, sigma_z, sigma)
    lr_not_missing.train()

    assert np.allclose(theta, lr_not_missing.theta, atol=0.1)


def test_2D_partial_missing():
    """
    Test the case where inputs are 2D, where one point has a single
    missing component.

    First, we check that the closed form solution for the posterior over z
    matches what we get from evaluating prior * likelihood (up to a
    constant of proportionality).

    Secondly, we generate samples from the analytical posterior and use them
    to verify our closed form expressions for E[Z] and E[Z * Z^T].

    """

    # Define data
    N = 50
    D = 2
    U = np.random.uniform(size=(N, D))
    theta = np.vstack(np.array([1., 2.]))
    sigma = 0.1
    Y = U @ theta + sigma * np.vstack(np.random.randn(N))

    # Prior
    mu_z = np.array([0.])
    sigma_z = np.array([1.])

    # Initial guess
    theta_init = np.array([1., 1.])

    # Create some missing data
    U[2][1] = np.nan

    # Initialise and run a single expectation step
    lr = LinearReg_MissingData(U, Y, N, D, theta_init, mu_z, sigma_z, sigma)
    lr.expectation()

    # Identify indices that have missing data
    d = np.isnan(U[2, :])
    d_c = np.invert(d)

    # Define prior over z
    prior = norm(loc=mu_z, scale=sigma_z)

    # Define the model
    def f(u, theta):
        return u @ theta

    # Define likelihood
    def likelihood(u, y, theta):
        p = norm(loc=y, scale=sigma)
        return p.pdf(f(u, theta))

    # Array of z values over which we'll check our results
    z_array = np.linspace(-3, 3, 1000)

    # Initialise array that will validate posterior
    posterior_val = np.zeros(1000)

    # Expression for the posterior over z
    posterior = lr.posterior_z[2].pdf(z_array)
    posterior /= np.max(posterior)

    # To validate posterior expression
    for i in range(1000):
        U[2, d] = z_array[i]
        posterior_val[i] = (likelihood(U[2], Y[2], theta_init) *
                            prior.pdf(z_array[i]))

    posterior_val /= np.max(posterior_val)

    assert np.allclose(posterior, posterior_val)

    # Generate lots of samples of u, from z posterior
    z_samples = lr.posterior_z[2].rvs(1000)
    u_sample = np.zeros(2)
    u_sample[d_c] = U[2, d_c]

    # Realise Monte Carlo estimates of EU and EUU
    EU = np.zeros(2)
    EUU = np.zeros([2, 2])
    for i in range(1000):
        u_sample[d] = z_samples[i]
        EU += u_sample / 1000
        EUU += (np.array([u_sample]).T @
                np.array([u_sample]) / 1000)

    assert np.allclose(EU, lr.EU[2], atol=0.1)
    assert np.allclose(EUU, lr.EUU[2], atol=0.1)


def test_2D_fully_missing():
    """
    Test the case where inputs are 2D, where one point has both components
    missing.

    First, we check that the moments of the closed form solution for the
    posterior over z matches what we get if we importance sample from
    prior * likelihood (with prior as the proposal distribution).

    """

    # Define data
    N = 20
    D = 2
    U = np.random.uniform(size=(N, D))
    theta = np.vstack(np.array([1., 2.]))
    sigma = 0.1
    Y = U @ theta + sigma * np.vstack(np.random.randn(N))

    # Prior
    mu_z = np.array([0.])
    sigma_z = np.array([1.])

    # Initial guess
    theta_init = np.array([0.5, 0.5])

    # Create some missing data
    z_true = [U[2][0], U[2][1]]
    U[2][0] = np.nan
    U[2][1] = np.nan

    # Run linear regression
    lr = LinearReg_MissingData(U, Y, N, D, theta_init, mu_z, sigma_z, sigma)
    lr.train(Nitt=20)

    # Define the model
    def f(u, theta):
        return u @ theta

    # Define likelihood
    def likelihood(u, y, theta):
        p = norm(loc=y, scale=sigma)
        return p.pdf(f(u, theta))

    # Generate samples from the prior
    Ns = 10000
    prior_z = mvn(mean=np.repeat(mu_z, 2),
                  cov=np.diag(np.repeat(sigma_z**2, 2)))
    prior_Z_samples = prior_z.rvs(Ns)

    # Importance sample from the z posterior
    w = np.zeros(Ns)
    for i in range(Ns):
        w[i] = likelihood(u=prior_Z_samples[i, :], y=Y[2], theta=lr.theta)
    wn = w / np.sum(w)

    # IS estimate of E[z] posterior mean
    EZ_IS = wn @ prior_Z_samples

    # IS estimate of E[z z^T]
    EZZ_IS = np.zeros([2, 2])
    for i in range(Ns):
        EZZ_IS += (np.array([prior_Z_samples[i, :]]).T @
                   np.array([prior_Z_samples[i, :]]) * wn[i])

    # IS estimate of posterior covariance matrix
    prior_Z_samples -= EZ_IS    # Remove mean
    COV_Z_IS = np.zeros([2, 2])
    for i in range(Ns):
        COV_Z_IS += (np.array([prior_Z_samples[i, :]]).T @
                     np.array([prior_Z_samples[i, :]]) * wn[i])

    # Posterior over z from code
    posterior_z = lr.posterior_z[2]

    # Check that the importance sampled mean is close to closed-form
    # expression for the mean
    assert np.allclose(EZ_IS, posterior_z.mean, atol=0.1)

    # Check that the importance sampled estimate of z z^T is close to
    # closed-form expression
    assert np.allclose(EZZ_IS, lr.EUU[2], atol=0.2)

    # Check that the importance sampled covariance matrix is close to
    # closed-form expression for the covariance matrix
    assert np.allclose(COV_Z_IS, posterior_z.cov, atol=0.2)
