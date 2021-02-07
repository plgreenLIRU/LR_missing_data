import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt

"""
A base class for linear regression with 1D output.

Notes on this implementation are in the 'EM algorithm' section of the
repository.

Note 1: currently, this assumes that each input dimension has the same
independent Gaussian prior over missing values.

Note 2: this implementation assumes that the model is linear in the
inputs and the parameters to be estimated. If nonlinear in the inputs,
we lose closed-form expressions for the posterior over z (the solution
in that case would be to a sampling scheme like MCMC, SMC etc.)

P.L.Green

"""


class LinearReg_MissingData():
    """
    Description
    -----------
    Linear regression with missing input data.

    Parameters
    ----------
    U - inputs, where missing parts are marked as 'nan'
    Y - observed outputs
    N - no. training points
    D - dimension of inputs
    theta_init - initial estimate of model parameters
    mu_z - mean of prior over z
    sigma_z - std of prior over z
    sigma - std of measurement noise
    Nitt - no. iterations of the EM algorithm used in training

    Methods
    -------
    expectation()
    maximisation()
    train()


    """

    def __init__(self, U, Y, N, D, theta_init, mu_z, sigma_z, sigma):

        self.U = np.vstack(U)
        self.Y = np.vstack(Y)
        self.theta = np.vstack(theta_init)
        self.N = N
        self.D = D

        # Identify indices that have missing data
        self.d = np.isnan(self.U)
        self.d_c = np.invert(self.d)

        # Prior over z
        self.mu_z = mu_z
        self.sigma_z = sigma_z

        # Noise standard deviation
        self.sigma = sigma

    def expectation(self):
        """
        Description
        -----------
        Finds the expected values of u and u * u^T

        """

        # Stores the expected values of u
        self.EU = np.copy(self.U)

        # Stores the expected values of u * u^T
        self.EUU = np.zeros([self.N, self.D, self.D])

        # Posterior distributions of z (we don't use these, but we output
        # them in case they are useful)
        self.posterior_z = []

        # Loop over all data points
        for n in range(self.N):

            # If there are missing values in this array
            if np.any(self.d[n]):

                # To simplify notation
                d = self.d[n]
                d_c = self.d_c[n]
                theta_d = self.theta[d]
                theta_d_c = self.theta[d_c]

                # 2D version of d
                D2 = np.array([d]).T @ np.array([d])

                # Create the mean and covariance matrix for our prior
                # over z. Note that, even if there's only one missing
                # component, Mu_z and Sigma_z will always be 2D.
                Mu_z = np.vstack(np.repeat(self.mu_z, np.sum(d)))
                Sigma_z = np.diag(np.repeat(self.sigma_z**2, np.sum(d)))

                # Posterior covariance matrix (note that this is the same
                # regardless of the number of known / missing points there
                # are).
                A = (self.sigma**-2 * theta_d @ theta_d.T +
                     inv(Sigma_z))

                # If the data point contains both missing and known values
                if np.any(self.d_c[n]):

                    # Known values
                    x = np.vstack(self.U[n, d_c])

                    # 2D version of d_c etc.
                    D2_c = np.array([d_c]).T @ np.array([d_c])
                    D2_cross = np.invert(D2 + D2_c)

                    # Needed to evaluate posterior mean (see notes)
                    b = (self.sigma**-2 * self.Y[n] * theta_d.T -
                         self.sigma**-2 * theta_d_c.T @ x @ theta_d.T +
                         Mu_z.T @ inv(Sigma_z)).T

                    # Find required expected functions of latent variables
                    EZ = inv(A) @ b
                    EZZ = inv(A) + EZ @ EZ.T

                    # Place E[x * x^T] in E[u * u^T]
                    self.EUU[n, D2_c] = (x @ x.T).flatten()
                    self.EUU[n, D2_cross] = np.repeat((x @ EZ.T).flatten(), 2)

                # If the data point only contains missing values
                else:

                    # Needed to evaluate posterior mean (see notes)
                    b = (self.sigma**-2 * self.Y[n] * theta_d.T -
                         Mu_z.T @ inv(Sigma_z)).T

                    # Find required expected functions of latent variables
                    EZ = inv(A) @ b
                    EZZ = inv(A) + EZ @ EZ.T

                # Place E[z] in E[u]
                self.EU[n, d] = EZ.T

                # Place E[z * z^T] in E[u * u^T]
                self.EUU[n, D2] = EZZ.flatten()

                # Save posterior over z
                self.posterior_z.append(mvn(mean=EZ.T[0], cov=inv(A)))

            else:
                self.posterior_z.append([])
                self.EUU[n] = np.array([self.U[n]]).T @ np.array([self.U[n]])

    def maximisation(self):
        """
        Description
        -----------
        Finds maximum likelihood values of model parameters

        """

        num = np.zeros([self.D, 1])
        den_inv = np.zeros([self.D, self.D])
        for n in range(self.N):
            num += self.Y[n] * np.vstack(self.EU[n])
            den_inv += self.EUU[n]

        self.theta = inv(den_inv) @ num

    def train(self, Nitt=10):
        """
        Description
        -----------
        Trains model using the EM algorithm

        Parameters
        ----------
        Nitt - no. EM iterations used in training
        """

        self.theta_store = np.zeros([Nitt + 1, self.D])
        self.theta_store[0] = self.theta.T

        for i in range(Nitt):
            self.expectation()
            self.maximisation()
            self.theta_store[i + 1] = self.theta.T
