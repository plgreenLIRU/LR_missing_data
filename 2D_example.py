import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from LinearReg_MissingData import *

"""
Linear regression applied to a 2D problem, where 2 of the inputs have
a single component missing. The code plots parameter convergence and
the posterior distributions of the missing parts of the data.

P.L.Green

p.l.green@liverpool.ac.uk
engineeringdataanalytics.com

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
z_true = [U[2][1], U[10][0]]
U[2][1] = np.nan
U[10][0] = np.nan

# Run linear regression
lr = LinearReg_MissingData(U, Y, N, D, theta_init, mu_z, sigma_z, sigma)
lr.train()

# Plot posterior of missing data
Z_range = np.linspace(-1, 2, 1000)
p1 = lr.posterior_z[2].pdf(Z_range)
p2 = lr.posterior_z[10].pdf(Z_range)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(Z_range, p1, color='black')
ax[0].plot(z_true[0], 0, 'o', color='green',
           label='Actual value of missing data point')
ax[0].set_ylabel('$p(z | y)$')
ax[0].legend()
ax[1].plot(Z_range, p2, color='black')
ax[1].plot(z_true[1], 0, 'o', color='green')
ax[1].set_ylabel('$p(z | y)$')
ax[1].set_xlabel('$z$')
plt.tight_layout()

# Plot parameter convergence
fig, ax = plt.subplots(nrows=2)
ax[0].plot(lr.theta_store[:, 0], color='black', label='Max. likelihood')
ax[0].plot(np.array([0, 10]), np.repeat(theta[0], 2), 'green', label='True')
ax[0].set_ylabel('$\\theta_1$')
ax[0].legend()
ax[1].plot(lr.theta_store[:, 1], color='black')
ax[1].plot(np.array([0, 10]), np.repeat(theta[1], 2), 'green')
ax[1].set_ylabel('$\\theta_2$')
ax[1].set_xlabel('Iteration')

plt.show()
