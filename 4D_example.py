import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from LinearReg_MissingData import *

"""
Linear regression applied to a 4D problem, where 3 of the inputs have
a single component missing. The code plots parameter convergence and
the posterior distributions of the missing parts of the data.

P.L.Green

p.l.green@liverpool.ac.uk
engineeringdataanalytics.com

"""

# Define data
N = 2000
D = 4
U = np.random.uniform(size=(N, D))
theta = np.vstack(np.array([3., 2., 3., 4.]))
sigma = 0.1
Y = U @ theta + sigma * np.vstack(np.random.randn(N))

# Prior
mu_z = np.array([1.])
sigma_z = np.array([0.5])

# Initial guess
theta_init = np.array([0.5, 0.5, 0.5, 0.5])

# Create some missing data
z_true = [U[2][0], U[5][1], U[10][2]]
U[2][0] = np.nan
U[5][1] = np.nan
U[10][2] = np.nan

# Run linear regression
lr = LinearReg_MissingData(U, Y, N, D, theta_init, mu_z, sigma_z, sigma)
lr.train(Nitt=30)

# Plot posterior of missing data
Z_range = np.linspace(-1, 2, 1000)
p1 = lr.posterior_z[2].pdf(Z_range)
p2 = lr.posterior_z[5].pdf(Z_range)
p3 = lr.posterior_z[10].pdf(Z_range)

fig, ax = plt.subplots(nrows=3)
ax[0].plot(Z_range, p1, color='black')
ax[0].plot(z_true[0], 0, 'o', color='green',
           label='Actual value of missing data point')
ax[0].set_ylabel('$p(z | y)$')
ax[0].legend()
ax[1].plot(Z_range, p2, color='black')
ax[1].plot(z_true[1], 0, 'o', color='green')
ax[1].set_ylabel('$p(z | y)$')
ax[1].set_xlabel('$z$')
ax[2].plot(Z_range, p3, color='black')
ax[2].plot(z_true[2], 0, 'o', color='green')
ax[2].set_ylabel('$p(z | y)$')
ax[2].set_xlabel('$z$')
plt.tight_layout()

# Plot parameter convergence
fig, ax = plt.subplots(nrows=3)
ax[0].plot(lr.theta_store[:, 0], color='black', label='Max. likelihood')
ax[0].plot(np.array([0, 30]), np.repeat(theta[0], 2), 'green', label='True')
ax[0].set_ylabel('$\\theta_1$')
ax[0].legend()
ax[1].plot(lr.theta_store[:, 1], color='black')
ax[1].plot(np.array([0, 30]), np.repeat(theta[1], 2), 'green')
ax[1].set_ylabel('$\\theta_2$')
ax[1].set_xlabel('Iteration')
ax[2].plot(lr.theta_store[:, 2], color='black')
ax[2].plot(np.array([0, 30]), np.repeat(theta[2], 2), 'green')
ax[2].set_ylabel('$\\theta_3$')
ax[2].set_xlabel('Iteration')

plt.show()
