import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lorenz system of differential equations
def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Set the parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Set the initial conditions
initial_conditions = [1.0, 0.0, 0.0]

# Set the time span
t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], num=100000)

# Solve the system of differential equations
sol = solve_ivp(
    lorenz,
    t_span,
    initial_conditions,
    args=(sigma, rho, beta),
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-8
)

# Extract the solution
x, y, z = sol.y

# Create a dataset as a numpy array
dataset = np.column_stack((x, y, z))

# Save the dataset to a file (e.g., CSV)
np.savetxt('lorenz_attractor_dataset.csv', dataset, delimiter=',', fmt='%.10f')


# Plot the Lorenz attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:1000], y[:1000], z[:1000], lw=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Lorenz Attractor')
plt.show()
