import numpy as np
import matplotlib.pyplot as plt


# Model parameters
M = 0.35     # Mass of ball in kg
g = 9.81    # Acceleration due to gravity (m/s^2)
# V = 80      # Initial velocity in m/s
# ang = 60.0  # Angle of initial velocity in radian
Cd = 0.47  # Drage coefficient
dt = 0.0    # time step in s

# You can check the variables by printing them
V = float(input("Enter Velocity to control ball:"))
ang = float(input("Enter Angle shooter:"))
ang = np.radians(ang)

# Set up the lists to store variables
# Initialize the velocity and position at t=0 (State)
t = [0]
vx = [V*np.cos(ang)]
vy = [V*np.sin(ang)]
x = [0]
y = [0.7]

# Drag force
drag = Cd*V**2  # drag force(F = -KV^2)

# Acceleration components [F = ma]
ax = [-(drag*np.cos(ang))/M]
ay = [-g-(drag*np.sin(ang))/M]

# Leave this out for students to try
# We can choose to have better control of the 
dt = 0.01

# Use Euler method to update variables
counter = 0
while (y[counter] >= 0):
    t.append(t[counter]+dt)

    # Update Velocity
    vx.append((vx[counter]+dt*ax[counter])) # Update
    vy.append((vy[counter]+dt*ay[counter]))

    # Update position
    x.append(x[counter]+dt*vx[counter])
    y.append(y[counter]+dt*vy[counter])

    # With the new velocity calculate the drag force
    vel = np.sqrt(vx[counter+1]**2 + vy[counter+1]**2)
    drag = Cd*vel**2
    ax.append(-(drag*np.cos(ang)/M))
    ay.append(-g-(drag*np.sin(ang)/M))

    # Increment the counter by 1
    counter = counter + 1
plt.figure(1,dpi=300)
plt.plot(x, y,'ro')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()
# Let's plot the trajectory
