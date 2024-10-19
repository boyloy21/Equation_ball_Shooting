import numpy as np
import pylab as plt

#gravity
g = 9.8 #[m/s**2]

alpha = float(input("Enter the angle in degress\n"))
alpha = np.radians(alpha)

# Initial velocity of projectile
u = float(input("Enter initial velocity in m/s\n"))

# Evaluating Range
R = u**2*np.sin(2*alpha)/g

# Evaluating max height
h = u**2*(np.sin(alpha))**2/(2*g)

# Creating an array of x with 20 points
x = np.linspace(0, R, 20)

# Solving for y
y = x*np.tan(alpha)-(1/2)*(g*x**2)/(u**2*(np.cos(alpha))**2)
print(h)
# Data plotting
plt.figure(1,dpi=300)
plt.plot(x,y,'r-',linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, h+1)
plt.show()
