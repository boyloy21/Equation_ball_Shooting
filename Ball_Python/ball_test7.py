import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Learn data
data = pd.read_csv('Estimate/data.csv')
x = data.iloc[:,0:1].values
y = data.iloc[:,1:].values

lin_regs = LinearRegression()
lin_regs.fit(x,y)
poly_regs = PolynomialFeatures(degree=2)
x_poly = poly_regs.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)


# Equation of shooting ball
# X = Xo + Vox*t  : Vox = Vo*cos(theta)
# Y = Yo + Voy*t - (1/2)*g*t**2  : Voy*sin(theta)
# Vx = Vox , Vy = Voy - gt
# V = np.sqrt(Vx**2 + Vy**2)
# t = X/Vx , 0 = h - g(X/Vo)^2/2  => h= g(X/Vo)^2/2 => 2*h/(g*X^2) = 1/Vo^2
# Vo = np.sqrt(g*X^2/2*h )
###  Find Velocity Start Vo  ###
D = float(input("Distance to goal:"))
theta = float(input("Angle start:"))
theta = np.radians(theta)
r = 0.085 #[m] radius of ball
rho = 1.27 #[kg/m^3] dentity
cd = 0.47 # For sphere drag coefficient
A = 4*np.pi*r**2 # [m^2] surface of cross object
m = 0.35 # [kg] mass of ball
g = 9.81 # [m/s^2] gravity
k = (1/2)*rho*A*cd # gain
# k = (rho*cd*A)/(2*m*g)
b = (g*m)/k
h = 0.35 # height
V_T = np.sqrt((2*m*g)/(rho*A*cd))
# F_ball = m*g - k*V**2 <=> ma = mg -kv^2
# Time 
# t = D/Vo*np.cos(theta) # take from X_direction
# t = np.sqrt(2*h/g)    # take from Y_direction

# V = V_T*np.tanh(g*t/V_T) # with drag
# V = np.sqrt(m*g/k)
# V = g*t  # no air resistance free  fall

# sum_F : F_newton = F + F_drag
# ma = mg - bv
# dv/dt = g - bv/m = (b/m)(g*m/b - v) => (dv/(g*m/b - v)) = b/m*dt
# Integral : ln((V' - mg/b)/-mg/b) = -bt'/m
# => V'(t) = mg/b(1-exp(-bt'/m))
# In short time limit t<<tau = m/b => exp(-bt'/m) may= 1 - bt'/m
# => V'(t) = gt

# No drag (F_dragr=0)
'''
    H = (Vo^2sin^2(theta))/2g
    T = 2*(Vo*sin(theta))/g = 2*np.sqrt(2H/g)
    Va = Vo*cos(theta)
    L = (Vo^2sin(2theta))/g = VaT
    ta = (Vosin(theta))/g = T/2
    Xa = L/2 = np.sqrt(LHcot(theta))
    theta_1 = -theta=-arctan[LH/(L-Xa)^2]
    V1 = Vo
    => Vo = np.sqrt((L*g)/(2*sin(theta)*cos(theta))
'''
Vo = np.sqrt((D*g)/(2*np.sin(theta)*np.cos(theta)))
# Quadratic drag force(F_drag=mgkV^2)
'''
    k = (rho*Cd*A)/(2*m*g)
    H = (Vo^2sin^2(theta))/g*(2+kVo^2sin(theta))
    T = 2*np.sqrt(2H/g)
    Va = (Vo*cos(theta)/np.sqrt(1+kVo^2(sin(theta)+cos^2(theta)*ln*tan(theta/2+pi/4))))
    L = Va*T
    ta = (T-kHVa)/2
    Xa = np.sqrt(LHcot(theta))
    theta1 = -arctan[LH/(L-Xa)^2]
    V1 = V(theta_1)
    => Vo = np.sqrt((gX^2)/(8*H*cos^2(theta)-X^2*k(sin(theta) + cos^2(theta)*ln(tan(theta/2 + np.pi/4))))))
'''
# H = lin_reg_2.predict(poly_regs.fit_transform([[D]])) + 0.6
H = 0.002357*(D**3) + 0.001569*(D**2) + 0.00487*D - (9.91/(10**5)) + 0.35
print(H)
# H = ((Vo**2)*(np.sin(theta)**2))/2*g
# H = 0.6
# H = 0.05904*D - 0.07528 
# if (D>0.0 and D<=1.0):
#    H = 0.05 
# elif (D>1.0 and D<=2.0):
#    H = 0.08
# elif (D>2.0 and D<=3.0):
#    H = 0.1
# elif (D>3.0 and D<=4.0):
#    H = 0.15
# elif (D>4.0 and D<=5.0):
#    H = 0.20
# elif (D>4.0 and D<=5.0):
#    H = 0.25
# elif (D>5.0 and D<=6.0):
#    H = 0.30
# elif (D>6.0 and D<=7.0):
#    H = 0.35
# elif (D>7.0 and D<=8.0):
#    H = 0.4
# elif (D>8.0 and D<=9.0):
#    H = 0.45
# elif (D>9.0 and D<=10.0):
#    H = 0.5
# Y = ax + b
# H = 0.0646095*D + (-0.1146)
# P = ax^2 + bx + c
# a = 0.01206
# b = -0.01883
# c = 0.01545
# degree3:
# P = ax^3 + bx^2 + cx + d 
# a = -0.007135
# b = 0.097
# c = 0.06514
# d = -0.0153
Vo = 167.9030*D+157.34028
# Vo = np.sqrt(((D**2)*g)/(8*H*((np.cos(theta))**2) - (D**2)*k*g*(np.sin(theta) + ((np.cos(theta))**2)*(2.303*np.log(np.tan(theta/2 + np.pi/4))))))
omega = Vo/0.025

print(Vo,omega)
print(H)
# Simulation graph
# t = [0]
# vx = [Vo*np.cos(theta)]
# vy = [Vo*np.sin(theta)]
# x = [0]
# y = [0.6]
# for i in range(100):
# theta = np.linspace(np.radians(15),-np.radians(15)+np.pi,100)
# x = Vo*np.cos(theta)*t
# y = 0.6 + Vo*np.sin(theta)*t - (1/2)*g*t**2
drag = k*Vo**2  # drag force(F = -KV^2)

# Acceleration components [F = ma]
# ax = [-(drag*np.cos(theta))/m]
# ay = [-g-(drag*np.sin(theta))/m]
h = H - 0.6
dt = 2*np.sqrt(2*H/g) 
t = np.linspace(0,dt,100)
x = Vo*np.cos(theta)*t 
y = 0.6 + Vo*np.sin(theta)*t - 0.5*g*t**2

# Use Euler method to update variables
# counter = 0
# while (y[counter] >= 0):
#     t.append(t[counter]+dt)

#     # Update Velocity
#     vx.append((vx[counter]+dt*ax[counter])) # Update
#     vy.append((vy[counter]+dt*ay[counter]))

#     # Update position
#     x.append(x[counter]+dt*vx[counter])
#     y.append(y[counter]+dt*vy[counter])

#      # With the new velocity calculate the drag force
#     vel = np.sqrt(vx[counter+1]**2 + vy[counter+1]**2)
#     drag = k*vel**2
#     ax.append(-(drag*np.cos(theta)/m))
#     ay.append(-g-(drag*np.sin(theta)/m))

#     # Increment the counter by 1
#     counter = counter + 1

plt.figure(1,dpi=300)
plt.plot(x,y,"r.", label="Tracking ball")
plt.title("Velocity of shooting ball [m/s]: %f" %(Vo))
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


