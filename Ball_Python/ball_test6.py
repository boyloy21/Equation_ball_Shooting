import numpy as np
import matplotlib.pyplot as plt


distance = float(input("Enter distance:"))  # Xaxis go to X_end
theta = float(input("Enter angle:"))
theta = np.radians(theta)
X_start = 0.0
Y_start = 0.0
X_end = 0.0
Y_end = 0.0
r = 0.085 # [m]
rho = 1.2 #[kg/m^3]
A = np.pi*r**2
h = 0.35  # Yaxis start at h
g = 9.81
m = 0.2 # mass of ball
b = 0.6
k = 1.225 # Air resistance constant (tune this value as needed, depending on the projectile and environment)
cd = 0.47
X_start = 0.0
Y_start = h
X_end = distance
Y_end = 0.0
def map(Input, min_input, max_input, min_output, max_output):
    value = ((Input - min_input)*(max_output-min_output)/(max_input - min_input) + min_output)
    return int(value)
# find Vo and t : y = y0 + Vot + 1/2*a*t^2, a = -g
# MethodeI: h = (tan(theta)x - (g/2(Vo*cos(theta))^2)x^2
# (-h + (tan(theta)x)*2/gx^2 = 1/(Vo*cos(theta))^2
# Vo = np.sqrt(cos(theta)^2*g*x^2)/(2*(x*tan(theta) - h))
# Vo = np.sqrt((9.8*(distance**2))/(distance*np.tan(theta) - h))
omega = 167.9030*distance+157.34028
# MethodeII: Me
# Vox = Vo*np.cos(theta)
# Voy = Vo*np.sin(theta)
# t = np.sqrt(2*h/g)
# # Horizontal: Vox = distance/t
# Vox = distance/t
# # Vertical: Voy = -h + (1/2)*g*t
# Voy = -h + (1/2)*g*t

# Vo = np.sqrt(Vox**2+ Voy**2)
# omega = (Vo/0.01)-50
# MethodIII : distance = Vo^2*sin(2*theta)/g

# Vo = np.sqrt((distance*g)/np.sin(2*theta))

# MethodIV: 
# Vo = np.sqrt((((4.9)*(distance**2)*(np.cos(theta))**2)/((h)+distance*np.sin(theta)*np.cos(theta))))

# MethodeV:
# X = Vcos(theta)*t => t = X/V*cos(theta)
# Y = Yo + Vyt - (1/2)gt^2 : take Yo = h => Y = h + Vsin(theta)*t -(1/2)gt^2 (2)
# take (1) to (2): Y = h + X*tan(theta) - (1/2)g(X/V*cos(theta))^2
# => Y - h - X*tan(theta) = (gX^2/2*V^2*(cos(theta))^2)
# => 2*V^2*(cos(theta))^2 = (gX^2)/(Y - h - X*tan(theta))
# => V = np.sqrt((gX^2)/2*(cos(theta))^2(Y - h - X*tan(theta)))
# Vo = (1/np.cos(theta))*(np.sqrt(((g/2)*(distance**2))/(-Y_end + h + distance*np.tan(theta))))


# MethodeVI = Add resistance air
# Vo = (distance*cd)/(m*np.cos(theta))
# Vo = np.sqrt((g*distance**2)/(2*h))
# Vo = distance/(np.cos(theta)*(np.sqrt(2*h/g)))

# MethodeVII = Add Force drage
#y(t)= (mg/b)t + (m^2g/b)[e^(-x)-1]
#t = (x/Vo)*cos(theta)
#=> y = (mg/b)*(x/Vo)*cos(theta) + (m^2g/b)[e^(-x)-1] = (mg*x*cos(theta))/(b*Vo) + (m^2g/b)[e^(-x)-1]
#=> Vo = (mg*x*cos(theta))/b*(y-(m^2g/b)[e^(-x)-1])
# Vo = (m*g*distance*(np.cos(theta)))/(b*h-((m**2)*g*(np.exp(-distance)-1)))

# MethodeVI = Me take Friction air 
# Model physicals of Project
# F = F + F_drag
# ma = mg - kV => V= {vx,vy} , a= {ax,ay} 
# ax = Vx/t, ay = Vy/t , ax = Fx/m , ay = Fy/m
# => y = ((bvy + mg)x)/bvx + (m^2g/b^2)ln(1-(kx/mvx))
# vx = Vocos(theta) , vy = Vosin(theta)
# => y = ((bvsin(theta)+mg)*x)/bvcos(theta) + (m^2g/b^2)ln(1-kx/mvcos(theta))
# => y = (bvtan(theta) + (mg/vcos(theta))*x) + (m^2g/b^2)ln(1-kx/mvcos(theta))
# => 

# omega = (Vo/0.025)
pwm= int(map(omega, 0, 1200, 0, 1000))
print(omega)
print(pwm)