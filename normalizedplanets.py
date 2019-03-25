import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import scipy.ndimage

numcosos = 9

def modul(x,y,z):
	return (x**2+y**2+z**2)**(1/2)
def modul2(x,y,z):
	return (x**2+y**2+z**2)
def modul3(x,y,z):
	return (x**2+y**2+z**2)**(3/2)


##################################(data)
masasol = 1.989e30
masaterra = 5.972e24
masamarte = 6.39e23
masamercurio = 3.285e23
masavenus = 4.867e24
masajupiter = 1.898e27
masasaturno = 5.683e26
masaurano = 8.681e26
masaneptuno = 1.024e26

posmercurix = -3.0035e-1
posmercuriy = -3.3763e-1
posmercuriz = -3.5458e-5

velmercurix = 1.5307e-2
velmercuriy = -1.7403e-2
velmercuriz = -2.8264e-3

posvenusx = -5.6277e-1
posvenusy = 4.4506e-1
posvenusz = 3.8583e-2

velvenusx = -1.2623e-2
velvenusy = -1.5967e-2
velvenusz = 5.0938e-4

posterrax = -1.8795e-1
posterray = 9.6517e-1
posterraz = -4.5855e-5

velterrax = -1.7353e-2
velterray = -3.38941e-3
velterraz = 4.8569e-7

posmartx = 1.0814
posmarty = 9.7292e-1
posmartz = -6.1494e-3

velmartx = -8.8277e-3
velmarty = 1.1597e-2
velmartz = 4.5961e-4

posjupiterx = -2.1274
posjupitery = -4.9079
posjupiterz = 6.798e-2

veljupiterx = 6.839e-3
veljupitery = -2.6486e-3
veljupiterz = -1.42044e-4

possaturnox = 1.9647
possaturnoy = -9.866
possaturnoz = 9.3293e-2

velsaturnox = 5.1712e-3
velsaturnoy = 1.0708e-3
velsaturnoz = -2.2456e-4

posuranox = 1.7014e1
posuranoy = 1.024e1
posuranoz = -1.8226e-1

veluranox = -2.0512e-3
veluranoy = 3.1833e-3
veluranoz = 3.8336e-5

posneptunox = 2.8983e1
posneptunoy = -7.4855
posneptunoz = -5.1387e-1

velneptunox = 7.7096e-4
velneptunoy = 3.0558e-3
velneptunoz = -8.0881e-5


###########################################

cmas = [masasol, masamercurio, masavenus, masaterra, masamarte, masajupiter, masasaturno, masaurano, masaneptuno] 

cpos = [[0,0,0],
		[posmercurix, posmercuriy, posmercuriz],
		[posvenusx, posvenusy, posvenusz],
		[posterrax, posterray, posterraz],
		[posmartx, posmarty, posmartz],
		[posjupiterx, posjupitery, posjupiterz],
		[possaturnox, possaturnoy, possaturnoz],
		[posuranox, posuranoy, posuranoz],
		[posneptunox, posneptunoy, posneptunoz]]

cvel = [[0,0,0],
		[velmercurix, velmercuriy, velmercuriz],
		[velvenusx, velvenusy, velvenusz],
		[velterrax, velterray, velterraz],
		[velmartx, velmarty, velmartz],
		[veljupiterx, veljupitery, veljupiterz],
		[velsaturnox, velsaturnoy, velsaturnoz],
		[veluranox, veluranoy, veluranoz],
		[velneptunox, velneptunoy, velneptunoz]]

##################################(normalisation)

rnorm = 1
mnorm = 1.989e30
tnorm = 360.25

for i in range(numcosos):
	for j in range(3):
		cpos[i][j] = (cpos[i][j])/rnorm
		cvel[i][j] = (cvel[i][j])*tnorm/rnorm
	cmas[i] = cmas[i]/mnorm


def accx(x, y, z, x1, y1, z1, m):
	return (-4*(np.pi**2)*m*(x-x1)/(modul3(x-x1, y-y1, z-z1)))

def accy(x, y, z, x1, y1, z1, m):
	return (-4*(np.pi**2)*m*(y-y1)/(modul3(x-x1, y-y1, z-z1)))

def accz(x, y, z, x1, y1, z1, m):
	return (-4*(np.pi**2)*m*(z-z1)/(modul3(x-x1, y-y1, z-z1)))	

#####################################(runge kutta step-size conditions)

iterations = 2100
# h = (tmax-tmin)/iterations #pas runge-kutta
h = 1e-3

#####################################

xpos = np.zeros((numcosos, iterations))
ypos = np.zeros((numcosos, iterations))
zpos = np.zeros((numcosos, iterations))

xvel = np.zeros((numcosos, iterations))
yvel = np.zeros((numcosos, iterations)) 
zvel = np.zeros((numcosos, iterations))

veltotal = np.zeros((numcosos, iterations, 3))
postotal = np.zeros((numcosos, iterations, 3))

for k in range(iterations):
	for i in range(1, numcosos):

		k1vx, k1vy, k1vz = 0,0,0
		k2vx, k2vy, k2vz = 0,0,0
		k3vx, k3vy, k3vz = 0,0,0
		k4vx, k4vy, k4vz = 0,0,0	

		for j in range(numcosos):
			if j != i:

				k1vx += accx(cpos[i][0], cpos[i][1], cpos[i][2], cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k1vy += accy(cpos[i][0], cpos[i][1], cpos[i][2], cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k1vz += accz(cpos[i][0], cpos[i][1], cpos[i][2], cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])

		k1rx = cvel[i][0]
		k1ry = cvel[i][1]
		k1rz = cvel[i][2]

		for j in range(numcosos):
			if j != i:

				k2vx += accx(cpos[i][0]+(h/2)*k1rx, cpos[i][1]+(h/2)*k1ry, cpos[i][2]+(h/2)*k1rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k2vy += accy(cpos[i][0]+(h/2)*k1rx, cpos[i][1]+(h/2)*k1ry, cpos[i][2]+(h/2)*k1rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k2vz += accz(cpos[i][0]+(h/2)*k1rx, cpos[i][1]+(h/2)*k1ry, cpos[i][2]+(h/2)*k1rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])

		k2rx = cvel[i][0] + (h/2)*k1vx
		k2ry = cvel[i][1] + (h/2)*k1vy
		k2rz = cvel[i][2] + (h/2)*k1vz

		for j in range(numcosos):
			if j != i:

				k3vx += accx(cpos[i][0]+(h/2)*k2rx, cpos[i][1]+(h/2)*k2ry, cpos[i][2]+(h/2)*k2rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k3vy += accy(cpos[i][0]+(h/2)*k2rx, cpos[i][1]+(h/2)*k2ry, cpos[i][2]+(h/2)*k2rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k3vz += accz(cpos[i][0]+(h/2)*k2rx, cpos[i][1]+(h/2)*k2ry, cpos[i][2]+(h/2)*k2rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])

		k3rx = cvel[i][0] + (h/2)*k2vx
		k3ry = cvel[i][1] + (h/2)*k2vy
		k3rz = cvel[i][2] + (h/2)*k2vz

		for j in range(numcosos):
			if j != i:

				k4vx += accx(cpos[i][0]+h*k3rx, cpos[i][1]+h*k3ry, cpos[i][2]+h*k3rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k4vy += accy(cpos[i][0]+h*k3rx, cpos[i][1]+h*k3ry, cpos[i][2]+h*k3rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])
				k4vz += accz(cpos[i][0]+h*k3rx, cpos[i][1]+h*k3ry, cpos[i][2]+h*k3rz, cpos[j][0], cpos[j][1], cpos[j][2], cmas[j])

		k4rx = cvel[i][0] + h*k3vx
		k4ry = cvel[i][1] + h*k3vy
		k4rz = cvel[i][2] + h*k3vz

		cvel[i][0] += (h/6)*(k1vx + 2*k2vx + 2*k3vx + k4vx)
		cvel[i][1] += (h/6)*(k1vy + 2*k2vy + 2*k3vy + k4vy)
		cvel[i][2] += (h/6)*(k1vz + 2*k2vz + 2*k3vz + k4vz)

		cpos[i][0] += (h/6)*(k1rx + 2*k2rx + 2*k3rx + k4rx)
		cpos[i][1] += (h/6)*(k1ry + 2*k2ry + 2*k3ry + k4ry)
		cpos[i][2] += (h/6)*(k1rz + 2*k2rz + 2*k3rz + k4rz)

		xvel[i][k] = cvel[i][0]
		yvel[i][k] = cvel[i][1]
		zvel[i][k] = cvel[i][2]

		xpos[i][k] = cpos[i][0] #+= o =?
		ypos[i][k] = cpos[i][1]
		zpos[i][k] = cpos[i][2]



##############################################(conservacio del moment angular)

veltotal = np.zeros((numcosos, iterations, 3))
postotal = np.zeros((numcosos, iterations, 3))

for i in range(numcosos):
	for j in range(iterations):
		postotal[i][j][0] = xpos[i][j]
		postotal[i][j][1] = ypos[i][j]
		postotal[i][j][2] = zpos[i][j]

for i in range(numcosos):
	for j in range(iterations):
		veltotal[i][j][0] = xvel[i][j]
		veltotal[i][j][1] = yvel[i][j]
		veltotal[i][j][2] = zvel[i][j]

momentum = np.zeros((numcosos, iterations, 3))#3?   
for i in range(numcosos):
	for j in range(iterations):
		momentum[i][j] = np.cross(postotal[i][j], cmas[i]*veltotal[i][j])

modulmoment = np.zeros((numcosos, iterations))
for i in range(numcosos):
	for j in range(iterations):
		modulmoment[i][j] = modul(momentum[i][j][0], momentum[i][j][1], momentum[i][j][2])

itera = np.linspace(0, iterations, iterations)

plt.plot(itera, modulmoment[1], label = 'mercuri')
plt.plot(itera, modulmoment[2], label = 'venus')
plt.plot(itera, modulmoment[3], label = 'terra')
plt.plot(itera, modulmoment[4], label = 'mart')
plt.legend()
plt.xlabel('Iteracions')
plt.ylabel(r'$\|\vec{L}\|$')
plt.show()




##############################################(aphelion, perihelion)

rmax = modul(posmartx, posmarty, posmartz)
rmin = modul(posmartx, posmarty, posmartz)
imin = 0
imax = 0
for i in range(iterations):
	if modul(xpos[4][i],ypos[4][i],zpos[4][i]) >rmax:
		rmax = modul(xpos[4][i],ypos[4][i],zpos[4][i])
		imax = i
	if modul(xpos[4][i],ypos[4][i],zpos[4][i]) <rmin:
		rmin = modul(xpos[4][i],ypos[4][i],zpos[4][i])
		imin = i

print(rmax,rmin)

##############################################(error)

def potencial(m, r):
	return (4*np.pi**2)*m/r**2

pot = np.zeros((iterations))
ecin = np.zeros((iterations))
etot = np.zeros((iterations))
for i in range(iterations):
	for j in range(numcosos):
		if j != 3:
			pot[i] += potencial(cmas[j], modul(xpos[3][i]-xpos[j][i], ypos[3][i]-ypos[j][i], zpos[3][i]-zpos[j][i])) 
	ecin[i] = 0.5*cmas[3]*(modul(xvel[3][i], yvel[3][i], zvel[3][i]))**2				
	etot[i] = pot[i]+ecin[i]

errortot = (max(etot)-min(etot))/max(etot)
erroriter = abs(etot[0]-etot[1])/etot[0]
print(erroriter, errortot)

plt.plot(itera, etot)
plt.xlabel('Iteracions')
plt.ylabel('E')
plt.show()

################################(3dplot)

fig = plt.figure()
ax = fig.gca(projection= '3d')

# ax.plot(xpos[0], ypos[0], zpos[0], label = 'sol')
ax.plot(xpos[1], ypos[1], zpos[1], label = 'mercurio')
ax.plot(xpos[2], ypos[2], zpos[2], label = 'venus')
ax.plot(xpos[3], ypos[3], zpos[3], label = 'tierra')
ax.plot(xpos[4], ypos[4], zpos[4], label = 'marte')
# # ax.plot(xpos[5], ypos[5], zpos[5], label = 'jupiter')
# # ax.plot(xpos[6], ypos[6], zpos[6], label = 'saturno')
# # ax.plot(xpos[7], ypos[7], zpos[7], label = 'urano')
# # ax.plot(xpos[8], ypos[8], zpos[8], label = 'neptuno')

ax.set_zlim3d(-1,1)
ax.set_xlabel(r'$\tilde{x}$')
ax.set_ylabel(r'$\tilde{y}$')
ax.set_zlabel(r'$\tilde{z}$')

plt.legend()
plt.show()