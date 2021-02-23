import numpy as np
import matplotlib.pyplot as plt

#create x and y between -10 and 10 with the defference of 0.5
x=np.arange(-10,10.1,0.5)
y=np.arange(-10,10.1,0.5)

print(x.shape)
print(y.shape)

print(x)
print(y)

# for changing 1D array to 2D array
X,Y=np.meshgrid(x,y)

#you can see it :)
print(x.shape)
print(y.shape)

Z = X ** 2 + Y ** 2

fig = plt.figure(figsize=(12, 8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.rainbow)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
fig.savefig('surface1.png')
fig.savefig('surface1.pdf')


fig = plt.figure(figsize=(12, 8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.rainbow)
cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=plt.cm.rainbow)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
fig.savefig('surface2.png')