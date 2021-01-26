# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# t в эВ
t = 2.7
kx = np.linspace(-4*np.pi/3, 4*np.pi/3, 100)
ky = np.linspace(-4*np.pi/3, 4*np.pi/3, 100)
# kx,ky  в еденицах постоянной решётки a = 1.42 A
kx, ky = np.meshgrid(kx, ky)
E = t*np.sqrt(3 +2*np.cos(np.sqrt(3)*ky)\
                            + 4*np.cos(np.sqrt(3)/2*ky)*np.cos(3/2*kx) )

# Plot the surface
ax.plot_surface(kx, ky, E, color='b')
Em = - t*np.sqrt(3 +2*np.cos(np.sqrt(3)*ky)\
                            + 4*np.cos(np.sqrt(3)/2*ky)*np.cos(3/2*kx) )
ax.plot_surface(kx, ky, Em, color='c')
plt.show()



