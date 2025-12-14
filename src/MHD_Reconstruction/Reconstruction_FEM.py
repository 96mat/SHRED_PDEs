#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import firedrake
import numpy as np
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
#%%
n=80
mesh=UnitSquareMesh(n, n)
fig, ax = plt.subplots()
triplot(mesh, axes=ax)
ax.legend(loc='upper left')
U=FunctionSpace(mesh,"RT",2) # space
# %%
data=np.load("SHRED_reconstruction.npz") # load data
print(data.files)
print(np.shape(data['arr_0']))
data=data['arr_0']
print(data.shape)
#%%
print("data.shape =", data.shape)   # dopo data = data['arr_0']
print("U.dim()    =", U.dim())
# %%

uh = Function(U)
x = data[:U.dim()]

with uh.dat.vec_wo as v: #temporal vector, stores values in "uh"
    v.setArray(x.T)
    v.assemble()

# %%
coords = mesh.coordinates.dat.data_ro  #Retrieving mesh coordinates
vals = np.asarray(uh.at(coords))       #In array uh @ mesh-coords

x = coords[:, 0]
y = coords[:, 1]
u = vals[:, 0]
v = vals[:, 1]

# ---- Rotation 90° CW ----
x_cw = y
y_cw = 1.0 - x

u_cw = v
v_cw = -u

speed = np.sqrt(u_cw**2 + v_cw**2) # for colormap

fig, ax = plt.subplots()
col=ax.quiver(x_cw, y_cw, u_cw, v_cw,speed,cmap="viridis")
plt.colorbar(col)
ax.set_aspect("equal")
plt.title("velocity (90° CW)")
plt.show()
# %%
import matplotlib.tri as mtri
Qcg = FunctionSpace(mesh, "CG", 1)
speed = Function(Qcg).project(sqrt(inner(uh, uh)))   # nodal scalar

# -----------------------------
# Extract vertices + triangle connectivity (from coordinate FunctionSpace)
# -----------------------------
coords = mesh.coordinates.dat.data_ro  # (Nverts, 2)

Vcoord = mesh.coordinates.function_space()
cells = Vcoord.cell_node_map().values  # typically shape (Ncells, 3) for triangles

# Make sure it's (Ncells, 3)
cells = np.asarray(cells)
if cells.ndim == 1:
    cells = cells.reshape(-1, 3)

# -----------------------------
# Rotate coordinates 90° CW on [0,1]^2: (x,y)->(y, 1-x)
# -----------------------------
x = coords[:, 0]
y = coords[:, 1]
x_cw = y
y_cw = 1.0 - x

tri = mtri.Triangulation(x_cw, y_cw, cells) #Triangulation generation for tripcolor()

# -----------------------------
# Plot rotated tripcolor
# -----------------------------
fig, ax = plt.subplots()
tc = ax.tripcolor(tri, speed.dat.data_ro, shading="gouraud", cmap="viridis")
plt.colorbar(tc, ax=ax)
ax.set_aspect("equal")
ax.set_title(r"$||u||_{L^2}$ (rotated 90° CW)")
plt.show()
#%%
