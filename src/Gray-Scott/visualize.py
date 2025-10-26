#%%
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
# %%
import os
print("Working directory:", os.getcwd()) #working directory python
# %%
# print the list of paths of files in the training set
set_path = "train"
paths = sorted(glob.glob(f"/home/orion/SHRED_tests/GRSReacDiff/datasets/datasets/gray_scott_reaction_diffusion/data/{set_path}/*.hdf5"))
print(paths)
print(len(paths))
# %%

# select the first path
p = paths[2]

# print the first layer of keys
with h5py.File(p, "r") as f:
    print(f.keys())
#%%
# In 'boundary_conditions' is stored the information about the boundary conditions:
with h5py.File(p, "r") as f:
    print("print bc available:", f["boundary_conditions"].keys())
    print("print attributes of the bc:", f["boundary_conditions"]["x_periodic"].attrs.keys())
    print("get the bc type:", f["boundary_conditions"]["x_periodic"].attrs["bc_type"])
# %%# Reminder: 't0_fields', 't1_fields', 't2_fields' are respectively scalar fields, vector fields and tensor fields
# print the different fields available in the dataset
with h5py.File(p, "r") as f:
    print("t0_fields:", f["t0_fields"].keys())
    print("t1_fields:", f["t1_fields"].keys())
    print("t2_fields:", f["t2_fields"].keys())

# %%
# The data is of shape (n_trajectories, n_timesteps, x, y)
# Get the first t0_field and save it as a numpy array
with h5py.File(p, "r") as f:
    A = f["t0_fields"]["A"][:]  # HDF5 datasets can be sliced like a numpy array
    print("shape of the selected t0_field: ", A.shape)
# %%
traj = 159  # select the trajectory
traj_toplot = A[traj]
# field is now of shape (n_timesteps, x, y).
# Let's do a subplot to plot it at t= 0, t= T/3, t= 2T/3 and t= T:
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
T = traj_toplot.shape[0]

# same colorbar for all subplots:
normalize_plots = False
cmap = "viridis"

if normalize_plots:
    vmin = np.nanmin(traj_toplot)
    vmax = np.nanmax(traj_toplot)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i, t in enumerate([0, T // 3, (2 * T) // 3, T - 1]):
        axs[i].imshow(traj_toplot[t], cmap=cmap, norm=norm)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(f"t={t}")
else:
    for i, t in enumerate([0, T // 3, (2 * T) // 3, T - 1]):
        axs[i].imshow(traj_toplot[t], cmap=cmap)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(f"t={t}")
plt.tight_layout()
# %%
