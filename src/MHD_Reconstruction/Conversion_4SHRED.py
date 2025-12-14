#%%
import numpy as np
import h5py

def walk(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}[GROUP] /{name}")
    else:
        print(f"{indent}[DATASET] /{name}  shape={obj.shape} dtype={obj.dtype}")


with h5py.File("MHD_values/files.h5", "r") as f:
    print("[GROUP] /")
    f.visititems(walk)
#%%
from firedrake import CheckpointFile, FunctionSpace

Nt = 200

with CheckpointFile("MHD_values/files.h5", "r") as f:
    mesh = f.load_mesh()

    Vu = FunctionSpace(mesh, "RT", 2)
    VB = FunctionSpace(mesh, "N1curl", 1)
    Vp = FunctionSpace(mesh, "DG", 1)

    n_u = Vu.dim()
    print(n_u)
    n_B = VB.dim()
    n_p = Vp.dim()

    X = np.zeros((Nt, n_u + n_B + n_p), dtype=np.float32)

    for n in range(Nt):
        u = f.load_function(mesh, name="u", idx=n)
        B = f.load_function(mesh, name="B", idx=n)
        p = f.load_function(mesh, name="p", idx=n)

        X[n, :n_u] = u.dat.data_ro
        X[n, n_u:n_u+n_B] = B.dat.data_ro
        X[n, n_u+n_B:] = p.dat.data_ro  # nt x u x B x p

np.savez("MHD_values/shred_dataset.npz", X=X, n_u=n_u, n_B=n_B, n_p=n_p)
# %%