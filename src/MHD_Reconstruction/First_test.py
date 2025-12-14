#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import firedrake
import numpy as np
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
from firedrake.output import VTKFile
from Diagnostic import *
from Initial_Conditions import *
from Stabilzation_Burman import *
#--------------------------------------------------------
# Mesh set-up
n=80
mesh=UnitSquareMesh(n, n)
fig, ax = plt.subplots()
triplot(mesh, axes=ax)
ax.legend(loc='upper left')


# vtk output for Paraview
basename = 'MHDB_'
outfileU = File("MHD-Balmaz/velocity.pvd")
outfileP = File("MHD-Balmaz/Pressure.pvd")
outfileB = File("MHD-Balmaz/Magnetic.pvd")

# %%
ord=2
Udiv = FunctionSpace(mesh, 'RT',ord)
Ucurl=FunctionSpace(mesh, 'CG',ord) # Same order of RT
Q = FunctionSpace(mesh, 'DG', ord-1) ## By DeRham complex DG=O(RT)-1
W=MixedFunctionSpace([Udiv, Udiv, Q, Ucurl, Ucurl, Ucurl])

UH2=FunctionSpace(mesh,'RT',ord)
Bned=FunctionSpace(mesh,'N1curl',1)
UH2_FS=FunctionSpace(mesh,'P',ord)
W=MixedFunctionSpace([UH2, Bned, Q, UH2_FS, UH2_FS, UH2_FS])
# %%
x=SpatialCoordinate(mesh)

dt=0.01; g=1e+1; eta=5e-6; R_h=5e-1; nu=1; S=10; re_m=100
n=FacetNormal(mesh)

# IC: as_vector -> func(W)
u_ic=ic_velocity_regular(x[0],x[1])
B_ic_onZ=ic_magnetic_Lshape(x[0],x[1])

# IC 2
u_ic_2 = ic_velocity_corona(x[0],x[1],0.5,0.5,0.2,0.06)

#Hartmann
hart=Hartmann(nu,re_m,x[1],S)
u_prof=hart.u_profile()
B_prof=hart.B_profile()
u_in=(0.20,0)

# BC
g_lid=as_vector((1,0))
bc_v_L=DirichletBC(W.sub(0),Constant((0,0)),4)
bc_v_L1=DirichletBC(W.sub(0),Constant((0,0)),(1,2,3))
bc_B_L=DirichletBC(W.sub(1),Constant((0,0)),(1,2,3))
bc_w = DirichletBC(W.sub(3), Constant(0.0), (1,2,3))
bc_J = DirichletBC(W.sub(4), Constant(0.0), (1,2,3))
bc_E = DirichletBC(W.sub(5), Constant(0.0), (1,2,3))

bcs=(bc_B_L,bc_v_L,bc_v_L1,bc_w,bc_J,bc_E)


# Cell diameter used in DG formulation
h_p=avg(CellDiameter(mesh))
h_S = avg(CellVolume(mesh)/FacetArea(mesh)) #per integrale su dS (lati interni)
h=CellVolume(mesh)/FacetArea(mesh)          #per integrale su bordo ds(x)
# Penalty parameter used in DG formulation
sigma = Constant(0.1) #* W.sub(0).ufl_element().degree()**2
# tangential vector
tau_p=as_vector((-n("+")[1],n("+")[0]))
tau_m=as_vector((-n("-")[1],n("-")[0]))
tau=as_vector((-n[1],n[0]))


#Nullspace
nsp = MixedVectorSpaceBasis(
    W,
    [
        W.sub(0),                      # u: nessun contributo al nullspace
        W.sub(1),                      # B: nessun contributo
        VectorSpaceBasis(constant=True),  # p: costante globale
        W.sub(3),                      # w: nessun contributo
        W.sub(4),                      # J: nessun contributo
        W.sub(5),                      # E: nessun contributo
    ],
)

# Per problemi non simmetrici è prudente fornire anche il trasposto
tnsp = nsp
#print(assemble((sigma/h_S) * dS(domain=mesh)) / assemble(1.0 * dS(domain=mesh)))
#print(assemble((sigma/h)   * ds(domain=mesh)) / assemble(1.0 * ds(domain=mesh)))
# %%
def weak_form_NSB(u,B,p,w,J,E,v,C,q,z,K,F,dt,u_old,B_old,g):
  """
  v,C,q        = TestFunctions
  u,B,p        = TrialFunctions
  u_old, B_old = Functions
  """
  def grad_m(n,u,tau_p,tau_m):
    mean_pos=dot(n("+"),grad(dot(u("+"),tau_p)))
    mean_neg=dot(n("-"),grad(dot(u("-"),tau_m)))
    grad_m_i=0.5*(mean_pos+mean_neg)
    return grad_m_i                                  # {{eps(u)}}
  
  def jump_val(v,tau_p,tau_m):
    jump_val_i=(dot(v("+"),tau_p)+dot(v("-"),tau_m)) # [[v]]
    return jump_val_i

  grad_m_iu=grad_m(n,u,tau_p,tau_m) # {{eps(u)}}
  grad_m_iv=grad_m(n,v,tau_p,tau_m) # {{eps(v)}}
  jump_val_iu=jump_val(u,tau_p,tau_m) # [[u]]
  jump_val_iv=jump_val(v,tau_p,tau_m) # [[v]]

  stab_weight=Constant(3e-3)
  Bs=BurmanStab(u,v,u_old,stab_weight,mesh)
  j = as_matrix([[0, -1],
               [1,  0]])
  
  u_perp = dot(j,u_old) 
  w_vec = w*u_perp

  B_perp= dot(j , B_old)
  j_vec=J*B_perp

  Re_m=Re_magnetic(uh,eta,1)

  a= +1/dt*inner(u,v)*dx\
    +inner(w_vec,v)*dx\
    +g*div(u)*div(v)*dx\
    +nu*inner(grad(u),grad(v))*dx\
    -nu*grad_m_iu*jump_val_iv*dS\
    +nu*sigma/h_S*jump_val_iu*jump_val_iv*dS\
    +nu*grad_m_iv*jump_val_iu*dS\
    -S*inner(j_vec, v)*dx\
    -p*div(v)*dx\
    +1/dt*dot(B, C)*dx\
    +dot(curl(E), C)*dx\
    +g*div(B)*div(C)*dx\
    +1/re_m*div(B)*div(C)*dx\
    -R_h*dot(j_vec,C)*dx\
    -q*div(u)*dx\
    +inner(w, z)*dx\
    -inner(u, curl(z))*dx\
    +inner(J, K)*dx\
    -inner(B, curl(K))*dx\
    +inner(E, F)*dx\
    +(u[0]*B_old[1] - u[1]*B_old[0])* F*dx\
    -1/re_m*(J*F)*dx\

  
  a += nu * (
      - dot(grad(dot(u,tau)),n) * dot(v, tau) * ds\
      + (sigma/h) * dot(u, tau) * dot(v, tau) * ds\
      + dot(grad(dot(v,tau)),n) * dot(u, tau) * ds
      )  
    


  L = 1/dt*dot(u_old,v)*dx\
     +1/dt*dot(B_old,C)*dx\
     +dot(g_lid,tau)*(sigma/h*dot(v,tau)+dot(grad(dot(v,tau)),n))*ds(4,domain=mesh)\
     
  
  return a,L

"""
termine penality pressione
+1e-3*h_p/nu*jump(p)*jump(q)*dS\
"""
"""
a += nu * (
      - dot(grad(u,tau),n) * dot(v, tau) * ds(1,2,3,domain=mesh)                 # - ∂n(u·τ) (v·τ)
      + (sigma/h_S) * dot(u, tau) * dot(v, tau) * ds(1,2,3,domain=mesh)  # + α h^{-1} (u·τ)(v·τ)
      + - dot(grad(v,tau),n) * dot(u, tau) * ds(1,2,3,domain=mesh)                 # + ∂n(v·τ) (u·τ)
)
"""

"""
+nu*inner(grad(u),grad(v))*dx\              #diffusion DG
-inner(avg(nu*grad(u)),jump(outer(v, n)))*dS\
-inner(jump(nu*grad(u),n),avg(v))*dS\
"""

"""
-1/Re_m*(J*F)*dx\
"""

"""
+g*rot(u)*rot(v)*dx\     #for velocity diffusion DG
+g*div(u)*div(v)*dx\
+Rh*rot(B)*rot(C)*dx\     #for magnetic field diffusion
+Rh*div(B)*div(C)*dx\

 #hom's law
-1/Re_m*(J*F)*dx\
-1/Re_m*dot(B,curl(F))*dx\
-1/Re_m*dot(B,curl(F))*dx\#ohm's law lagrangian stabilizier
"""

"""
Hall's Term in components

-R_h*(-J*B_old[1])*F*dx\ 
-R_h*(J*B_old[0])*F*dx\
"""
# %%
#Check Dimension
j = as_matrix([[0, -1],
               [1,  0]])
#check_dimensions(cross(Jh,Bh), 1)
# %%
u,B,p,w,J,E=TrialFunctions(W)
v,C,q,z,K,F=TestFunctions(W)
# %%
Wh=Function(W)

uh,Bh,ph,wh,Jh,Eh=Wh.subfunctions

uh.interpolate(Constant((0,0)))
Bh.interpolate(u_ic_2)
ph.interpolate(Constant(0))

uh_old=Function(W[0])
Bh_old=Function(W[1])
ph_old=Function(W[2])
ph_mean=Function(W[2])

# For interpolation in DG
G=VectorFunctionSpace(mesh,'DG',2)
uh_DG=Function(G)
Bh_DG=Function(G)

# ----------------------------------
uh_old.assign(uh)
ph_old.assign(ph)
Bh_old.assign(Bh)

# %%
norm_divU=divU_check(uh)
norm_divB=divB_check(Bh,mesh)
if mesh.comm.rank==0:
  print('divB =',norm_divB,flush=True) #check divB=0
  print('divU =',norm_divU,flush=True)
# %%
# vtk output for Paraview
uh_old.rename("Velocity")   # this names will be used in Paraview
ph_old.rename("Pressure")
Bh_old.rename("Magnetic")
outfileU.write(uh_old)
outfileP.write(ph_old)
outfileB.write(Bh_old)
# %%
param = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}

param_gmres = {'ksp_type': 'gmres',
         'ksp_rtol': 1e-6,
         'ksp_max_it': 2000 ,
         "ksp_monitor": None
           }

# %%
a,L=weak_form_NSB(u,B,p,w,J,E,v,C,q,z,K,F,dt,uh_old,Bh_old,g)
problem=LinearVariationalProblem(a,L,Wh,bcs=bcs)
solver=LinearVariationalSolver(problem,solver_parameters=param_gmres)
solver.solve()
# %%
fig, ax = plt.subplots()
col = tripcolor(ph, axes=ax)
plt.colorbar(col)
plt.title('pressure')
fig, ax = plt.subplots()
col = quiver(uh, axes=ax)
plt.colorbar(col)
plt.title('velocity')
fig, ax = plt.subplots()
col = quiver(Bh, axes=ax)
plt.colorbar(col)
plt.title('Magnetic Field')
fig, ax = plt.subplots()
col = tripcolor(Eh, axes=ax)
plt.colorbar(col)
plt.title('Slak Electric Field')


# %%
T=2
t_vec = np.arange(0, T+0.1*dt, dt)  # T+0.1*dt to include also T: range/arange exclude the upper bound of the range
os.makedirs("MHD_values", exist_ok=True)
with CheckpointFile("MHD_values/files.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    for ii in range(1, len(t_vec)):     # start from 1 to skip t=0
        t = t_vec[ii]

        if mesh.comm.rank==0:
          print('Time = ', t)

        solver.solve()
        uh,Bh,ph,wh,Jh,Eh=Wh.subfunctions

        #--------------------------------------------
        #sottraggo la media, la pressione deve stare su uno spazio a media nulla

        p_mean=p_mean_(ph,mesh)
        ph_mean.assign(p_mean)
        deltap=ph-ph_mean
        ph=Function(W[2])
        ph.assign(deltap)

        #print(type(ph))
        #--------------------------------------------
        u_L2_ratio=errornorm(uh,uh_old,'L2')/norm(uh_old,'L2')
        p_L2_ratio=errornorm(ph,ph_old,'L2')/norm(ph_old,'L2')
        B_L2_ratio=errornorm(Bh,Bh_old,'L2')/norm(Bh_old,'L2')

        if mesh.comm.rank==0:
          print('||u||_L2',u_L2_ratio,'||p||_L2',p_L2_ratio,'||B||_L2',B_L2_ratio)

        #--------------------------------------------
        #RT2 space to DG2 space
        uh_DG.project(uh_old)
        Bh_DG.project(Bh_old)
        
        #------
        norm_divB=divB_check(Bh,mesh)
        norm_divU=divU_check(uh)
        Pe_fluid=CFL_Pe_eval(mesh,uh,dt,g)[0]
        Pe_mag=CFL_Pe_eval_mag(mesh,Bh,dt,eta)[0]
        CFL_fluid=CFL_Pe_eval(mesh,uh,dt,g)[1]
        CFL_mag=CFL_Pe_eval_mag(mesh,Bh,dt,eta)[1]
        Re_mag=Re_magnetic(uh,eta,1)
        

        if mesh.comm.rank==0:
          print('div(B)_MSE',norm_divB,flush=True)
          print(f'divU={norm_divU}',flush=True)
          print(f'Pe_Fluid = {Pe_fluid}, CFL_Fluid = {CFL_fluid}',flush=True)
          print(f'Pe_Mag = {Pe_mag}, CFL_Mag = {CFL_mag}',flush=True)
          print(f'Re_m = {Re_mag}',flush=True)
        #------

        uh.rename("Velocity")
        ph.rename("Pressure")
        Bh.rename("Magnetic")

        # Post-processing
        outfileU.write(uh_old, time=t)
        outfileP.write(ph_old, time=t)
        outfileB.write(Bh_old, time=t)

        #to h5
        afile.save_function(uh, name="u", idx=ii)
        afile.save_function(Bh, name="B", idx=ii)
        afile.save_function(ph, name="p", idx=ii)
        
        uh_old.assign(uh)
        Bh_old.assign(Bh)
        ph_old.assign(ph) # valutare se aggiornalo prima del campo magnetico
# %%
fig, ax = plt.subplots()
col = tripcolor(ph, axes=ax)
plt.colorbar(col)
plt.title('pressure')
fig, ax = plt.subplots()
col = quiver(uh, axes=ax)
plt.colorbar(col)
plt.title('velocity')
fig, ax = plt.subplots()
col = quiver(Bh, axes=ax)
plt.colorbar(col)
plt.title('Magnetic Field')
fig, ax = plt.subplots()
col = quiver(Bh_DG, axes=ax)
plt.colorbar(col)
plt.title('Magnetic Field_DG')
# %%

#%%