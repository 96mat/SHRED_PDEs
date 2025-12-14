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
Q = FunctionSpace(mesh, 'P', ord-1) ## By DeRham complex DG=O(RT)-1
W=MixedFunctionSpace([Udiv, Udiv, Q, Ucurl, Ucurl, Ucurl])

UH2=VectorFunctionSpace(mesh,'P',ord)
Bned=FunctionSpace(mesh,'RT',1)
UH2_FS=FunctionSpace(mesh,'P',ord)
W=MixedFunctionSpace([UH2, Bned, Q, UH2_FS, UH2_FS, UH2_FS])
# %%
x=SpatialCoordinate(mesh)

# IC: as_vector -> func(W)
u_ic=ic_velocity_regular(x[0],x[1])
B_ic_onZ=ic_magnetic_Lshape(x[0],x[1])
B_ic_Lid=as_vector((0,1))
# IC 2
u_ic_2 = ic_velocity_corona(x[0],x[1],0.5,0.5,0.2,0.06)

# BC
bc_v_L=DirichletBC(W.sub(0),Constant((0,0)),(1,2,3))
bc_v_L1=DirichletBC(W.sub(0),Constant((1,0)),4)
bc_B_L=DirichletBC(W.sub(1),Constant((0,0)),(1,2,3))
bc_w = DirichletBC(W.sub(3), Constant(0.0), (1,2,3))
bc_J = DirichletBC(W.sub(4), Constant(0.0), (1,2,3))
bc_E = DirichletBC(W.sub(5), Constant(0.0), (1,2,3))


bcsB=(bc_B_L)
bcs=(bc_v_L,bc_v_L1,bc_B_L,bc_w,bc_J,bc_E)

""" """
dt=0.01; g=1e+1; eta=5e-6; R_h=1e-6; nu=1e-2; S=10
n=FacetNormal(mesh)


# %%
def weak_form_NSB(u,B,p,w,J,E,v,C,q,z,K,F,dt,u_old,B_old,g):
  """
  v,C,q        = TestFunctions
  u,B,p        = TrialFunctions
  u_old, B_old = Functions
  """


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
    +nu*inner(grad(u),grad(v))*dx\
    +g*div(u)*div(v)*dx\
    -S*inner(j_vec, v)*dx\
    -p*div(v)*dx\
    +1/dt*dot(B, C)*dx\
    +dot(curl(E), C)*dx\
    +1/100*div(B)*div(C)*dx\
    +q*div(u)*dx\
    +inner(w, z)*dx\
    -inner(u, curl(z))*dx\
    +inner(J, K)*dx\
    -inner(B, curl(K))*dx\
    +inner(E, F)*dx\
    +(u[0]*B_old[1] - u[1]*B_old[0])* F*dx\
    -1/100*(J*F)*dx\



  L=1/dt*dot(u_old,v)*dx\
    +1/dt*dot(B_old,C)*dx\
    
   
  
  return a,L

"""


"""


"""
+g*rot(u)*rot(v)*dx\     #for velocity diffusion
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
#check_dimensions(cross(J,F), 1)
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
T=dt*5
t_vec = np.arange(0, T+0.1*dt, dt)  # T+0.1*dt to include also T: range/arange exclude the upper bound of the range

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
# Continuation method/time-stepping if necessary

param_newton = ( {'snes_type':'newtonls', # nonlinear solver types: https://petsc.org/release/manual/snes/#the-nonlinear-solvers
           'snes_rtol':1e-6, 'snes_atol':5e-4,
           "snes_linesearch_type": "basic",
           "snes_linesearch_maxstep": 1.0,
           'snes_stol':1e-12, 'snes_maxit':1000,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-6,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            "ksp_max_it": 50,
            "fieldsplit_0_ksp_max_it": 50,
            "fieldsplit_1_ksp_max_it": 40,

            # === blocco (u,p,w): fluido + ausiliari
            "pc_fieldsplit_0_fields": "0,2,3",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "lu",
            "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_0_pc_factor_mat_ordering_type": "nd", # evitare fill-in di LU
            # === blocco magnetico (B,J,E)
            "pc_fieldsplit_1_fields": "1,4,5",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "lu",
            "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_1_pc_factor_mat_ordering_type": "nd",  # evitare fill-in di LU       
            # logs essenziali
            "snes_monitor": "",
            "snes_linesearch_monitor": "",
            "snes_converged_reason": "",
            "ksp_monitor": "",
            "ksp_monitor_true_residual": "",
            "ksp_converged_reason": "",

          })


# %%

def newton_FNL(u,B,p,w,J,E,v,C,q,z,K,F,dt,u_old,B_old,g):
  """
  v,C,q        = TestFunctions
  u,B,p        = TrialFunctions
  u_old, B_old = Functions
  """


  stab_weight=Constant(3e-3)
  Bs=BurmanStab(u,v,u_old,stab_weight,mesh) # add Burman Stabilization
  j = as_matrix([[0, -1],
               [1,  0]])
  
  u_perp = dot(j,u) 
  w_vec = w*u_perp

  B_perp= dot(j , B)
  j_vec=J*B_perp

  Re_m=Re_magnetic(uh,eta,1) # varing Re_m during simulation

  F= +1/dt*inner(u,v)*dx\
    +inner(w_vec,v)*dx\
    +nu*inner(grad(u),grad(v))*dx\
    +g*div(u)*div(v)*dx\
    -S*inner(j_vec, v)*dx\
    -p*div(v)*dx\
    +1/dt*dot(B, C)*dx\
    +dot(curl(E), C)*dx\
    +1/100*div(B)*div(C)*dx\
    +q*div(u)*dx\
    +inner(w, z)*dx\
    -inner(u, curl(z))*dx\
    +inner(J, K)*dx\
    -inner(B, curl(K))*dx\
    +inner(E, F)*dx\
    +(u[0]*B[1] - u[1]*B[0])* F*dx\
    -1/100*(J*F)*dx\
    -1/dt*dot(u_old,v)*dx\
    -1/dt*dot(B_old,C)*dx

  return F
# %%

Wh_1 = Function(W)
u_1, B_1, p_1, w_1, J_1, E_1=split(Wh_1)
v_1, C_1, q_1, z_1, K_1, F_1=TestFunctions(W)


u_1old=Function(W[0])
B_1old=Function(W[1])
p_1old=Function(W[2])

Wh_1.sub(0).interpolate(uh)
Wh_1.sub(1).interpolate(Bh)
Wh_1.sub(2).interpolate(ph)
Wh_1.sub(3).interpolate(wh)
Wh_1.sub(4).interpolate(Jh)
Wh_1.sub(5).interpolate(Eh)

"""
#Check split interpolation: ok!
fig, ax = plt.subplots()
col = quiver(u_1, axes=ax)
plt.colorbar(col)
plt.title('velocity')"""

u_1old.assign(Wh_1.sub(0)) # fondamentale: from Picard
B_1old.assign(Wh_1.sub(1))
p_1old.assign(Wh_1.sub(2))

dwh = TrialFunction(W)
F = newton_FNL(u_1, B_1, p_1, w_1, J_1, E_1, v_1, C_1, q_1, z_1, K_1, F_1, dt, u_1old, B_1old, g)
J_cob = derivative(F,Wh_1,dwh)
A=NonlinearVariationalProblem(F,Wh_1,bcs=bcs,J=J_cob)
solver_snes=NonlinearVariationalSolver(A,solver_parameters=param_newton)


"""
# Check Newton-solver: ok!
solver_snes.solve()
u_1, B_1, p_1, w_1, J_1, E_1=split(Wh_1)"""
# %%
"""fig, ax = plt.subplots()
col = quiver(u_1, axes=ax)
plt.colorbar(col)
plt.title('velocity')"""
#%%

basename = 'MHDB_1'
outfileU_1 = File("MHD-Balmaz/NonLinear/velocity.pvd")
outfileP_1 = File("MHD-Balmaz/NonLinear/Pressure.pvd")
outfileB_1 = File("MHD-Balmaz/NonLinear/Magnetic.pvd")

u_1old.rename("Velocity")   # this names will be used in Paraview
p_1old.rename("Pressure")
B_1old.rename("Magnetic")
outfileU_1.write(u_1old)
outfileP_1.write(p_1old)
outfileB_1.write(B_1old)

# %%
# Time loop: Newton

T=5
t_vec = np.arange(0, T+0.1*dt, dt)  # T+0.1*dt to include also T: range/arange exclude the upper bound of the range

for ii in range(1, len(t_vec)):     # start from 1 to skip t=0
    t = t_vec[ii]

    if mesh.comm.rank==0:
      print('Time = ', t)

    solver_snes.solve()
    u_1, B_1, p_1, w_1, J_1, E_1=Wh_1.subfunctions

    #--------------------------------------------
    #sottraggo la media, la pressione deve stare su uno spazio a media nulla
    p_mean=p_mean_(p_1,mesh)
    ph_mean.assign(p_mean)
    deltap=p_1-ph_mean
    p_1.assign(deltap)
    #print(type(ph))
    #--------------------------------------------
    u_L2_ratio=errornorm(u_1,u_1old,'L2')/norm(u_1old,'L2')
    p_L2_ratio=errornorm(p_1,p_1old,'L2')/norm(p_1old,'L2')
    B_L2_ratio=errornorm(B_1,B_1old,'L2')/norm(B_1old,'L2')

    if mesh.comm.rank==0:
      print('||u||_L2',u_L2_ratio,'||p||_L2',p_L2_ratio,'||B||_L2',B_L2_ratio)

    #--------------------------------------------
    #RT2 space to DG2 space
    uh_DG.project(u_1old)
    Bh_DG.project(B_1old)
    
    #------
    norm_divB=divB_check(B_1,mesh)
    norm_divU=divU_check(u_1)
    Pe_fluid=CFL_Pe_eval(mesh,u_1,dt,g)[0]
    Pe_mag=CFL_Pe_eval_mag(mesh,B_1,dt,eta)[0]
    CFL_fluid=CFL_Pe_eval(mesh,u_1,dt,g)[1]
    CFL_mag=CFL_Pe_eval_mag(mesh,B_1,dt,eta)[1]
    Re_mag=Re_magnetic(u_1,eta,1)
    

    if mesh.comm.rank==0:
      print('div(B)_MSE',norm_divB,flush=True)
      print(f'divU={norm_divU}',flush=True)
      print(f'Pe_Fluid = {Pe_fluid}, CFL_Fluid = {CFL_fluid}',flush=True)
      print(f'Pe_Mag = {Pe_mag}, CFL_Mag = {CFL_mag}',flush=True)
      print(f'Re_m = {Re_mag}',flush=True)
    #------

    u_1.rename("Velocity")
    p_1.rename("Pressure")
    B_1.rename("Magnetic")

    # Post-processing
    outfileU_1.write(u_1old, time=t)
    outfileP_1.write(p_1old, time=t)
    outfileB_1.write(B_1old, time=t)
    
    
    u_1old.assign(u_1)
    B_1old.assign(B_1)
    p_1old.assign(p_1)
# %%

fig, ax = plt.subplots()
col = quiver(u_1, axes=ax)
plt.colorbar(col)
plt.title('velocity')
# %%
