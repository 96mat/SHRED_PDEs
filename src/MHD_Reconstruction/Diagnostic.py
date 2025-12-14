from firedrake import *

#Evaluation of divB
def divB_check(Bh,mesh):
  Q0 = FunctionSpace(mesh, "DG", 0)
  #divB = project(div(Bh), Q0)
  val=sqrt(assemble(inner(div(Bh), div(Bh))*dx))
  #val=project(divB,Q0)
  return val

def divU_check(uh):
   val=sqrt(assemble(div(uh)*div(uh)*dx))
   return val


#Evaluation of CFL & Pe
def CFL_Pe_eval(mesh,uh,dt,nu):
    h = mesh.cell_sizes.dat.data_ro.min()
    u_max = np.max(np.abs(uh.dat.data_ro))   # vale anche per campi vettoriali (perché prende il max su tutte le componenti)
    Pe  = u_max*h/(2*nu)
    CFL = u_max*dt/h
    arr=np.array([Pe,CFL])
    return arr

def CFL_Pe_eval_mag(mesh,Bh,dt,eta):
   h=mesh.cell_sizes.dat.data_ro.min()
   B_max=np.max(np.abs(Bh.dat.data_ro))
   Pe= B_max*h/(2*eta)
   CFL=B_max*dt/h
   arr=np.array([Pe,CFL])
   return arr


#Evaluation of the mean pressure
def p_mean_(ph,mesh):
  area = assemble(Constant(1.0) * dx(domain=mesh))   # area del dominio (2D)
  p_mean = assemble(ph * dx(domain=mesh))
  return p_mean

# Check dimension for the WeakFormulation
def check_dimensions(x,y):
    j = as_matrix([[0, -1],
                   [1,  0]])
    
    try:
      expr=x*y
      try:
        print(expr.ufl_shape)
        print('transformation matrix j = as_matrix([[0, -1],[1,  0]])')
      except Exception:
        print('try the matrix-vector multiplication')
      return
    except Exception:
      pass
    
    try:
      expr=dot(x,y)
      try:
          print(expr.ufl_shape)
          print('transformation matrix j = as_matrix([[0, -1],[1,  0]])')
      except Exception:
          print('last chance is the Frobenius multiplication')
      return
    except Exception:
      pass

    try:
      expr=inner(x,y)
      try:
          print(expr.ufl_shape)
          print('transformation matrix j = as_matrix([[0, -1],[1,  0]])')
      except Exception:
          print('this is not a novel')
      return
    except Exception:
      print('check your weak formulation')


def Re_magnetic(u,eta,L):
   U=u.dat.data_ro.max()
   Re_m=U*L/eta
   return Re_m
    



   
    



   