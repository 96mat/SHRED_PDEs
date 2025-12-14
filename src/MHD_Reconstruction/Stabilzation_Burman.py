from firedrake import *

def BurmanStab(B,C,wind,stab_weight,mesh):
    n=FacetNormal(mesh)
    h=FacetArea(mesh)
    beta=avg(facet_avg(sqrt(inner(wind,wind)+1e-10)))
    gamma1=stab_weight
    stabilization_form=0.5*gamma1*avg(h)**2*beta*dot(jump(grad(B),n),jump(grad(C),n))*dS
    return stabilization_form
