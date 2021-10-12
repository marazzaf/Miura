from dolfin import *

# Apply some scalar-to-scalar mapping `f` to each component of `T`:
def applyElementwise(f,T):
    from ufl import shape
    sh = shape(T)
    if(len(sh)==0):
        return f(T)
    fT = []
    for i in range(0,sh[0]):
        fT += [applyElementwise(f,T[i]),]
    return as_tensor(fT)

# Example where `f` takes the positive part of its argument:
def positive_part(T):
    return applyElementwise(lambda x : 0.5*(abs(x)+x) , T)

# Test:
T = as_tensor([[[1,2],[-3,4]],[[-5,6],[-7,8]]])
print(positive_part(T))
