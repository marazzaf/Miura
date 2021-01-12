#coding: utf-8

from dolfin import *
from numpy import arctanh

def atanh(func):
    vec = func.vector().get_local()
    vec = arctanh(vec)
    return Function(func.function_space(), vec)
