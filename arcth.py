#coding: utf-8

import math
from ufl.mathfunctions import MathFunction
from ufl.operators import _mathfunction
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import Zero, RealValue, FloatValue

@ufl_type()
class Atanh(MathFunction):
    __slots__ = ()

    def __new__(cls, argument):
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.atanh(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "atan", argument)

def atanh(f):
    "UFL operator: Take the inverse hyperbolic tangent of *f*."
    return _mathfunction(f, Atanh)
