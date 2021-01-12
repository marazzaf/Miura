from fenics import *
from fenics_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

from atanh import atanh


backend_atanh = atanh


class AtanhBlock(Block):
    def __init__(self, func, **kwargs):
        super(AtanhBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'AtanhBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        x = inputs[idx].vector()
        return adj_input / (1 - x*x)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_atanh(inputs[0])


atanh = overload_function(atanh, AtanhBlock)
