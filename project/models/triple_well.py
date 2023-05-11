import numpy as np
import torch

from .system import System

#################################################################################

class TripleWell(System):
    """
    Triple well potential as used by
    """
    def __init__(self, params = None, **kwargs):
        params_default = {
                "a" : 1.0,
                "b" : 6.0,
                "c" : 1.0,
                "d" : 1.0
            }

        # Set Parameters
        if (params != None):
            self.params = params
        else:
            self.params = params_default

    def energy(self,x):

        if (len(x.shape) > 1):
            return  3 * torch.exp( -x[:,0]**2 - (x[:,1] - (1/3))**2) - \
                        3 * torch.exp( -x[:,0]**2 - (x[:,1] - (5/3))**2) - \
                            5 * torch.exp( - (x[:,0] - 1)**2 - x[:,1]**2) - \
                                5 * torch.exp( - (x[:,0] + 1)**2 - x[:,1]**2) + \
                                    0.2*x[:,0] + 0.2*(x[:,1] - (1/3)**4)
        else:
            return  3 * torch.exp( -x[0]**2 - (x[1] - (1/3))**2) - \
                        3 * torch.exp( -x[0]**2 - (x[1] - (5/3))**2) - \
                            5 * torch.exp( - (x[0] - 1)**2 - x[1]**2) - \
                                5 * torch.exp( - (x[0] + 1)**2 - x[1]**2) + \
                                    0.2*x[0] + 0.2*(x[1] - (1/3)**4)
    # def energy(self, x):
    #     a = self.params['a']
    #     b = self.params['b']
    #     c = self.params['c']
    #     d = self.params['d']
    #     if (len(x.shape) > 1):
    #        return a*x[:,0]**4 - b*x[:,0]**2 + c*x[:,0] + d*x[:,1]**2/2
    #     else:
    #         return a*x[0]**4 - b*x[0]**2 + c*x[0] + d*x[1]**2/2
