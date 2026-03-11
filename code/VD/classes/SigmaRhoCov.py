#this is diagcov (1).py written and sent by Francesco Paolo Casale by email on 29/11/2019, renamed to diagcov.py

import pdb
import scipy.special as ssp
import scipy as sp
import numpy as np
import scipy.linalg as la

from limix_core.util import assert_make_float_array
from limix_core.util import assert_finite_array
from limix_core.covar.covar_base import Covariance
from limix_core.hcache import cached


class SigmaRhoCov(Covariance):
    """
    SigmaRhoCov covariance matrix.
    A SigmaRhoCov covariance matrix has 2 parameter2
    """
    def __init__(self, K0=None):
        """
        Args:
            dim:        dimension 
            K0:         matrix with 1s where animals are in the same cage (including on the diagonal) and 0 elsewhere
        """
        Covariance.__init__(self)
        assert K0.shape[0]==K0.shape[1], 'Input covariance matrix to SigmaRhoCov should be square'
        self.dim = K0.shape[0]
        self._log_var = 0
        self._atan_rho = 0.5
        self.K0 = K0
        self._calcNumberParams()

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return sp.exp(self._log_var)

    @property
    def rho(self):
        return sp.tanh(self._atan_rho)

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self, value):
        assert value >= 0, 'Scale must be >= 0.'
        self._log_var = sp.log(value)
        self.clear_all()

    @rho.setter
    def rho(self, value):
        assert value >= 0, 'Scale must be >= 0.'
        self._atan_rho = ssp.arctanh(value)
        self.clear_all()

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        self._log_var = params[0]
        self._atan_rho = params[1]
        self.clear_all()

    def _calcNumberParams(self):
        self.n_params = 2

    def getParams(self):
        return sp.array([self._log_var, self._atan_rho])

    def getNumberParams(self):
        """
        return the number of hyperparameters
        """
        return self.n_params 

    #####################
    # Cached
    #####################
    @cached(['covar_base', 'K0'])
    def K(self):
        RV = self.rho * sp.ones([self.dim, self.dim])
        sp.fill_diagonal(RV, 1)
        RV = self.scale * RV
        return RV * self.K0 

    @cached(['covar_base', 'K0'])
    def K_grad_i(self,i):
        if i==0:
            return self.K() 
        else:
            RV = sp.ones([self.dim, self.dim]) / sp.cosh(self._atan_rho)**2
            sp.fill_diagonal(RV, 0)
            RV = self.scale * RV * self.K0
            return RV


if __name__=='__main__':

    pdb.set_trace()

    N = 6 # 5 rats
    C = 2 # 2 cages
    Cage = 1. * (sp.random.randint(0, C, (N, 1))==sp.arange(C)) # assigns rats to cages
    CageCov = sp.dot(Cage, Cage.T) # build cage covariance
    print('CageCov (look it has 1s on diagonal):')
    print(CageCov)

    C = SigmaRhoCov(N, CageCov)
    print('SigmaRhoCov:')
    print(C.K())

    pdb.set_trace()
