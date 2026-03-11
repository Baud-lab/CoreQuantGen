import numpy as np
import scipy as sp
import scipy.linalg as LA
from limix_core.covar.covar_base import Covariance
from limix_core.hcache import cached
import pdb

import logging as LG

class PartCorrCov(Covariance):
    """
    General semi-definite positive matrix with no contraints.
    A free-form covariance matrix of dimension d has 1/2 * d * (d + 1) params
    """
    def __init__(self, dim, jitter=1e-4, zero_corr_j=0, zero_corr_i=1):
        """
        Args:
            dim:        dimension of the free-form covariance
            jitter:     extent of diagonal offset which is added for numerical stability
                        (default value: 1e-4)
        """
        Covariance.__init__(self, dim)
        self._K_act = True
        self._calcNumberParams()
        self.dim = dim
        self.params = np.zeros(self.n_params)
        self.idx_r, self.idx_c = np.tril_indices(self.dim)
        self.set_jitter(jitter)

        assert zero_corr_j==0, 'Not supported' 
        R = np.zeros((self.dim, self.dim))
        R[(self.idx_r, self.idx_c)] = np.arange(self.getNumberParams()+1)
        self.param_index_zero = int(R[zero_corr_i, zero_corr_j])

    #####################
    # Properties
    #####################
    @property
    def variance(self):
        return self.K().diagonal()

    @property
    def correlation(self):
        R = self.K().copy()
        inv_diag = 1./np.sqrt(R.diagonal())[:,np.newaxis]
        R *= inv_diag
        R *= inv_diag.T
        return R

    @property
    def X(self):
        return self.L()

    #####################
    # Activation handling
    #####################
    @property
    def act_K(self):
        return self._K_act

    @act_K.setter
    def act_K(self, act):
        self._K_act = bool(act)
        self._notify()

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        if not self._K_act and len(params) > 0:
            raise ValueError("Trying to set a parameter via setParams that "
                             "is not active.")
        if self._K_act:
            self.params[:] = params
            self.clear_all()

    def getParams(self):
        if not self._K_act:
            return np.array([])
        return self.params

    def getNumberParams(self):
        return int(self._K_act) * self.n_params

    def _calcNumberParams(self):
        self.n_params = int(0.5*self.dim*(self.dim+1) - 1)

    def set_jitter(self,value):
        self.jitter = value

    def setCovariance(self,cov):
        """ set hyperparameters from given covariance """
        chol = LA.cholesky(cov,lower=True)
        params = chol[np.tril_indices(self.dim)]
        Ikeep = np.arange(params.shape[0])!=self.param_index_zero
        self.setParams(params[Ikeep])

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        RV = np.dot(self.L(),self.L().T)+self.jitter*np.eye(self.dim)
        return RV

    @cached('covar_base')
    def K_grad_i(self,i):
        if not self._K_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        RV = np.dot(self.L(),self.Lgrad(i).T)+np.dot(self.Lgrad(i),self.L(i).T)
        return RV

    @cached
    def K_hess_i_j(self, i, j):
        if not self._K_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        RV = np.dot(self.Lgrad(i),self.Lgrad(j).T)
        RV+= RV.T
        return RV

    def K_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = np.zeros((self.dim, self.dim))
            R[np.tril_indices(self.dim)] = np.sqrt(self.getFIinv().diagonal())
            # symmetrize
            R = R + R.T - np.diag(R.diagonal())
        return R

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        # VARIANCE + CORRELATIONS
        #R1 = self.variance
        #R2 = self.correlation[np.tril_indices(self.dim, k = -1)]
        #R = np.concatenate([R1,R2])

        # COVARIANCES
        R = self.K()[np.tril_indices(self.dim)]
        return R

    # DERIVARIVE WITH RESPECT TO COVARIANCES
    def K_grad_interParam_i(self, i):
        ix, iy = np.tril_indices(self.dim)
        ix = ix[i]
        iy = iy[i]
        R = np.zeros((self.dim,self.dim))
        R[ix, iy] = R[iy, ix] = 1
        return R

    # DERIVARIVE WITH RESPECT TO VARIANCES AND CORRELATIONS
    #def K_grad_interParam_i(self, i):
    #    if i < self.dim:
    #        # derivative with respect to the variance
    #        R = sp.zeros((self.dim,self.dim))
    #        R[i,:] = self.K()[i,:] / (2 * self.variance[i])
    #        R += R.T
    #    else:
    #        # derivarice with respect to a correlation
    #        ## 1. take the corresponding off diagonal element
    #        ix, iy = sp.tril_indices(self.dim, k = -1)
    #        ix = ix[i - self.dim]
    #        iy = iy[i - self.dim]
    #        ## 2. fill it with sqrt(var * var)
    #        R = sp.zeros((self.dim,self.dim))
    #        R[ix,iy] = R[iy,ix] = sp.sqrt(self.variance[ix] * self.variance[iy])
    #    return R

    ######################
    # Private functions
    ######################

    def extend_params(self, params):
        p1 = params[:self.param_index_zero]
        p2 = np.zeros(1)
        p3 = params[self.param_index_zero:]
        ext_params = np.concatenate([p1, p2, p3], axis=0)
        return ext_params


    @cached('covar_base')
    def L(self):
        R = np.zeros((self.dim, self.dim))
        R[(self.idx_r, self.idx_c)] = self.extend_params(self.getParams())
        return R

    @cached
    def Lgrad(self, i):
        params_grad = np.zeros(self.getNumberParams())
        params_grad[i] = 1
        R = np.zeros((self.dim, self.dim))
        R[(self.idx_r, self.idx_c)] = self.extend_params(params_grad)
        return R

    def Xgrad(self, i):
        return self.Lgrad(i)

if __name__ == '__main__':
    n = 2
    cov = FreeFormCov(n)
    print((cov.K()))
    print((cov.K_grad_i(0)))
