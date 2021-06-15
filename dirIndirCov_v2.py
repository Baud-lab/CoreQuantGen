import numpy as np
import scipy as sp
import scipy.linalg as LA
from limix_core.covar import Covariance
from limix_core.covar import FreeFormCov
from limix_core.hcache import cached
import pdb

import logging as LG

class DirIndirCov(Covariance):
    """
    Covariance matrix for decomposing direct and indirect genetic effects
    """
    def __init__(self, kinship, design, kinship_all = None, kinship_cross = None, C=None, jitter = 1e-4):
        if kinship_all is None:      kinship_all = kinship
        if kinship_cross is None:   kinship_cross = kinship
        ff_dim = 2
        if C is None:   self.C = FreeFormCov(ff_dim, jitter = 1e-4)
        else:               self.C = C
        self._K = kinship
        self._ZK = sp.dot(design, kinship_cross.T)
        self._KZ = sp.dot(kinship_cross, design.T)
        self._ZKZ = sp.dot(design, sp.dot(kinship_all, design.T))
        Covariance.__init__(self, kinship.shape[0])

    def dirIndirCov_K(self):
        return self.C.K()

    def dirIndirCov_K_ste(self):
        return self.C.K_ste()

    #####################
    # Properties
    #####################
    @property
    def variance(self):
        return self.C.variance

    @property
    def correlation(self):
        return self.C.correlation

    #####################
    # Params handling
    #####################
    def getParams(self):
        return self.C.getParams()

    def setParams(self,params):
        self.C.setParams(params)
        self.clear_all()

    def getNumberParams(self):
        return self.C.getNumberParams()

    def setCovariance(self,cov):
        """ set hyperparameters from given covariance """
        return self.C.setCovariance(cov)

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        C = self.C.K()
        RV  = C[0,0] * self._K
        RV += C[0,1] * (self._KZ + self._ZK)
        RV += C[1,1] * self._ZKZ
        return RV

    @cached('covar_base')
    def K_grad_i(self,i):
        Cgrad = self.C.K_grad_i(i)
        RV  = Cgrad[0,0] * self._K
        RV += Cgrad[0,1] * (self._KZ + self._ZK)
        RV += Cgrad[1,1] * self._ZKZ
        return RV

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return self.C.getInterParams()

    def K_grad_interParam_i(self,i):
        Cgrad = self.C.K_grad_interParam_i(i)
        RV  = Cgrad[0,0] * self._K
        RV += Cgrad[0,1] * (self._KZ + self._ZK)
        RV += Cgrad[1,1] * self._ZKZ
        return RV

    def setFIinv(self, value):
        self.C.setFIinv(value)

    def getFIinv(self):
        return self.C.getFIinv()

if __name__ == '__main__':
    # generate data
    n = 100
    f = 10
    X  = 1.*(sp.rand(n,f)<0.2)
    X -= X.mean(0); X /= X.std(0)
    kinship  = sp.dot(X,X.T)
    kinship /= kinship.diagonal().mean()
    design = sp.zeros((n,n))
    for i in range(n/2):
        design[2*i,2*i+1] = 1
        design[2*i+1,2*i] = 1

    # test covariance
    cov = DirIndirCov(kinship,design)
    cov.setRandomParams()
    print((cov.K()))
    print((cov.K_grad_i(0)))
