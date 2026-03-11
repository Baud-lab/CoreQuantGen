#no need to change anything here

import numpy as np
import scipy as sp
import scipy.linalg as LA
#from .covar_base import Covariance
#from .freeform import FreeFormCov
from limix_core.covar import Covariance
from limix_core.covar import FreeFormCov
from limix_core.hcache import cached
from .partcorrCov import PartCorrCov
from .lowrankCov_mt import LowRankCovMT
import pdb

import logging as LG

class DirIndirCovMT(Covariance):
    """
    Covariance matrix for decomposing direct and social genetic effects
    """
    def __init__(self, kinship, design, kinship_all = None, kinship_cross = None, Iok=None, jitter = 1e-4, corr_zero=None, lowrank=False, idxs = None):
        ff_dim = 4
        if corr_zero is None:
            if lowrank == False:
                self.C = FreeFormCov(ff_dim, jitter = 1e-4)
            else:
                self.C = LowRankCovMT(ff_dim, jitter=1e-4, idxs=idxs)
        else:
            self.C = PartCorrCov(ff_dim, jitter = 1e-4, zero_corr_j=corr_zero[1], zero_corr_i=corr_zero[0])
        
        #pdb.set_trace()    
        if kinship_all is None:      kinship_all = kinship
        if kinship_cross is None:   kinship_cross = kinship
        self._K = kinship
        self._ZK = sp.dot(design, kinship_cross.T)
        self._KZ = sp.dot(kinship_cross, design.T)
        self._ZKZ = sp.dot(design, sp.dot(kinship_all, design.T))
        self.Iok = Iok
        dim = 2*kinship.shape[0]
        if Iok is not None:     dim = Iok.sum()
        Covariance.__init__(self, dim)
        #so Covariance is dim x dim matrix (corresponding to individuals with non missing phenotype value)

    def biDirIndirCov_K(self):
        return self.C.K()

    def biDirIndirCov_K_ste(self):
        return self.C.K_ste()

    def setBiDirIndirCov(self,cov):
        """ set hyperparameters from given covariance """
        return self.C.setCovariance(cov)

    #####################
    # Properties
    #####################
    @property
    def variance(self):
        return self.C.variance

    @property
    def correlation(self):
        return self.C.correlation

    @property
    def Cd(self): #Cd is a 2x2 D1 D2
        return self.C.K()[:2,:2] 

    @property
    def Cds(self): #Cds is 2x2 
        return self.C.K()[:2,2:] #1,2

    @property
    def Cs(self):
        return self.C.K()[2:,2:] 

#D1D2 S1S2

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

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        C = self.C.K()
        RV  = sp.kron(self.Cd, self._K)
        RV += sp.kron(self.Cds, self._KZ)
        RV += sp.kron(self.Cds.T, self._ZK)
        RV += sp.kron(self.Cs, self._ZKZ)
        #print 'dims of initial mat:', RV.shape
        if self.Iok is not None:
            RV = RV[self.Iok,:][:,self.Iok]
        return RV

    @cached('covar_base')
    def K_grad_i(self,i):
        Cgrad = self.C.K_grad_i(i)
        RV  = sp.kron(Cgrad[:2,:2], self._K)
        RV += sp.kron(Cgrad[:2,2:], self._KZ)
        RV += sp.kron(Cgrad[:2,2:].T, self._ZK)
        RV += sp.kron(Cgrad[2:,2:], self._ZKZ)
        if self.Iok is not None:
            RV = RV[self.Iok,:][:,self.Iok]
        return RV

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return self.C.getInterParams()

    def K_grad_interParam_i(self,i):
        Cgrad = self.C.K_grad_interParam_i(i)
        RV  = sp.kron(Cgrad[:2,:2], self._K)
        RV += sp.kron(Cgrad[:2,2:], self._KZ)
        RV += sp.kron(Cgrad[:2,2:].T, self._ZK)
        RV += sp.kron(Cgrad[2:,2:], self._ZKZ)
        if self.Iok is not None:
            RV = RV[self.Iok,:][:,self.Iok]
        return RV

    def setFIinv(self, value):
        self.C.setFIinv(value)

    def getFIinv(self):
        return self.C.getFIinv()

if __name__ == '__main__':
    # generate data
    import pdb
    n = 100
    f = 10
    X  = 1.*(sp.rand(n,f)<0.2)
    X -= X.mean(0); X /= X.std(0)
    kinship  = sp.dot(X,X.T)
    kinship /= kinship.diagonal().mean()
    kinship += 1e-4*sp.eye(n)
    design = sp.zeros((n,n))
    for i in range(n/2):
        design[2*i,2*i+1] = 1
        design[2*i+1,2*i] = 1
    pdb.set_trace()

    # test covariance
    cov = DirIndirCovMT(kinship,design)
    cov.setRandomParams()
    print((cov.K()))
    print((LA.eigh(cov.K())[0]).min())
    #print((cov.K_grad_i(0)))

