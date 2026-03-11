import numpy as np
import scipy as sp
import scipy.linalg as LA

from limix_core.covar.covar_base import Covariance
from limix_core.hcache import cached

class LowRankCovMT(Covariance):
    """
    Free-form covariance via Cholesky, with a hard constraint:
        L[1,1] = 0   (0-based indexing; i.e. l22 in 1-based notation)

    Assumes dim >= 2.
    """

    def __init__(self, dim, jitter=1e-4, idxs=None):
        """
        Args:
            dim:    dimension of the covariance (assumed >= 2)
            jitter: diagonal offset added for numerical stability (default: 1e-4)
            idxs:   indexes to flip covariance so that correlation to constrain to 1 is in position [0,1] (default: None)
        """
        if dim < 2:
            raise ValueError("LowRankCovMT requires dim >= 2.")

        Covariance.__init__(self, dim, idxs)
        self._K_act = True
        self.dim = dim
        self.set_jitter(jitter)
        if idxs is None:
            self.perm = None
            self.invperm = None
        else:
            self.perm = np.asarray(idxs, dtype=int)
            if self.perm.shape[0] != self.dim:
                raise ValueError("Permutation has wrong length.")
        
            # inverse permutation: user -> internal
            self.invperm = np.empty(self.dim, dtype=int)
            self.invperm[self.perm] = np.arange(self.dim)        

        # All lower-triangular indices (including diagonal)
        self.idx_r_all, self.idx_c_all = sp.tril_indices(self.dim)

        # Position (in packed tril order) corresponding to (1,1)
        mask_l22 = (self.idx_r_all == 1) & (self.idx_c_all == 1)
        self._l22_pos = int(np.flatnonzero(mask_l22)[0])

        # Free parameter indices exclude (1,1)
        free_mask = np.ones(self.idx_r_all.shape[0], dtype=bool)
        free_mask[self._l22_pos] = False
        self.idx_r, self.idx_c = self.idx_r_all[free_mask], self.idx_c_all[free_mask]

        # Number of free parameters and storage
        self.n_params = int(self.idx_r.shape[0])
        self.params = sp.zeros(self.n_params)

    #####################
    # Properties
    #####################
    @property
    def variance(self):
        # K() is already in user order
        return self.K().diagonal()

    @property
    def correlation(self):
        R = self.K().copy() # K() is already in user order
        inv_diag = 1.0 / sp.sqrt(R.diagonal())[:, sp.newaxis]
        R *= inv_diag
        R *= inv_diag.T
        return R

    @property
    def X(self):
        L_int = self.L() #internal
        return self._to_user_rows(L_int) # permute rows only 

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
            raise ValueError("Trying to set a parameter via setParams that is not active.")
        if self._K_act:
            if len(params) != self.n_params:
                raise ValueError(f"Expected {self.n_params} params, got {len(params)}.")
            self.params[:] = params
            self.clear_all()

    def getParams(self):
        if not self._K_act:
            return np.array([])
        return self.params

    def getNumberParams(self):
        return int(self._K_act) * self.n_params

    def set_jitter(self, value):
        self.jitter = value
        self.clear_all()

    def setCovariance(self, cov):
        """
        Set hyperparameters from a given covariance by Cholesky factorization.
        Note: the resulting L[1,1] is forced to 0 regardless of cov.
        """
        cov_int = self._to_internal_order(cov) # internal covariance with indices so permutated
        chol = LA.cholesky(cov_int, lower=True)
        packed = chol[sp.tril_indices(self.dim)]
        packed = np.delete(packed, self._l22_pos)  # drop constrained entry
        self.setParams(packed)

    #####################
    # Cached core
    #####################
    @cached('covar_base')
    def K(self):
        L = self.L()
        K_int = sp.dot(L, L.T) + self.jitter * sp.eye(self.dim)
        return self._to_user_order(K_int)

    @cached('covar_base')
    def K_grad_i(self, i):
        if not self._K_act:
            raise ValueError("Trying to retrieve gradient over an inactive parameter.")
        L = self.L()
        Li = self.Lgrad(i)
        K_grad_i_int = sp.dot(L, Li.T) + sp.dot(Li, L.T)
        return self._to_user_order(K_grad_i_int)

    @cached
    def K_hess_i_j(self, i, j):
        if not self._K_act:
            raise ValueError("Trying to retrieve hessian over an inactive parameter.")
        Li = self.Lgrad(i)
        Lj = self.Lgrad(j)
        RV = sp.dot(Li, Lj.T)
        RV += RV.T
        return self._to_user_order(RV)

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        # K() is in user order already
        return self.K()[sp.tril_indices(self.dim)]

    def K_grad_interParam_i(self, i):
        ix, iy = sp.tril_indices(self.dim)
        ix = ix[i]
        iy = iy[i]
        R = sp.zeros((self.dim, self.dim))
        R[ix, iy] = R[iy, ix] = 1
        return R

    ######################
    # Private functions
    ######################
    @cached('covar_base')
    def L(self):
        R = sp.zeros((self.dim, self.dim))
        R[(self.idx_r, self.idx_c)] = self.params
        R[1, 1] = 0.0  # enforce l22 = 0
        return R

    @cached
    def Lgrad(self, i):
        R = sp.zeros((self.dim, self.dim))
        R[self.idx_r[i], self.idx_c[i]] = 1
        return R

    def Xgrad(self, i):
        Lgrad_i_int = self.Lgrad(i) # internal
        return self._to_user_rows(Lgrad_i_int) # permute rows only 
      
    ############################
    # Permutations methods 
    ############################
    def _to_user_order(self, M_int):
        if self.perm is None:
            return M_int
        M_int = np.asarray(M_int)
        if M_int.ndim != 2 or M_int.shape[0] != M_int.shape[1]:
            raise ValueError("_to_user_order expects a square 2D matrix.")
        return M_int[np.ix_(self.perm, self.perm)]

    def _to_internal_order(self, M_user):
        if self.invperm is None:
            return M_user
        M_user = np.asarray(M_user)
        if M_user.ndim != 2 or M_user.shape[0] != M_user.shape[1]:
            raise ValueError("_to_internal_order expects a square 2D matrix.")
        return M_user[np.ix_(self.invperm, self.invperm)]
    
    def _to_user_rows(self, A_int):
        """internal -> user for factors: permute rows only (P @ A)."""
        if self.perm is None:
            return A_int
        A_int = np.asarray(A_int)
        if A_int.ndim != 2:
            raise ValueError("_to_user_rows expects a 2D matrix.")
        return A_int[self.perm, :]
    


if __name__ == "__main__":
    dim = 3
    cov = LowRankCovMT(dim)
    print("n_params:", cov.getNumberParams())
    print("L:\n", cov.L())
    print("K:\n", cov.K())
    print("corr:\n", cov.correlation)
