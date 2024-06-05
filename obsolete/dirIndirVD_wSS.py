#IGE (Indirect genetic effects) = SGE (Social genetic effects). 
#IEE: Indirect environmental effects

#the potential presence of NAs in input phenotype, covs, cages etc means that in this code we introduce two sets of animals (not discussed in the paper):
#focal animals, defined as having phenotype, covs (if covs are provided), cage (if cages are provided, which is not necessary in the case where DGE but no IGE are modelled) and kinship
#all animals with cage and kinship and in subset_IDs; this set of animals is referred to using _all in this code. 
#!! there can be animals in _all that are not included in the analysis (e.g after NA in phenotypes are filtered out)
#Note that in the paper cage mate has a different meaning, namely "the other mice in the cage of the focal individual".

import sys
import warnings
import numpy as np
import scipy.linalg as la
#you need to install limix_core from https://github.com/limix/limix-core
from limix_core.mean.mean_base import MeanBase as lin_mean
#dirIndirCov_v2 allows to pass low rank 2x2 genetic matrix (DGE, IGE and their covariance).
from dirIndirCov_v2 import DirIndirCov
from diagcov import SigmaRhoCov
from limix_core.covar.fixed import FixedCov 
from limix_core.covar import LowRankCov 
from limix_core.covar import DiagonalCov 
from limix_core.covar.combinators import SumCov
from limix_core.util.preprocess import covar_rescaling_factor
from limix_core.util.preprocess import covar_rescale
from limix_core.gp.gp_base import GP
import pdb
import h5py
import copy
import math
import gc

class DirIndirVD():

    #many of inputs below are passed from a SocialData object built using social_data.py
    #pheno: missing values should be encoded as -999 (needs to be numeric for HDF5). pheno can be 1xN or Nx1 vector (where N is number of individuals)
    #pheno_ID: IDs corresponding to pheno.
    #covs: missing values should be encoded as -999.  covs can be 1xN or Nx1 vector, or a NxK matrix (K number of covariates). intercept column not expected. cage_density not necessary. ok if present as long as option independent_covs = True
    #covs_ID: IDs corresponding to covs
    #covariates_names: name of covariates in covs
    #kinship_all: GRM for all mice ever considered (focal or cagemate or nothing). no missing values allowed. symmetric matrix.
    #kinship_all_ID: rownames and colnames of kinship_all
    #cage_all: cages of all mice ever considered (focal or cagemate or nothing). missing values should be encoded as NA. NA cage will lead to this animal being ignored from the analysis. if you have reason to believe this animal was a cage mate (but you don't know its cage), this is an issue and you need to be aware of it.
    #cage_all_ID: IDs corresponding to cage_all
    #independent_covs: if True (default) LIMIX will check whether any covariate is a linear combination of the others and if so will fix the issue by updating covs to independent columns
    #DGE: should DGE be included in the model?
    #IGE: should IGE be included in the model?
    #IEE: should IEE be included in the model? note that if IGE = True, IEE will be set to True by the program (silently) - see below
    #cageEffect: should cage effects be included in the model?
    #calc_ste: should the standard errors of the variance components be estimated and output? 
    #standardize_pheno: should the phenotype be standardized to variance 1? better to do so (numerically more stable)
    #subset_IDs: subset of IDs to consider *as focal individuals or cage mate*. this means that any animal not in subset_IDs will be ignored completely from analysis. This will be an ok thing to do only if all animals from a cage are excluded! if instead want to exclude animals only as focal individuals, set their phenotype to NA.
    #subset_on_cage: defaults to False. Turn to True if cageEffect, IEE and IGE = False but still want to only consider those animals that have a non-NA cage
    #vc_init_type: characteristics of the 2x2 matrix that sigma_Ad^2, var_Ads, sigma_As^2. 
    #Can be lowrank and used in combination with providing vc_init, which would have been estimated in prior run
    #Can also be diagonal and used in combination with vc_init
    #Can also be random diagonal to initialise at corr_Ads = 0 but without providing vc_init
    #Can also be None in which case it will be a freeform 2x2 matrix (good for usual cases I think)
    #vc_init: defaults to None. Can be provided in a second run as an instance of DirIndirVD
    #std_genos = True: standardize genotypes. used to calculate variance explained by SNP
    #SimplifNonIdableEnvs: defaults to False. When all groups are of equal size, non-genetic variance components are not identifiable. Can (but don't have to) turn SimplifNonIdableEnvs to True to report rho instead of individual environmental VCs. No STE yet on rho I think. genetic estimates will be valid no matter what.

    def __init__(self, vc_init, vc_init_type, pheno = None, pheno_ID = None, covs = None, covs_ID = None, covariates_names = None, kinship_all = None, kinship_all_ID = None, cage_all = None, cage_all_ID = None, independent_covs = True, DGE = False, IGE = False, IEE = False, cageEffect = False, calc_ste = False, standardize_pheno = True, subset_IDs = None, subset_on_cage = False, std_genos = False, SimplifNonIdableEnvs = False):
 
        # the purpose of the IEE argument is to specify whether IEE should be off when IGE are off. When IGE are on, IEE must be on and therefore IEE will be automatically set to True when IGE is True.
        if IGE:
            IEE = True

        self.parseNmore(pheno, pheno_ID, covs, covs_ID, covariates_names, kinship_all, kinship_all_ID, cage_all, cage_all_ID, independent_covs,standardize_pheno, subset_IDs, cageEffect, IEE, subset_on_cage, std_genos)

        #define the genetic, environmental and cage covariance matrices
        self.VD(DGE = DGE, IGE = IGE, IEE = IEE, cageEffect = cageEffect, vc_init = vc_init, vc_init_type = vc_init_type, SimplifNonIdableEnvs = SimplifNonIdableEnvs)
        #optimize to estimates the variance components
        self.optimize(vc_init = vc_init,  vc_init_type = vc_init_type)
        # get_VCs gets the variance components, standard errors, etc. for output
        # self.output useful to retrieve output using getOutput without having to specify calc_ste, DGE, IGE, etc.
        self.output = self.get_VCs(calc_ste = calc_ste, DGE = DGE, IGE = IGE, IEE = IEE, cageEffect = cageEffect, SimplifNonIdableEnvs = SimplifNonIdableEnvs)

    def parseNmore(self, pheno, pheno_ID, covs, covs_ID, covariates_names, kinship_all, kinship_all_ID, cage_all, cage_all_ID, independent_covs,standardize_pheno, subset_IDs,cageEffect, IEE, subset_on_cage, std_genos):
        """match various inputs"""

        assert pheno is not None, 'Specify pheno!'
        assert pheno_ID is not None, 'Specify pheno IDs!'
        assert pheno.shape[0] == len(pheno_ID), 'Lengths of pheno and pheno IDs do not match!'
        assert kinship_all is not None, 'Specify kinship!'
        assert kinship_all_ID is not None, 'Specify kinship IDs!'
        assert kinship_all.shape[0] == kinship_all.shape[1], 'Kinship is not a square matrix!'
        assert kinship_all.shape[0] == len(kinship_all_ID), 'Dimension of kinship and length of kinship IDs do not match!'
        assert calc_ste and SimplifNonIdableEnvs, sys.exit("can't get STE with SimplifNonIdableEnvs")

####hack to shorten runtime or debug
#        uCage=np.unique(cage_all)
#        assert len(uCage) > 120, 'Too few'
#        nb_cages_to_keep = 3
#        nb_to_remove = (len(uCage) - nb_cages_to_keep)
#        remove=uCage[0:nb_to_remove]
#        idx_remove = np.concatenate([np.where(cage_all==remove[i])[0] for i in range(len(remove))])
#        cage_all[idx_remove]='NA'
####hack to shorten runtime

        #1. define set of animals in subset IDs, with kinship information and non-NA cage if cage requested (_all)
        #1.1 _all animals need to be in subset_IDs. !!!! this is only correct is subset_IDs constructed so that all animals in a cage are in or out. not correct to include only some animals in a cage.
        if subset_IDs is not None:
            Imatch = np.nonzero(subset_IDs[:,np.newaxis]==kinship_all_ID)
            kinship_all = kinship_all[Imatch[1],:][:,Imatch[1]]
            kinship_all_ID=kinship_all_ID[Imatch[1]]

        #1.2 NA allowed in cage information so first of all exclude missing cage data and corresponding animals
        if cageEffect or IEE or subset_on_cage:
            assert cage_all is not None, 'Specify cage!'
            assert cage_all.shape[0] == len(cage_all_ID), 'Lengths of cage_all and cage_all_ID do not match!'
            has_cage = (cage_all!='NA')
            if sum(has_cage)==0:
                cage_all = None
                assert cage_all is not None, 'All cages missing'
            cage_all=cage_all[has_cage]
            cage_all_ID=cage_all_ID[has_cage]

            #1.3 match cages and kinship.
            Imatch = np.nonzero(cage_all_ID[:,np.newaxis]==kinship_all_ID)
            cage_all_ID = cage_all_ID[Imatch[0]]
            cage_all = cage_all[Imatch[0]]
            kinship_all = kinship_all[Imatch[1],:][:,Imatch[1]]
            kinship_all_ID=kinship_all_ID[Imatch[1]]
            #(kinship_all_ID==cage_all_ID).all()
            #True
            #cage and kinship now have no missing values and are matched - IDs are in cage_all_ID and kinship_all_ID
            # put IDs in sampleID_all
        sampleID_all = kinship_all_ID
        assert len(sampleID_all)!=0, 'No _all animals'
        print('Number of mice in subset, with kinship information and with cage info if cage requested: '  + str(len(sampleID_all)))

        #2. define focal animals now: those in sampleID_all that have non missing phenotype and non missing covs
        #2.1 remove NAs from pheno
        if len(pheno.shape)==1:
            pheno = pheno[:,np.newaxis]
        has_pheno = (pheno!=(-999))[:,0]
        pheno=pheno[has_pheno,:]
        pheno_ID=pheno_ID[has_pheno]

        #2.2 add intercept to covs
        #if no covs are provided, make it a vector of 1s for intercept
        if covs is None:
            assert covariates_names is None, "covariates_names should be None as covs is None"
            covs = np.ones((pheno.shape[0],1))
            covs_ID = pheno_ID
            covariates_names = "mean"
        #if covs are provided, append a vector of 1s for intercept
        else:
            new_col=np.ones([covs.shape[0],1])
            if len(covs.shape)==1:
                covs = covs[:,np.newaxis]
            covs=np.append(new_col,covs,1)
            covariates_names = np.append("mean",covariates_names)
        
        #2.3 remove NAs from covs
        has_covs = (covs!=(-999)).all(1)
        covs=covs[has_covs,:]
        covs_ID=covs_ID[has_covs]
        
        #2.4 match pheno and covs
        Imatch = np.nonzero(pheno_ID[:,np.newaxis]==covs_ID)
        pheno = pheno[Imatch[0],:]
        pheno_ID=pheno_ID[Imatch[0]]
        covs = covs[Imatch[1],:]
        covs_ID=covs_ID[Imatch[1]]
        #(pheno_ID==covs_ID).all()
        #True
        #pheno and covs now have no missing values and are matched - IDs are in pheno_ID and covs_ID

        #2.5 check which of those are in sampleID_all (and thus are in subset_IDs, have kinship and cage if cage important)
        has_geno = np.array([pheno_ID[i] in sampleID_all for i in range(pheno_ID.shape[0])])
        pheno = pheno[has_geno,:]
        covs = covs[has_geno,:]
        #create sampleID that has focal individuals.
        sampleID=pheno_ID[has_geno]
        assert len(sampleID)!=0, 'No focal animals'
        #remember sampleID_all and sampleID are in different order (and of different length possibly)
        print('Number of mice with pheno and covs information: '  + str(len(sampleID)))

        #3. create cage and kinship for focal animals
        #3.1 create kinship for focal animals
        idxs = np.array([np.where(sampleID_all==sampleID[i])[0][0] for i in range(sampleID.shape[0])])
        kinship=kinship_all[idxs,:][:,idxs]
        
        #3.2 create cage for focal animals
        if cageEffect or IEE or subset_on_cage:
            cage=cage_all[idxs]

            #calculate real sample_size_all (i.e. of all the _all animals, which ones are actually in a cage with a focal animal?)
            is_real_all = np.array([cage_all[i] in cage for i in range(cage_all.shape[0])])
            self.real_sample_size_all = is_real_all.sum()        
            #equivalent to:
            idx_real_all = np.concatenate([np.where(cage_all==cage[i])[0] for i in range(len(cage))])
            self.real_sample_size_all2 = len(np.unique(idx_real_all))
            assert self.real_sample_size_all==self.real_sample_size_all2, 'Pb self.real_sample_size_all'

            #transpose cage if necessary
            if len(cage.shape)==1:
                cage = cage[:,np.newaxis]
           
            #create cage density covariate and adds it as a covariate. it's ok if it has already been given as input as a covariate as colinear covariates are pruned out (as long as independent_covs=True; if they're not and 100% colinear the code will crash; if not 100% colinear then it will "only" cost 1 df)
            cage_density = np.array([len(np.where(cage_all == cage[i])[0]) for i in range(cage.shape[0])])
            cage_density = cage_density[:,np.newaxis]
            assert cage_density.shape[0] == covs.shape[0], 'length cage and covs diff!'
            covs=np.append(covs,cage_density,1)
            covariates_names = np.append(covariates_names,"cage_density")
        else:
            cage_density = None
            self.real_sample_size_all = None

        #4. create focal x _all genetic cross-covariance
        kinship_cross = kinship_all[idxs,:]
        #so sampleID along rows and sampleID_all along colummns

        #5. now create environmental matrices
        env = np.eye(kinship.shape[0])
        env_all = np.eye(kinship_all.shape[0])
        env_cross = env_all[idxs,:]

        # standardize_pheno
        if standardize_pheno:
            pheno -= pheno.mean(0)
            pheno /= pheno.std(0)
            print('Pheno has been standardized')

        print("Covariates in dirIndirVD_sWW before making them independent: " + str(covariates_names))

        #make covariates independent
        if independent_covs and covs.shape[1]>1:
            tol = 1e-6
            R = la.qr(covs,mode='r')[0][:covs.shape[1],:]
            I = (abs(R.diagonal())>tol)
            if np.any(~I):
                print('Covariates  '+ str(covariates_names[(np.where(~I)[0])]) +' have been removed because linearly dependent on the others')
            covs = covs[:,I]
            covariates_names =  covariates_names[I]
            print('Final covariates are  ' + str(covariates_names))

        # standardize genotypes (e.g. to study var explained by a SNP - then put that SNP as a cov)
        if std_genos:
            covs[:,1:] -= np.delete(covs, 0, axis=1).mean(0)
            covs[:,1:] /= np.delete(covs, 0, axis=1).std(0)
            print('all covs have been standardized')
        else:
            is_not_direct_geno = [not "_direct" in covariates_name for covariates_name in covariates_names]
            is_not_social_geno = [not "_social" in covariates_name for covariates_name in covariates_names]
            is_not_geno = np.logical_and(is_not_direct_geno,is_not_social_geno)
            idxs = np.where(is_not_geno)[0]
            covs[:,idxs[1:]] -= covs[:,idxs[1:]].mean(0)
            covs[:,idxs[1:]] /= covs[:,idxs[1:]].std(0)
            print('covs except genos have been standardized')

        #for i in range(covs.shape[1]):
        #    print covariates_names[i]
        #    print covs[:,i].mean(0)
        #    print covs[:,i].std(0)
        # so mean is 1 sd 0 for mean, and mean is ~0 and var 1 for all other non-genotype covariates    

        self.sampleID=sampleID
        self.pheno=pheno
        self.covs=covs
        self.covariates_names = covariates_names
        if cageEffect or IEE or subset_on_cage:
            self.cage=cage
        self.kinship=kinship
        self.env=env
        self.sampleID_all=sampleID_all
        self.cage_all=cage_all
        self.kinship_all=kinship_all
        self.env_all=env_all
        self.kinship_cross=kinship_cross
        self.env_cross=env_cross
        self.cage_density = cage_density

    def VD(self, DGE, IGE, IEE, cageEffect,vc_init, vc_init_type, SimplifNonIdableEnvs):
        """ defines covariance for variance decomposition."""

        #defines mean
        mean = lin_mean(self.pheno,self.covs)

#### to check covariates input
#        check_outfile = h5py.File('/homes/abaud/HSmice_paper/output/check_covariates.h5', "w")
#        check_outfile.create_dataset(name="pheno", data=self.pheno)
#        check_outfile.create_dataset(name="sample_ID", data=self.sampleID)
#        check_outfile.create_dataset(name="covs", data=self.covs)
#        check_outfile.close()


        print("mean.n_covs is " + str(mean.n_covs))

        #define cagemate assignment for analyses involving IGE and/or IEE. Z is N focal x N_all and has 0s in cells Z_i,i (i.e. an animal is not its own cage mate)
        if IEE:
            #transform boolean to float. shape of same_cage is len(self.cage) x len(self.cage_all)
            same_cage = 1. * (self.cage==self.cage_all)
            diff_inds = 1. * (self.sampleID[:,np.newaxis]!=self.sampleID_all)
            Z = same_cage * diff_inds

        #define the overall genetic covariance matrix
        if DGE or IGE:
            #scales the DGE component of the sample to sample covariance matrix to have sample variance 1
            sf_K = covar_rescaling_factor(self.kinship)
            self.kinship *= sf_K

            #now create and scale the IGE component of the sample to sample covariance matrix to 1, and same for the DGE/IGE covariance component
            if IGE:
                #first IGE variance: ZKallZ' in this code (ZKZ' in paper)
                _ZKallZ = np.dot(Z,np.dot(self.kinship_all,Z.T))
                sf_ZKallZ = covar_rescaling_factor(_ZKallZ)
                self.kinship_all *= sf_ZKallZ
                 #then DGE/IGE covariance:
                self.kinship_cross *= np.sqrt(sf_K * sf_ZKallZ)
        
        if DGE and not IGE:
            self._genoCov = FixedCov(self.kinship)
        elif IGE and not DGE:
            self._genoCov = FixedCov(_ZKallZ)
        elif DGE and IGE:
            if vc_init_type == 'lowrank': #to initialize to |corr_Ads| = 1. Default as key point of SGE paper is to show that |corr_Ads| != 1. For other studies initialize at random I should think.
                self._genoCov = DirIndirCov(self.kinship,Z,kinship_all=self.kinship_all,kinship_cross=self.kinship_cross,C=LowRankCov(2,1))
            elif vc_init_type == 'diagonal': #to initialize to corr_Ads = 0 
                self._genoCov = DirIndirCov(self.kinship,Z,kinship_all=self.kinship_all,kinship_cross=self.kinship_cross,C=DiagonalCov(2))
            elif vc_init_type == 'random_diagonal': #to initialize as a random diagonal matrix (ie corr_Ads = 0)
                self._genoCov = DirIndirCov(self.kinship,Z,kinship_all=self.kinship_all,kinship_cross=self.kinship_cross)
                #create a vc_init here (as opposed to having one passed on from the user) that is diagonal and has random parameters. ie initialises at cor = 0 
                self.vc1_init = DirIndirCov(self.kinship,Z,kinship_all=self.kinship_all,kinship_cross=self.kinship_cross,C=DiagonalCov(2))
                self.vc1_init.setRandomParams()
            else:
                self._genoCov = DirIndirCov(self.kinship,Z,kinship_all=self.kinship_all,kinship_cross=self.kinship_cross)
        else:
            self._genoCov = None


        #define the overall environmental covariance matrix        
        if SimplifNonIdableEnvs:
            assert self.cage_all is not None, 'Cant run IGE analysis without cage info'
            assert IEE and cageEffect,  'No need to simplify environmental effects if IEE or cageEffects are omitted'
            assert len(np.unique(self.cage_density)) == 1, 'Not all groups of equal size - environmental effects might be identifiable'
            N = self.pheno.shape[0]
            uCage = np.unique(self.cage)
            #W, the cage design matrix, is N x n_cages (where N is number of focal animals) 
            W = np.zeros((N,uCage.shape[0]))
            for cv_i, cv in enumerate(uCage):
                W[:,cv_i] = 1.*(self.cage[:,0]==cv)
            #WWt, the cage effect covariance matrix, is N x N and has 1s in cells WWt_i,i (hence WWt it is different from ZZt, which have "number of mice in the cage - 1" in Z_i,i)
            WW = np.dot(W,W.T)
            #NO RESCALING HERE
            self._envCov = SigmaRhoCov(WW)
            self._cageCov = None

        else:
            if IEE:
                assert self.cage_all is not None, 'Cant run IGE analysis without cage info'
                _ZZ  = np.dot(Z,Z.T)
                sf_ZZ = covar_rescaling_factor(_ZZ)
                self.env_all *= sf_ZZ
                self.env_cross *= np.sqrt(1 * sf_ZZ)
                self._envCov = DirIndirCov(self.env,Z,kinship_all=self.env_all,kinship_cross=self.env_cross)
            else:
                self._envCov = FixedCov(self.env)

            ##define cage effect covariance matrix
            if cageEffect:
                assert self.cage_all is not None, 'Cant run IGE analysis without cage info'
                N = self.pheno.shape[0]
                uCage = np.unique(self.cage)
                #W, the cage design matrix, is N x n_cages (where N is number of focal animals) 
                W = np.zeros((N,uCage.shape[0]))
                for cv_i, cv in enumerate(uCage):
                    W[:,cv_i] = 1.*(self.cage[:,0]==cv)
                #WWt, the cage effect covariance matrix, is N x N and has 1s in cells WWt_i,i (hence WWt it is different from ZZt, which has "number of mice in the cage - 1" in Z_i,i)
                WW = np.dot(W,W.T)

                #this is equivalent to getting covar_rescaling_factor first and then multiplying, as done for other matrices above
                WW = covar_rescale(WW)
                self._cageCov = FixedCov(WW)
            else:
                self._cageCov = None

        # define overall covariance matrix as sum of genetic, environmental and cage covariance matrices
        if self._genoCov is None:
            if self._cageCov is None:
                self.covar = SumCov(self._envCov)
            else:
                self.covar = SumCov(self._envCov,self._cageCov)
        else:
            if self._cageCov is None:
                self.covar = SumCov(self._genoCov,self._envCov)
            else:
                self.covar = SumCov(self._genoCov,self._envCov,self._cageCov)

        ## define gp
        self._gp = GP(covar=self.covar,mean=mean)
        

    def optimize(self, vc_init, vc_init_type):
        """optimises the covariance matrix = estimate variance components"""
        if vc_init_type == 'lowrank' or vc_init_type == 'diagonal':
                #as soon as set them approximation is made so don't expect them to be the same
                self._genoCov.setCovariance(vc_init._genoCov.C.K())
                #as soon as set them 10-4 added to diagonal so don't expect them to be the same
                self._envCov.setCovariance(vc_init._envCov.C.K())
                self._cageCov.scale =  vc_init._cageCov.scale

        elif vc_init_type == 'random_diagonal':
                self._genoCov.setCovariance(self.vc1_init.C.K())
                #as soon as set them 10-4 added to diagonal so don't expect them to be the same
                self._envCov.setCovariance(self.vc1_init.C.K())
                self._cageCov.setRandomParams()
        else:
            print('using random freeform matrix for initialization')
            self._gp.covar.setRandomParams()
            if isinstance(self._genoCov,DirIndirCov):
                init_corr = self._genoCov.C.K()[0,1]/np.sqrt(self._genoCov.C.K()[0,0]*self._genoCov.C.K()[1,1])
                print('corr_Ads init is ' + str(init_corr))

        #optimization - keep calc_ste = False as we don't want the STE for the variance components: we want them for the proportions of phenotypic variance explained by xx
        print("about to optimize")
        self.conv, self.info = self._gp.optimize(calc_ste = False)
        print("done optimizing")
        
    def get_VCs(self, calc_ste, DGE, IGE, IEE, cageEffect, SimplifNonIdableEnvs):
        """function to access estimated variance components, standard errors"""

        R = {}
        #whether the run converged
        R['conv'] = self.conv
        #should be small (e.g. < 10^-4)
        R['grad'] = self.info['grad']
        #Contrary to what it says, this is -LML
        R['LML']  = self._gp.LML()
        #number of focal animals
        R['sample_size'] = len(self.sampleID)
        #number of _all animals.
        R['sample_size_all'] = self.real_sample_size_all
        
        #effect sizes of fixed effects in the model
        R['covariates_names'] = self.covariates_names
        R['covs_betas'] = self._gp.mean.b[:,0]

        # genetic variance components and genetic variance
        if DGE and (not IGE):
            R['var_Ad'] = self._genoCov.scale
            R['var_As'] = (-999)
            R['total_gen_var'] = 1/covar_rescaling_factor(self._genoCov.K())
            R['corr_Ads'] = (-999)
        elif IGE and (not DGE):
            R['var_Ad'] = (-999)
            R['var_As'] = self._genoCov.scale
            R['total_gen_var'] = 1/covar_rescaling_factor(self._genoCov.K())
            R['corr_Ads'] = (-999)
        elif DGE and IGE:
            R['var_Ad'] = self._genoCov.C.K()[0,0]
            R['var_As'] = self._genoCov.C.K()[1,1]
            R['total_gen_var'] = 1/covar_rescaling_factor(self._genoCov.K())
            R['corr_Ads'] = self._genoCov.C.K()[0,1]/np.sqrt(self._genoCov.C.K()[0,0]*self._genoCov.C.K()[1,1])

            print('final corr_Ads is ' + str(R['corr_Ads']))
        else:
            R['var_Ad'] = (-999)
            R['var_As'] = (-999)
            R['total_gen_var'] = (-999)
            R['corr_Ads'] = (-999)

        if not SimplifNonIdableEnvs:
            R['env_rho'] = (-999)
            R['env_sigma2'] = (-999)

            if not IEE:
                R['var_Ed'] = self._envCov.scale
                R['var_Es'] = (-999)
                R['corr_Eds'] = (-999)
            else:
                R['var_Ed'] = self._envCov.C.K()[0,0]
                R['var_Es'] = self._envCov.C.K()[1,1]
                R['corr_Eds'] = self._envCov.C.K()[0,1] / (np.sqrt(R['var_Ed']) * np.sqrt(R['var_Es']))

            #cage VC and environmental variance
            if cageEffect:
                R['var_C'] = self._cageCov.scale
                #environmental covariance matrix (fitted)
                envK = self._envCov.K() + self._cageCov.K()
            else:
                R['var_C'] = (-999)
                envK = self._envCov.K()
        else:
            R['var_Ed'] = (-999)
            R['var_Es'] = (-999)
            R['corr_Eds'] = (-999)
            R['var_C'] = (-999)

            R['env_rho'] = self._envCov.rho
            R['env_sigma2'] = self._envCov.scale
            envK = self._envCov.K()

        R['total_env_var'] = 1/covar_rescaling_factor(envK)
            
        #overall (total) covariance matrix (fitted)
        if DGE or IGE:
           totK = self._genoCov.K() + envK
        else:
            totK = envK
        R['total_var'] = 1/covar_rescaling_factor(totK)
        
        if calc_ste and not SimplifNonIdableEnvs: #at this point cannot calculate STE for genetic parameters when SimplifNonIdableEnvs is used 
            STE_output = self.getGenoSte(DGE, IGE, IEE, cageEffect)
            rho_ste = STE_output['rho_ste']
            genetic_STEs = STE_output['R']
            corr_params = STE_output['corr_params']            
        else:
            genetic_STEs = np.array([[-999,-999],[-999,-999]])
            corr_params = (-999)
            rho_ste = (-999)

        #STEs below will be -999 if no corresponding term in the model (as handled by if/else above)
        #standard error for *the proportion of variance explained by* DGE (not the VC)
        R['STE_Ad'] = genetic_STEs[0,0]
        #standard error for *the proportion of variance explained by* IGE (not the VC)
        R['STE_As'] = genetic_STEs[1,1]
        #standard error for *the proportion of variance explained by* the covariance between DGE and IGE (not the VC)
        R['STE_Ads'] = genetic_STEs[0,1]
        #covariance between DGE and IGE *estimates*
        R['corr_params'] = corr_params
        #standard error for the correlation between DGE and IGE
        R['STE_corr_Ads'] = rho_ste

        return R

    def getGenoSte(self, DGE, IGE, IEE, cageEffect):

        #this function is complex because what we do here is calculate the STE not for the variance components but for the proportion of phenotypic variance explained by xx
        # see STE_props_var.pdf from Paolo Casale for the derivations

        assert DGE or IGE, sys.exit("can't get STE for genetic parameters when DGE = None and IGE = None")
        #self._gp.covar.getFisherInf()
        F = self._gp.covar.getFisherInf()

        # scalar in front of each term
        # ordering for geno and env is: direct, covar, indirect (as in fisher matrix)
        aP = []
        vi = []

        if DGE and (not IGE):
            aP.append(self._genoCov.scale)
            vi.append(1. / covar_rescaling_factor(self._genoCov.K0))
        elif IGE and (not DGE):
            aP.append(self._genoCov.scale)
            vi.append(1. / covar_rescaling_factor(self._genoCov.K0))
        elif DGE and IGE:
            aP.append(self._genoCov.C.K()[0,0])
            aP.append(self._genoCov.C.K()[0,1])
            aP.append(self._genoCov.C.K()[1,1])
            vi.append(1. / covar_rescaling_factor(self._genoCov._K))
            vi.append(1. / covar_rescaling_factor(self._genoCov._KZ + self._genoCov._ZK))
            vi.append(1. / covar_rescaling_factor(self._genoCov._ZKZ))
        else:
            pass

        if not IEE:
            aP.append(self._envCov.scale)
            vi.append(1. / covar_rescaling_factor(self._envCov.K0))
        else:
            aP.append(self._envCov.C.K()[0,0])
            aP.append(self._envCov.C.K()[0,1])
            aP.append(self._envCov.C.K()[1,1])
            vi.append(1. / covar_rescaling_factor(self._envCov._K))
            vi.append(1. / covar_rescaling_factor(self._envCov._KZ + self._envCov._ZK))
            vi.append(1. / covar_rescaling_factor(self._envCov._ZKZ))

        if cageEffect:
            aP.append(self._cageCov.scale)
            vi.append(1. / covar_rescaling_factor(self._cageCov.K0))
        else:
            pass

        # make them vectors
        aP = np.array(aP)
        vi = np.array(vi)

        # overall variance
        # this should correspond to the one we get from sampling (checked)
        v = (aP*vi).sum()

        # fractions of variance exaplined by each term
        # (can be negative)
        h = (aP*vi) / v

        # jacobean
        J = np.zeros((aP.shape[0], aP.shape[0]))
        J[:, 0] = h / vi
        J[-1, 1:] = -v / vi[-1]
        for i in range(aP.shape[0]-1):
            J[i, i+1] = v / vi[i]

        # transformation of Fisher
        Fnew = np.dot(J.T, np.dot(F, J))

        # invert the new Fisher
        S,U = np.linalg.eigh(Fnew)
        I = S>1e-9
        U = U[:,I]
        S = S[I]
        FI = np.dot(U,np.dot(np.diag(S**(-1)),U.T))
        # reorder to have same ordering as before (see Equation 5 in STE_props_var.pdf: denominator as d(sigma2, h0...) - here we're moving sigma2 to the end)
        idxs = list(range(1, aP.shape[0]))
        idxs.append(0)
        FI = FI[idxs, :][:, idxs]
        # R is 2x2 matrix: STE_Ad and STE_As on diag, STE_Ads off
        R = np.zeros((2, 2))
        STE_output = {}

        if DGE and IGE:
            FI_geno = FI[:3,:][:,:3]
            #STEs = np.sqrt(FI_geno.diagonal()) ( ordered as Ad Ads As)
            #STEs = sqrt of var of VC corr_params 
            #fills diag and 1 off first
            R[np.tril_indices(2)] = np.sqrt(FI_geno.diagonal())
            #now fills other off
            R = R + R.T - np.diag(R.diagonal())
        
            corr_param_Ad_As = FI_geno[0,2]/(np.sqrt(FI_geno[0,0])*np.sqrt(FI_geno[2,2]))

            #the two blocks below - which calculate STE of rho - are independent of all lines above in getGenoSte
            #calculating STE for rho is much simpler than calculating STE for the proportions of phenotypic variance explained by the other params
            def get_rho_ste(va, vab, vb, V):
                # V ordered as va vab vb
                Drho = np.zeros(3)
                Drho[0] = (-vab) / (2*va*np.sqrt(va * vb))
                Drho[1] = 1/np.sqrt(va*vb) 
                Drho[2] = (-vab) / (2*vb*np.sqrt(va * vb))
                rho_ste = np.sqrt(np.dot(Drho.T, np.dot(V, Drho)))
                return rho_ste

            #V required in get_rho_ste to calcualte STE on rho
            V = la.pinv(self._gp.covar.getFisherInf())
            V = V[:3,:3] # variance matrix of estimates of [direct, covariance, indirect]
            rho_ste = get_rho_ste(self._genoCov.C.K()[0,0],self._genoCov.C.K()[0,1],self._genoCov.C.K()[1,1],V)

        
        elif DGE and (not IGE):
            R[0,0] = np.sqrt(FI[0,0])
            R[0,1] = -999
            R[1,0] = -999
            R[1,1] = -999
            corr_param_Ad_As = -999
            rho_ste = -999
        
        elif (not DGE) and IGE:
            R[0,0] = -999
            R[0,1] = -999
            R[1,0] = -999
            R[1,1] = np.sqrt(FI[0,0])
            corr_param_Ad_As = -999
            rho_ste = -999
       
        else:
            R[0,0] = -999
            R[0,1] = -999
            R[1,0] = -999
            R[1,1] = -999
            corr_param_Ad_As = -999
            rho_ste = -999

        
        STE_output['R']=R
        STE_output['corr_params']= corr_param_Ad_As
        STE_output['rho_ste']= rho_ste

        return STE_output

    def getOutput(self):
        """to get output without having to specify DGE, IGE, ...."""
        return self.output
    


