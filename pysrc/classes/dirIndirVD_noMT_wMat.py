#Before starting: 
#1. Check to have 'diagcov' 'dirIndirCov_v2' 'dirIndirCov_mt' in "classes" subfolder where exp_bivar is 
#2. need to have installed limix_core 

#IGE (Indirect genetic effects) = SGE (Social genetic effects). 
#IEE: Indirect environmental effects

#the potential presence of NAs in input phenotype, covs, cages etc means that in this code we introduce two sets of animals (not discussed in the paper):
#FOCAL animals, defined as having phenotype, covs (if covs are provided), cage (if cages are provided, which is not necessary in the case where DGE but no IGE are modelled) and kinship
#ALL animals with cage and kinship and in subset_IDs; this set of animals is referred to using _all in this code. 
#!! there can be animals in _all that are not included in the analysis (e.g after NA in phenotypes are filtered out)
#Note that in the paper cage mate has a different meaning, namely "the other mice in the cage of the focal individual".

import sys
import warnings
import time
import pdb
import torch
import random
import re # needed for regular expessions
import numpy as np
import scipy.linalg as la

#dirIndirCov_mt allows to pass low rank 4x4 genetic matrix (Ad_1, Ad_2, As_1, As_2, and their covariances). 
#from classes.dirIndirCov_mt import DirIndirCovMT
#dirIndirCov_v2 allows to pass low rank 2x2 genetic matrix (DGE, IGE and their covariance).
from classes.dirIndirCov_v2 import DirIndirCov
from classes.SigmaRhoCov import SigmaRhoCov

#you need to install limix_core from https://github.com/limix/limix-core
from limix_core.mean.mean_base import MeanBase as lin_mean
from limix_core.covar.fixed import FixedCov 
from limix_core.covar import LowRankCov 
from limix_core.covar import DiagonalCov 
from limix_core.covar import FreeFormCov 
from limix_core.covar import KronCov
from limix_core.covar.combinators import SumCov
from limix_core.util.preprocess import covar_rescaling_factor
from limix_core.util.preprocess import covar_rescale
from limix_core.gp.gp_base import GP


start_time = time.time()
print("Starting DirIndirVD")
  
class DirIndirVD():

    #many of inputs below are passed from a SocialData object built using social_data.py
    #pheno: missing values should be encoded as -999 (needs to be numeric for HDF5). 
    #       UNIV (N): pheno can be 1xN or Nx1 vector (where N is number of individuals) 
    #       BIVAR (2N): pheno comes from social_data as CONCATENATED PHENOS - i.e. 2Nx1 or 1x2N
    #pheno_ID (N): IDs corresponding to pheno. Corresponds to N in any case.
    #covs: missing values should be encoded as -999.  
    #      UNIV: covs can be 1xN or Nx1 vector, or a NxK matrix (K number of covariates). intercept column not expected. cage_density not necessary. ok if present as long as option independent_covs = True
    #      BIVAR: covs can be 1x2N or 2Nx1 vector, or a 2NxK matrix (K number of covariates). For pheno-specific covs, e.g. pheno1, should have Nx1 of values concat to vector of 0s for pheno2 (viceversa, 0s and then Nx1 of values if pheno2-specific cov) 
    #             intercept column not expected. cage_density not necessary. ok if present as long as option independent_covs = True
    #covs_ID (N): IDs corresponding to covs. 
    #covariates_names: name of covariates in covs
    #kinship_all (N_a x N_a): GRM for all mice ever considered (focal or cagemate or nothing). no missing values allowed. symmetric matrix.
    #kinship_all_ID (N_a): rownames and colnames of kinship_all
    #cage_all (N_a): cages of all mice ever considered (focal or cagemate or nothing). missing values should be encoded as NA. NA cage will lead to this animal being ignored from the analysis. if you have reason to believe this animal was a cage mate (but you don't know its cage), this is an issue and you need to be aware of it.
    #cage_all_ID (N_a): IDs corresponding to cage_all
    
    #independent_covs: if True (default) LIMIX will check whether any covariate is a linear combination of the others and if so will fix the issue by updating covs to independent columns
    #DGE: should DGE be included in the model?
    #IGE: should IGE be included in the model?
    #IEE: should IEE be included in the model? note that if IGE = True, IEE will be set to True by the program (silently) - see below
    #cageEffect: should cage effects be included in the model?
    #calc_ste: should the standard errors of the variance components be estimated and output? (09/19/23: became True by default)
    #standardize_pheno: should the phenotype be standardized to variance 1? better to do so (numerically more stable)
    #subset_IDs: subset of IDs to consider *as focal individuals or cage mate*. this means that any animal not in subset_IDs will be ignored completely from analysis. This will be an ok thing to do only if all animals from a cage are excluded! if instead want to exclude animals only as focal individuals, set their phenotype to NA.
    #subset_on_cage: defaults to False. Turn to True if cageEffect, IEE and IGE = False but still want to only consider those animals that have a non-NA cage
    #vc_init_type: characteristics of the 2x2 matrix that sigma_Ad^2, var_Ads, sigma_As^2. 
    #Can be lowrank and used in combination with providing vc_init, which would have been estimated in prior run
    #Can also be diagonal and used in combination with vc_init
    #Can also be random diagonal to initialise at corr_Ads = 0 but without providing vc_init
    #Can also be None in which case it will be a freeform 2x2 matrix (good for usual cases I think)
    #vc_init: set to None or an instance of DirIndirVD in a second run
    #std_genos = True: standardize genotypes. used to calculate variance explained by SNP
    #SimplifNonIdableEnvs: defaults to False. When all groups are of equal size, non-genetic variance components are not identifiable. Can (but don't have to) turn SimplifNonIdableEnvs to True to report rho instead of individual environmental VCs. No STE yet on rho I think. genetic estimates will be valid no matter what.

    ###comments TOREMOVE###
    #FYI: at 27/05/24: 
    # vc_init = None - defined in exp_bivar12
    # vc_init_type = None - defined in exp_bivar12
    # subset_on_cage = False - defined in exp_bivar12
    # calc_ste = True put by default at 09/19/23
    ### end of remove ###
    
    def __init__(self, vc_init, vc_init_type , pheno = None, pheno_ID = None, covs = None, covs_ID = None, covariates_names = None, kinship_all = None, kinship_all_ID = None, cage_all = None, cage_all_ID = None, maternal_all = None, maternal_all_ID = None, independent_covs = True, DGE = False, IGE = False, IEE = False, cageEffect = False, maternalEffect = False, calc_ste = True, standardize_pheno = True, subset_IDs = None, subset_on_cage = False, std_genos = False, SimplifNonIdableEnvs = False, seed = None): 

        # the purpose of the IEE argument is to specify whether IEE should be off when IGE are off. When IGE are on, IEE must be on and therefore IEE will be automatically set to True when IGE is True.
        if IGE:
            IEE = True
        
        if pheno.shape[0] == len(pheno_ID): 
            bivariate = False
        elif pheno.shape[0] == (2*len(pheno_ID)): 
            bivariate = True
        else:
            sys.exit('length of pheno_ID and pheno not compatible')
        
        # set seed for reproducible randomness (NB: not all analysis have randomness, so seed will be relevant only when random is involved)
        np.random.seed(seed) # if seed is None, no seed is set, so will be random; None is the default
        print("seed =", seed)

        #1. parse input:  - filter out when needed, allign ID for kinship, cage and dam;  - build Iok;   - get sample sizes;   - standardize pheno, covs and geno;
        self.parseNmore(pheno, pheno_ID, covs, covs_ID, covariates_names, kinship_all, kinship_all_ID, cage_all, cage_all_ID, maternal_all, maternal_all_ID, independent_covs, standardize_pheno, subset_IDs, cageEffect, maternalEffect, IEE, subset_on_cage, std_genos, bivariate)

        #2. define the genetic, environmental and cage covariance matrices
        self.VD(DGE = DGE, IGE = IGE, IEE = IEE, cageEffect = cageEffect, maternalEffect = maternalEffect, vc_init = vc_init, vc_init_type = vc_init_type, SimplifNonIdableEnvs = SimplifNonIdableEnvs, bivariate = bivariate)

        #3. optimize to estimate the variance components
        self.optimize(vc_init = vc_init,  vc_init_type = vc_init_type, bivariate = bivariate)

        if self.optim_error is None:
        #4. get variant components with get_VCs only if optim_error is None, look at optimize() to know more
        # self.output useful to retrieve output using getOutput without having to specify calc_ste, DGE, IGE, etc.
            self.output = self.build_output(calc_ste = calc_ste, DGE = DGE, IGE = IGE, IEE = IEE, cageEffect = cageEffect, maternalEffect = maternalEffect, SimplifNonIdableEnvs = SimplifNonIdableEnvs, bivariate = bivariate)
        else:
            self.output = {'sampleID_all' : self.sampleID_all, 'pheno' : self.pheno,'covs' : self.covs, 'optim_error': self.optim_error} # Can expand this with what wanted

        
    def parseNmore(self, pheno, pheno_ID, covs, covs_ID, covariates_names, kinship_all, kinship_all_ID, cage_all, cage_all_ID, maternal_all, maternal_all_ID, independent_covs,standardize_pheno, subset_IDs,cageEffect, maternalEffect, IEE, subset_on_cage, std_genos, bivariate):
      
        """match various inputs"""

        assert pheno is not None, 'Specify pheno!'
        assert pheno_ID is not None, 'Specify pheno IDs!'
        assert kinship_all is not None, 'Specify kinship!'
        assert kinship_all_ID is not None, 'Specify kinship IDs!'
        assert kinship_all.shape[0] == kinship_all.shape[1], 'Kinship is not a square matrix!'
        assert kinship_all.shape[0] == len(kinship_all_ID), 'Dimension of kinship and length of kinship IDs do not match!'

        # TOREMOVE: lines below used to check if working when mismatched: pheno_ID, kinship_ID, cage_ID and covs_ID not same order
        #pheno_ID_or = pheno_ID # stored everything somewhere to check and debug
        #pheno_or = pheno
        #covs_ID_or = covs_ID
        #covs_or = covs
        #kinship_all_ID_or = kinship_all_ID
        #kinship_all_or = kinship_all
        #cage_all_ID_or = cage_all_ID
        #cage_all_or = cage_all
        #maternal_all_ID_or = maternal_all_ID
        #maternal_all_or = maternal_all
        #
        ### Removing one from cage info --> still working (tested on two pairs of phenos, roughly same results)
        #cage_all_ID = cage_all_ID[1:]
        #cage_all = cage_all[1:]
        #
        ### Removing one from kinship info --> still working (tested on two pairs of phenos, roughly same results )
        #kinship_all_ID = kinship_all_ID[2:]
        #kinship_all = kinship_all[2:,2:]
        #
        ### Removing one from maternal info
        #maternal_all_ID = maternal_all_ID[:-1]
        #maternal_all = maternal_all[:-1]

        
        ## Shuffling pheno_IDs --> still working even though I got a similar result but NOT IDENTICAL as I was expecting (tested on two pairs of phenos, roughly same results)
        ## apparently the order matters?
        #pidxs =np.arange(len(pheno_ID))
        #np.random.shuffle(pidxs)
        #pheno_ID = pheno_ID[pidxs]
        #pheno = pheno[np.append(pidxs,pidxs+int(len(pheno_ID))),:] # NB: impo: if doing it on pheno_ID have to do the same on pheno
        ##pheno_ID = np.append(pheno_ID[int(len(pheno_ID)/2):],pheno_ID[:int(len(pheno_ID)/2)]) #if do it like this have to do the same with pheno
        ##pmatch = np.nonzero(pheno_ID[:,np.newaxis]==pheno_ID_or)
        ##pheno = pheno[np.append(pmatch[1],pmatch[1]+int(len(pheno_ID))),:]
        #print("pheno_ID are shuffled")
        
        
        ## Shuffling covs_ID --> working and getting IDENTICAL results to not shuffled covs_ID 
        #pidxs =np.arange(len(covs_ID))
        #np.random.shuffle(pidxs)
        #covs_ID = covs_ID[pidxs]
        #covs = covs[np.append(pidxs,pidxs+int(len(covs_ID))),:] # NB: impo: if doing it on covs_ID have to do the same on covs
        #print("pheno_ID are shuffled")
        #pdb.set_trace()
        ### end remove

        #assert not calc_ste or not SimplifNonIdableEnvs, sys.exit("can't get STE with SimplifNonIdableEnvs")
        
        ####hack to shorten runtime or debug
        #print('ran a hack to shortun runtime - THIS CANNOT BE A FINAL RESULT')
        #uCage=np.unique(cage_all)
        #assert len(uCage) > 120, 'Too few'
        #nb_cages_to_keep = 20
        #uCage = random.choices(uCage,k=nb_cages_to_keep)
        #idx_keep = np.concatenate([np.where(cage_all==uCage[i])[0] for i in range(len(uCage))])
        #mask = np.zeros(cage_all.shape,dtype=bool)
        #mask[idx_keep] = True
        #cage_all[~mask]='NA'

        #used to be:
        #nb_to_remove = (len(uCage) - nb_cages_to_keep)
        #remove=uCage[0:nb_to_remove]
        #idx_remove = np.concatenate([np.where(cage_all==remove[i])[0] for i in range(len(remove))])
        #cage_all[idx_remove]='NA'
        #
        #pdb.set_trace() #len(np.unique(cage_all[cage_all!='NA'])) -> 'Q-51', 'Q-52', 'Q-6', 'Q-7', 'Q-8'
        ####hack to shorten runtime


        #1. define set of animals in subset IDs, with kinship information and non-NA cage if cage requested (_all). 
        #1.1 _all animals need to be in subset_IDs. !!!! this is only correct is subset_IDs constructed so that all animals in a cage are in or out. not correct to include only some animals in a cage.
        if subset_IDs is not None: # subset_IDs = all IDs in normal conditions
            Imatch = np.nonzero(subset_IDs[:,np.newaxis]==kinship_all_ID) 
            kinship_all = kinship_all[Imatch[1],:][:,Imatch[1]]
            kinship_all_ID=kinship_all_ID[Imatch[1]]
        
        #1.2A. NA allowed in cage information so first of all exclude missing cage data and corresponding animals
        if cageEffect or IEE or subset_on_cage:
            assert cage_all is not None, 'Specify cage!'
            assert cage_all.shape[0] == len(cage_all_ID), 'Lengths of cage_all and cage_all_ID do not match!'
            
            #Accounting for the possibility that we have animals in cage_all_ID that do have a cage and cage mates but are not in kinship_all
            if IEE: # when doing IGE, if one animal in cage is missing genotype, we remove the whole cage 
                #first round to turn all values of cage_all that correspond to animals in cages with a missing genotype to NA
                Imatch = np.nonzero(cage_all_ID[:,np.newaxis]==kinship_all_ID)
                cages_w_missingGenos = np.unique(np.delete(cage_all, Imatch[0]))
                motch = np.nonzero(cages_w_missingGenos[:,np.newaxis]==cage_all)
                cage_all[motch[1]] = 'NA'
            #second round - go as before
            has_cage = (cage_all!='NA')
            if sum(has_cage)==0:
                cage_all = None
                assert cage_all is not None, 'All cages missing'
            cage_all=cage_all[has_cage] # selecting only individuals with cage
            cage_all_ID=cage_all_ID[has_cage] # selecting only ind_IDs with cage

            #1.3 match cages and kinship 
            Imatch = np.nonzero(cage_all_ID[:,np.newaxis]==kinship_all_ID)
            cage_all_ID = cage_all_ID[Imatch[0]]
            cage_all = cage_all[Imatch[0]]
            
            kinship_all = kinship_all[Imatch[1],:][:,Imatch[1]]
            kinship_all_ID = kinship_all_ID[Imatch[1]]
            #(kinship_all_ID==cage_all_ID).all()
            #True
            #cage and kinship now have no missing values and are matched - IDs are in cage_all_ID and kinship_all_ID
        
        #1.2B. NA allowed in dam information so first of all exclude missing individuals with missing dam
        # NB: dam with just 1 individual are allowed - TODO: change code in case you don't want that
        if maternalEffect:
            assert maternal_all is not None, 'Specify dam!'
            assert maternal_all.shape[0] == len(maternal_all_ID), 'Lengths of maternal_all and maternal_all_ID do not match!'
            
            #Removing animals that do NOT have dam info - are NA in dam
            has_dam = (maternal_all!='NA')
            if sum(has_dam)==0:
                maternal_all = None
                assert maternal_all is not None, 'all dams missing'
            maternal_all=maternal_all[has_dam] # selecting only individuals with dam
            maternal_all_ID=maternal_all_ID[has_dam] # selecting only ind_IDs with dam
            
            #1.3 match cages and maternal.
            Imatch = np.nonzero(maternal_all_ID[:,np.newaxis]==kinship_all_ID)
            maternal_all_ID = maternal_all_ID[Imatch[0]]
            maternal_all = maternal_all[Imatch[0]]
            
            # if have cage and cage_all_ID match them as well
            if cageEffect or IEE or subset_on_cage:
                cage_all_ID = cage_all_ID[Imatch[1]]
                cage_all = cage_all[Imatch[1]]

            kinship_all = kinship_all[Imatch[1],:][:,Imatch[1]]
            kinship_all_ID = kinship_all_ID[Imatch[1]]
            #(kinship_all_ID==cage_all_ID).all()
            #(kinship_all_ID==maternal_all_ID).all()
            # True - checked
        # now maternal, cage and kinship have no missing values and are matched - IDs are the same in maternal_all_ID and cage_all_ID and kinship_all_ID 
        
        # put IDs in sampleID_all
        #pdb.set_trace() #check that subsetted on cage; run below and len(sampleID_all)
        sampleID_all = kinship_all_ID   #from now on using sampleID_all for kinship_all and cage_all and maternal_all
        assert len(sampleID_all)!=0, 'No _all animals'
        print('Number of mice in subset, with kinship information and with cage info if cage requested: '  + str(len(sampleID_all)))
        
        
        #2. define FOCAL animals now: those in sampleID_all that have non missing phenotype and non missing covs
        if covs is not None: 
            #2.1 match pheno_ID and covs_ID in case come from different matrices with different samples, should both be N
            Imatch = np.nonzero(pheno_ID[:,np.newaxis]==covs_ID)
            #pdb.set_trace() # check in case have been shuffled, all(pidxs == Imatch[1])
            if bivariate is False:
                pheno_ID = pheno_ID[Imatch[0]]
                pheno = pheno[Imatch[0],:] 
                covs_ID = covs_ID[Imatch[1]]
                covs = covs[Imatch[1],:] # There was a bug here
                assert all(pheno_ID==covs_ID), 'cannot match pheno_ID and covs_ID'
            else: 
                Imatch0_rep = np.append(Imatch[0],Imatch[0]+int(len(pheno_ID))) #use to sort pheno
                Imatch1_rep = np.append(Imatch[1],Imatch[1]+int(len(covs_ID))) #use to sort covs
                
                pheno = pheno[Imatch0_rep,:] #length 2N
                pheno_ID=pheno_ID[Imatch[0]] #length N
                covs = covs[Imatch1_rep,:]
                covs_ID = covs_ID[Imatch[1]]
                assert all(pheno_ID==covs_ID), 'cannot match pheno_ID and covs_ID'
            #pheno and covs now are matched to pheno_ID
        
        #2.2 add intercepts
        if covs is None:
            covs = np.ones((pheno.shape[0],1)) # vector of 1s that needs to be 2N in bivariate
            covariates_names = "mean"
        else:     
            covs=np.append(np.ones((covs.shape[0],1)),covs,1) #added always, no matter what covs is
            covariates_names = np.append("mean",covariates_names) #mean assigned to covariates_names  

        if bivariate is True: #bivar so add trait-specific intercept 
            add = np.append(np.ones((int(pheno.shape[0]/2),1)),np.zeros((int(pheno.shape[0]/2),1)),0)
            covs= np.append(add,covs,1) 
            covariates_names = np.append("traitspe mean",covariates_names) #so order is "traitspe mean" "mean" covariates_names

        #2.3. subset pheno and covs to keep only indivs in kinship_all 
        # check which of those are in sampleID_all (and thus are in subset_IDs, have kinship and cage if cage important)
        has_geno = np.array([pheno_ID[i] in sampleID_all for i in range(pheno_ID.shape[0])])
        assert sum(has_geno)!=0, 'No indivs shared between pheno/covs and kinship_all/cage_all' #check this
        pheno_ID=pheno_ID[has_geno]
        if bivariate is True: 
            pheno = pheno[np.append(has_geno,has_geno,0),:]
            covs = covs[np.append(has_geno,has_geno,0),:]
        else:
            pheno = pheno[has_geno,:]
            covs = covs[has_geno,:]

        #3. create cage and kinship for focal animals
        #3.1 create kinship for focal animals (have geno values AND phenos, which may still have NAs; non-focal don't have pheno)
        idxs = np.array([np.where(sampleID_all==pheno_ID[i])[0][0] for i in range(pheno_ID.shape[0])]) 
        kinship=kinship_all[idxs,:][:,idxs] # kinship_all is N_a X N_a; kinship is NxN (N is a subset of N_a)
        
        #3.2 create focal x _all genetic cross-covariance
        kinship_cross = kinship_all[idxs,:] # NB: kinship_cross is NxN_a
        #so pheno_ID along rows and sampleID_all along colummns
        
        #3.3 create maternal for focal animals
        if maternalEffect: 
            maternal = maternal_all[idxs]
            if len(maternal.shape)==1:
                maternal = maternal[:,np.newaxis]

        #3.4 create cage for focal animals
        if cageEffect or IEE or subset_on_cage: 
            cage=cage_all[idxs] # NB: if pheno_ID has different order than sampleID_all, cage and cage_all have different order
            #transpose cage if necessary
            if len(cage.shape)==1:
                cage = cage[:,np.newaxis]
                
            if IEE:
                #create cage density covariate and adds it as a covariate. it's ok if it has already been given as input as a covariate as colinear covariates are pruned out (as long as independent_covs=True; if they're not and 100% colinear the code will crash; if not 100% colinear then it will "only" cost 1 df)
                cage_density = np.array([len(np.where(cage_all == cage[i])[0]) for i in range(cage.shape[0])])
                cage_density = cage_density[:,np.newaxis] # cage_density: (N,1)
                covariates_names = np.append(covariates_names, "cage_density_1")
                
                # Adding cage_density separately per phenos: 
                if bivariate is True:
                    cd_1 = np.append(cage_density, np.zeros(cage_density.shape[0])[:,np.newaxis], 0) # cage_density_1: stack N values then N 0s ; 
                    cd_2 = np.append(np.zeros(cage_density.shape[0])[:,np.newaxis], cage_density, 0) # cage_density_2: stack N 0s then N values
                    cage_density = np.append(cd_1, cd_2, 1) # cage_density becomes (2N,2); in univar is (N,1)
                    covariates_names = np.append(covariates_names,"cage_density_2")
                
                # same as for univ 
                assert cage_density.shape[0] == covs.shape[0], 'length cage and covs diff!'
                covs=np.append(covs,cage_density,1)
                assert covs.shape[1] == covariates_names.shape[0], 'covs and covariates_names dimensions are not consistent'

            else:
                cage_density = None #that's an issue when comparing DGE and IGE models!!! needs to be fixed # TODO
        else:
            cage_density = None

        #4. now create environmental matrices
        env = np.eye(kinship.shape[0]) # N x N 
        env_all = np.eye(kinship_all.shape[0]) # N_a x N_a
        env_cross = env_all[idxs,:] #N x N_a

        #5. buid Iok: for a given sample, require no NAs for any cov for the sample to be kept; if univar, removed NA in pheno as well; if bivariate, can have NA in one of the two phenos (removed if NA for both)
        #pot_2N means 2N if bivar, N if univar 
        #In bivariate: (fig 1)
        #  in this way, if missing values in a cov that is pheno specific, indiv will be kept for the other pheno (for which have no missing values as that cov is not used) 
        #  when the cov with missing value is for both pheno - indiv are filtered out
        Iok_pot2N = np.append(pheno!=(-999), (covs!=(-999)).all(1)[:,np.newaxis], 1).all(1) # this is now a vector
        if bivariate is True: # for MT Iok is 2N - need to have Iok_N, False when both are False; if NA for a cov that is pheno specific, will be False only for that half of covs 
            Iok_N = np.append(Iok_pot2N[:int(len(Iok_pot2N)/2),np.newaxis],Iok_pot2N[int(len(Iok_pot2N)/2):,np.newaxis],1).any(1) 
        else: # this for univariate
            Iok_N = Iok_pot2N
            
        #6. filter phenotypes and covariates to leave out the ones that are False in Iok 
        pheno = pheno[Iok_pot2N] # 2N-x in bivariate, N-x in univ
        covs = covs[Iok_pot2N,] # 2N-x in bivariate, N-x in univ
        #do NOT apply to pheno_ID
        
        #7. calculate sample sizes 
        #7a. calculate focal sample size 
        #    to get focal sample size, filter pheno_ID (N) with first/second half of Iok_pot2N; first half defined by the length of pheno_ID
        #pdb.set_trace()
        self.focal_sample_size1 = len(pheno_ID[Iok_pot2N[ : pheno_ID.shape[0], ]]) # 
        print('sample size focals for trait 1: ' + str(self.focal_sample_size1))
        if bivariate is True:
            self.focal_sample_size2 = len(pheno_ID[Iok_pot2N[ pheno_ID.shape[0] : , ]]) # 
            #TODO: check this assertion 
            # assert self.focal_sample_size1 + self.focal_sample_size2 == len(self.pheno_ID[self.Iok_pot2N]), "pb with focal sample sizes for pheno1 and pheno2"
            print('sample size focals for trait 2: ' + str(self.focal_sample_size2))
        else:
            self.focal_sample_size2 = None
        #7b. union and intersection of focal
        if bivariate is True:
            # union # or - need one of the two True
            self.focal_union = np.logical_or(Iok_pot2N[ : pheno_ID.shape[0], ], Iok_pot2N[ pheno_ID.shape[0] : , ]).sum() 
            # intersection # and - need both True
            self.focal_inter = np.logical_and(Iok_pot2N[ : pheno_ID.shape[0], ],  Iok_pot2N[ pheno_ID.shape[0] : , ]).sum() 
        else:
            self.focal_union = None
            self.focal_inter = None
            
        #7c. calculate real sample size (fig 2)
        print('Number of mice with potentially pheno and covs and kinship and cage and in subset: '  + str(len((pheno_ID))))
        if cageEffect or IEE or subset_on_cage: # probably this is needed only with IEE actually; what about subset_on_cage?
            # Splitting this in 2, for MT:
            #calculate real sample_size_all (i.e. of all the _all animals, which ones are actually in a cage with a focal animal -i.e. in uni, has no NA pheno, in biv has at least one pheno)
            # cage[Iok_N] —> gives me list of cages of animals with no NA pheno (in uni); at least one non-NA pheno (in biv)
            # —> filter out those that don’t have a focal animal, i.e. are in a cage that don’t have at least one animal with non-NA pheno
            is_real_all1 = np.array([cage_all[i] in cage[Iok_pot2N[ : cage_all.shape[0], ]] for i in range(cage_all.shape[0])])
            #old Amelie below
            #is_real_all1 = np.array([cage_all[i] in cage for i in range(cage_all.shape[0])])
            self.real_sample_size_all1 = is_real_all1.sum()
            #equivalent to: 
            idx_real_all1 = np.concatenate([np.where(cage_all==cage[Iok_pot2N[ : cage_all.shape[0], ]][i])[0] for i in range(len(cage[Iok_pot2N[ : cage_all.shape[0], ]]))])
            self.real_sample_size_all1B = len(np.unique(idx_real_all1))
            assert self.real_sample_size_all1==self.real_sample_size_all1B, 'Pb self.real_sample_size_all1'
            print('real sample size cm for trait 1: ' + str(self.real_sample_size_all1))
        else:
            self.real_sample_size_all1 = self.focal_sample_size1 # focal sample size  # (-999)
            print('no cage subset, real sample size cm for trait 1 same as focal sample size: ' + str(self.real_sample_size_all1))

        if bivariate is False:
            self.real_sample_size_all2 = None
            self.cm_union = None
            self.cm_inter = None
            print('univariate mode, real sample size cm for trait 2: ' + str(self.real_sample_size_all2))
        else:
            if cageEffect or IEE or subset_on_cage: 
                is_real_all2 = np.array([cage_all[i] in cage[Iok_pot2N[cage_all.shape[0] : , ]] for i in range(cage_all.shape[0])])
                self.real_sample_size_all2 = is_real_all2.sum()
            
                #equivalent to: 
                idx_real_all2 = np.concatenate([np.where(cage_all==cage[Iok_pot2N[cage_all.shape[0] : , ]][i])[0] for i in range(len(cage[Iok_pot2N[cage_all.shape[0] : , ]]))])
                self.real_sample_size_all2B = len(np.unique(idx_real_all2))
            
                assert self.real_sample_size_all2==self.real_sample_size_all2B, 'Pb self.real_sample_size_all2'
                print('real sample size cm for trait 2: ' + str(self.real_sample_size_all2))
                # union # or - need one of the two True
                self.cm_union = np.logical_or(is_real_all1, is_real_all2).sum() 
                # intersection # and - need both True
                self.cm_inter = np.logical_and(is_real_all1,  is_real_all2).sum() 
            else:
                self.real_sample_size_all2 = self.focal_sample_size2 # (-999) 
                self.cm_union = self.focal_union
                self.cm_inter = self.focal_inter
                print('no cage subset, real sample size cm for trait 2 same as focal sample size: ' + str(self.real_sample_size_all2))


        #8. standardize_pheno
        if standardize_pheno:
            if bivariate is False:
                pheno -= pheno.mean(0)
                pheno /= pheno.std(0)
            else: # pheno_ID.shape[0] = N
                pheno_ID1 = pheno_ID[Iok_pot2N[ : pheno_ID.shape[0], ] ] # phenoIDs for pheno1 - selected with Iok_pot2N first half
                pheno_ID2 = pheno_ID[Iok_pot2N[ pheno_ID.shape[0] : ,] ] # phenoIDs for pheno2 - selected with Iok_pot2N second half # even if not used after, check if filtering ID with Iok is consistent with pheno dim after filtering
                assert pheno_ID1.shape[0] + pheno_ID2.shape[0] == pheno.shape[0], "Cannot separate the two phenotypes, something is wrong with pheno_ID, Iok and pheno"
                
                pheno1 = pheno[ : pheno_ID1.shape[0],] # selecting pheno1 based on length of pheno_ID1
                pheno1 -= pheno1.mean(0)
                pheno1 /= pheno1.std(0)
                
                pheno2 = pheno[pheno_ID1.shape[0] : ,] # selecting pheno2 as what's left after pheno_ID1
                pheno2 -= pheno2.mean(0)
                pheno2 /= pheno2.std(0)
                
                pheno = np.append(pheno1, pheno2,0) # rejoining the two phenos
            print('Pheno has been standardized')

        print("Covariates in dirIndirVD_wMT before making them independent: " + str(covariates_names))        
        
        #9. make covariates independent (fig 5a)
        if independent_covs and covs.shape[1]>1:
            tol = 1e-6
            R = la.qr(covs,mode='r')[0][:covs.shape[1],:]
            I = (abs(R.diagonal())>tol)
            if np.any(~I):
                print('Covariates '+ str(covariates_names[(np.where(~I)[0])]) +' have been removed because linearly dependent on the others')
            covs = covs[:,I]
            covariates_names =  covariates_names[I]
            print('Final covariates are ' + str(covariates_names))

        #10. standardize genotypes (only for univariate model for now) and covariates: 
        #10a. standardize genotypes (e.g. to study var explained by a SNP - then put that SNP as a cov) but not first X covs (mean and - if MT - trait spe mean)
        if std_genos:
            # TODO: implement this for bivariate
            assert bivariate is False, 'cant standardize genotypes for multivariate model yet'
            covs[:,1:] -= np.delete(covs[:,1:], 0, axis=1).mean(0) #check 1 in there
            covs[:,1:] /= np.delete(covs[:,1:], 0, axis=1).std(0)
            print('all covs have been standardized')
        else:
            if covs.shape[1]==1: 
                is_not_direct_geno = [not "_direct" in covariates_name for covariates_name in [covariates_names]]
                is_not_social_geno = [not "_social" in covariates_name for covariates_name in [covariates_names]]
            # If have more than one element in covariates_names (i.e. more than one cov), need to put it in list to have T/F corresponding to each element of covariates_names
            else:
                is_not_direct_geno = [not "_direct" in covariates_name for covariates_name in list(covariates_names)]
                is_not_social_geno = [not "_social" in covariates_name for covariates_name in list(covariates_names)]
                
            is_not_geno = np.logical_and(is_not_direct_geno,is_not_social_geno)
            idxs_not_geno = list(np.where(is_not_geno)[0])

            ### This is how done now, so that a. remove also "traitspe mean" when any; b. stdise per pheno in bivariate
            ##a. removing means - "mean"; and "traitspe mean"(when in MT)
            id_means = np.append(np.where(covariates_names == 'mean')[0], (np.where(covariates_names == 'traitspe mean')[0])) # idxs of "mean" and "traitspe mean"
            idxs_not_geno = [i for i in idxs_not_geno if i not in id_means]
            
            def stdCovs(covs_to_std): # covs_to_std = matrix with covs to std, e.g. covs_not_geno or covs_1_notgeno/covs_2_notgeno
                zeros = np.all(covs_to_std[...,:] == 0, axis=0) # keep out of standirdisation cols that are all 0s - in case of bivariate - covs specific for other pheno (in case of univariate - if all 0s, no need to std)
                covs_nonZero = covs_to_std[:,~zeros]
                covs_nonZero -= covs_nonZero.mean(0)
                covs_nonZero /= covs_nonZero.std(0)
                covs_to_std[:,~zeros] = covs_nonZero
                return covs_to_std
            
            if bivariate is False:
                covs_notgeno = stdCovs(covs[:,idxs_not_geno])
            else: # pheno_ID.shape[0] = N
                pheno_ID1 = pheno_ID[ Iok_pot2N[ : pheno_ID.shape[0],] ] # phenoIDs for pheno1 - selected with Iok_pot2N first part
                pheno_ID2 = pheno_ID[ Iok_pot2N[pheno_ID.shape[0] : ,] ] # phenoIDs for pheno2 - selected with Iok_pot2N second part
                assert pheno_ID1.shape[0] + pheno_ID2.shape[0] == covs.shape[0], "Cannot separate covs into two phenotypes, something is wrong with pheno_ID, Iok and covs"
                
                covs_1_notgeno = covs[ : pheno_ID1.shape[0], idxs_not_geno] #selecting upper part; and where not_geno
                covs_1_notgeno = stdCovs(covs_1_notgeno)
                
                covs_2_notgeno = covs[pheno_ID1.shape[0] : , idxs_not_geno] #selecting lower part; and where not_geno
                covs_2_notgeno = stdCovs(covs_2_notgeno)
                
                covs_notgeno = np.append(covs_1_notgeno, covs_2_notgeno, 0) #stacking again  
                
            covs[:,idxs_not_geno] = covs_notgeno
            print('covs except genos have been standardized')

        #for i in range(covs.shape[1]):
        #    print covariates_names[i]
        #    print covs[:,i].mean(0)
        #    print covs[:,i].std(0)
        # so mean is 1 sd 0 for mean, and mean is ~0 and var 1 for all other non-genotype covariates   

        self.pheno_ID=pheno_ID # (N,)
        self.pheno=pheno #(N-x or 2N-x, 1) because already filtered 
        self.covs=covs #(N-x or 2N-x, n_covs) 
        self.covariates_names = covariates_names # array(['traitspe mean', 'mean', 'cage_density'], dtype='<U13')
        if cageEffect or IEE or subset_on_cage:
            self.cage=cage #(N, 1)
        if maternalEffect:
            self.maternal=maternal #(N,1?)
        self.kinship=kinship # (N, N)
        self.env=env # (N, N)
        
        self.sampleID_all=sampleID_all # (N_a,)
        self.cage_all=cage_all # (N_a,)
        self.maternal_all=maternal_all # (N_a,)
        self.kinship_all=kinship_all # (N_a,N_a)
        self.env_all=env_all # (N_a,N_a)
       
        self.kinship_cross=kinship_cross # (N,N_a)
        self.env_cross=env_cross # (N,N_a)
       
        self.cage_density = cage_density # (N,1) in uni; (2N, 2) # in MT, this has 1st col [Nval+0s], 2nd col [0s+Nvals] because it had to be added to covs
        self.Iok_N = Iok_N # N
        self.Iok_pot2N = Iok_pot2N # N or 2N


    def VD(self, DGE, IGE, IEE, cageEffect,maternalEffect,vc_init, vc_init_type, SimplifNonIdableEnvs, bivariate):

        """ defines covariance for variance decomposition."""

        #1. define mean from observed phenotypic values
        #pdb.set_trace() #np.isnan(self.pheno).all() np.isnan(self.covs).all() should return False
        mean = lin_mean(self.pheno,self.covs) 

        print("mean.n_covs is " + str(mean.n_covs))
        
        # pheno and covs are 2N-x or N-x if there were NAs
        
        #2. DEFINE CAGEMATE ASSIGNMENT FOR ANALYSES INVOLVING IGE AND/OR IEE. 
        # Z is N focal x N_all and has 0s in cells Z_i,i (i.e. an animal is not its own cage mate)
        if IEE:
            #transform boolean to float. shape of same_cage is len(self.cage) x len(self.cage_all)
            #same_cage and diff_inds are (N_a, N_a) and Z is (N, N_a) and no necessary relationship between the two axes. Still Zi_i (wherever that is) should be 0 and Zi_j should be 1 if i and j are in the same cage
            same_cage = 1. * (self.cage==self.cage_all) # self.cage.shape = (N, 1); self.cage_all.shape = (N,); same_cage.shape = (N,N)
            diff_inds = 1. * (self.pheno_ID[:,np.newaxis]!=self.sampleID_all) # self.pheno_ID.shape = (N,); self.sampleID_all.shape = (N,); diff_inds.shape = (N, N)
            Z = same_cage * diff_inds 
            # To check if Z corresponds to (cage_density - 1): cage_density_minus1 = sum(Z); all( (self.cage_density[:int(self.cage_density.shape[0]/2),0] - cage_density_minus1) == 1)
            # in case pheno_ID have different order than sample_ID:  all( (np.sort(self.cage_density[:int(self.cage_density.shape[0]/2),0]) - np.sort(cage_density_minus1)) == 1) 
            #pdb.set_trace() # check before filtering with Iok
            Z = Z[self.Iok_N,:]  # is 0/1 ; Z.shape (N-x, N_a) - realN_focal X all 

        #3. DEFINE THE OVERALL GENETIC COVARIANCE MATRIX  #that would actually be for bivariate too
        # Filtering matrices to keep only focal, no missing pheno (or at least one pheno in MT); transforming kinship to dim (N-x, N-x) and kinship_cross to (N-x, N_a).
        self.kinship = self.kinship[self.Iok_N,:][:,self.Iok_N] 
        self.kinship_cross =  self.kinship_cross[self.Iok_N,:] 

        #3.1 Scaling 
        if DGE or IGE:
            #scales the DGE component of the sample to sample covariance matrix to have sample variance 1
            sf_K = covar_rescaling_factor(self.kinship) # self.kinship.shape = N-x,N-x
            self.kinship *= sf_K

            if IGE:
                #first IGE variance: ZKallZ' in this code (ZKZ' in paper)
                _ZKallZ = np.dot(Z,np.dot(self.kinship_all,Z.T)) 
                sf_ZKallZ = covar_rescaling_factor(_ZKallZ) 
                self.kinship_all *= sf_ZKallZ 
                #then DGE/IGE covariance:
                self.kinship_cross *= np.sqrt(sf_K * sf_ZKallZ)
        
        #3.2 Setting new Iok_cov for bivariate (fig 3):
        # if kinship is of dim N-x (where x is number of animals with neither phenotype/covs), then Iok passed to KronCov and DirIndirCovMT below needs to be length 2(N-x) 
        # with False where the animal only has one phenotype (animals with neither phenotype are already excluded,i.e. are False in Iok_N)
        if bivariate is True:
            self.Iok_cov =self.Iok_pot2N[np.append(self.Iok_N,self.Iok_N)] # dim self.Iok_cov : 2(N-x)

        #3.3 Getting cov matrix
        if DGE and not IGE: # DGE only
            if bivariate is False: #univar
                self._genoCov = FixedCov(self.kinship) # kinship is (N-x, N-x)
            else: #bivariate
                # TOREMOVE: check with Amelie, but this here doesn't make sense, we are in DGE only anyways
                #if IGE:
                #    sys.exit("bivariate with IGE only not considered; needs checking")
                #else:
                if vc_init_type == 'diagonal': #to initialize to corr_Ad1d2 = 0 
                    C = DiagonalCov(2) # 2 as this is for one genetic effect only
                else:
                    C = FreeFormCov(2) # 2 as this is for one genetic effect only
                self._genoCov = KronCov(C, self.kinship, Iok = self.Iok_cov) # self._genoCov.K().shape = 2(N-x)-y where y is the number of individual missing only one of the two pheno - same dim as pheno
                self.nb_params = 1 # TODO should check why 1 

        elif IGE and not DGE: # IGE only
            if bivariate is False:
                self._genoCov = FixedCov(_ZKallZ)
            else: #MT
                sys.exit("bivariate mode with IGE not available yet, when including IGE set 'bivariate' to False")
                #C = FreeFormCov(2)
                ##self._genoCov = KronCov(C, _ZKcmZ, Iok = Iok_cov) ### TOREMOVE / FOR AMELIE: _ZKcmZ is NOT DEFINED, is it the same as _ZKallZ - it seems so from doc "models for Indirect..."
                #self._genoCov = KronCov(C, _ZKallZ, Iok = self.Iok_cov)
                #self.nb_params = 1          
       
        elif DGE and IGE: # both
            if bivariate is False:
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
                sys.exit("bivariate mode with IGE not available yet, when including IGE set 'bivariate' to False")
                #if vc_init is not None:
                #    sys.exit("low rank not implemented yet for MT covariance")
                #else:
                #    self._genoCov = DirIndirCovMT(self.kinship,Z,kinship_all=self.kinship_all,kinship_cross=self.kinship_cross, Iok = self.Iok_cov)
                #    #self._genoCov.K().shape = 2(N-x)-y where y is the number of individual missing only one of the two pheno 
                #self.nb_params = 2 #should check why 2 and is it 2 no matter what the vc_init is?         
        else: # no GE
            self._genoCov = None
            if bivariate is False:
                self.nb_params = None
            else:
                self.nb_params = 0
                

        #4. DEFINE THE OVERALL ENVIRONMENTAL COVARIANCE MATRIX        
        # Filtering matrices to keep only focal, no missing pheno; transforming cage to dim (N-x,), env to (N-x, N-x) and env_cross to (N-x, N_a)
        if cageEffect:
            self.cage = self.cage[self.Iok_N,:] 
        if maternalEffect:
            self.maternal = self.maternal[self.Iok_N,:] 
        self.env= self.env[self.Iok_N,:][:,self.Iok_N]
        self.env_cross= self.env_cross[self.Iok_N,:]
            
        if SimplifNonIdableEnvs: #implies bivariate is False
            
        ## TODO: check sigmaRhoCov and go back to plosGen paper - implement bivar
            if bivariate is True:
                sys.exit("low rank not implemented yet for MT model")
            assert self.cage_all is not None, 'Cant run IGE analysis without cage info'
            assert IEE and cageEffect,  'No need to simplify environmental effects if IEE or cageEffects are omitted'
            assert len(np.unique(self.cage_density)) == 1, 'Not all groups of equal size - environmental effects might be identifiable'
            # TODO: do need to add some assert for maternalEffect here?
            N = self.cage.shape[0] # N defined based on cage dim and not pheno in order to work for both univar and bivar
            uCage = np.unique(self.cage) # since filtered, here self.cage has dim as self.pheno
            #W, the cage design matrix, is N-x or 2N-x X n_cages (where N-x is number of focal animals, i.e. no NA in pheno/covs; 2N-x, focal animals, i.e. at least one pheno and non-NA covs) 
            W = np.zeros((N,uCage.shape[0]))
            for cv_i, cv in enumerate(uCage): 
                W[:,cv_i] = 1.*(self.cage[:,0]==cv)
            #WWt, the cage effect covariance matrix, is (N-x) X (N-x) and has 1s in cells WWt_i,i (hence WWt it is different from ZZt, which have "number of mice in the cage - 1" in Z_i,i)
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
                if bivariate is False:
                    self._envCov = DirIndirCov(self.env,Z,kinship_all=self.env_all,kinship_cross=self.env_cross) 
                else:
                    sys.exit("bivariate mode with IGE not available yet, when including IGE set 'bivariate' to False")
                    #self._envCov = DirIndirCovMT(self.env,Z,kinship_all=self.env_all,kinship_cross=self.env_cross, Iok = self.Iok_cov)
                    ## self._envCov.K().shape is 2(N-x)-y, 2(N-x)-y
                    #self.nb_params = self.nb_params + 2 # 2 for DEE + IEE
            else:
                if bivariate is False:
                    self._envCov = FixedCov(self.env)
                else:
                    C = FreeFormCov(2)
                    self._envCov = KronCov(C, self.env, Iok = self.Iok_cov)
                    print('in def VD env part no IEE MT')
                    self.nb_params = self.nb_params + 1 #1 for DEE

            ##define cage effect covariance matrix

        ### TODO - NB: have some bias on environmental effects
            
            if cageEffect:
                assert self.cage_all is not None, 'Cant run IGE analysis without cage info'
                N = self.cage.shape[0] # N defined based on cage dim and not pheno in order to work for both univar and bivar
                uCage = np.unique(self.cage) # since filtered, here self.cage has dim as self.pheno
                #W, the cage design matrix, is N-x or 2N-x X n_cages (where N-x is number of focal animals, i.e. no NA in pheno/covs; 2N-x, focal animals, i.e. at least one pheno and non-NA covs) 
                W = np.zeros((N,uCage.shape[0]))
                for cv_i, cv in enumerate(uCage): 
                    W[:,cv_i] = 1.*(self.cage[:,0]==cv)
                #WWt, the cage effect covariance matrix, is (N-x) X (N-x) and has 1s in cells WWt_i,i (hence WWt it is different from ZZt, which have "number of mice in the cage - 1" in Z_i,i)
                WW = np.dot(W,W.T) 
                
                #this is equivalent to getting covar_rescaling_factor first and then multiplying, as done for other matrices above
                WW = covar_rescale(WW)

                #get covariance for cageEffect
                if bivariate is False:
                    self._cageCov = FixedCov(WW)
                else:
                    C = FreeFormCov(2)
                    self._cageCov = KronCov(C, WW, Iok = self.Iok_cov) # 2(N-x)-y , 2(N-x)-y
                    self.nb_params = self.nb_params + 1                 
            else:
                self._cageCov = None
                
            
        if maternalEffect:
            assert self.maternal_all is not None, 'Cant run maternal analysis without maternal info'
            N_mat = self.maternal.shape[0]
            uMaternal = np.unique(self.maternal)
            #W, the cage design matrix, is N x n_cages (where N is number of focal animals) 
            W_mat = np.zeros((N_mat,uMaternal.shape[0]))
            for cv_i, cv in enumerate(uMaternal):
                W_mat[:,cv_i] = 1.*(self.maternal[:,0]==cv)
            #WWt, the cage effect covariance matrix, is N x N and has 1s in cells WWt_i,i (hence WWt it is different from ZZt, which has "number of mice in the cage - 1" in Z_i,i)
            WW_mat = np.dot(W_mat,W_mat.T)
            
            #this is equivalent to getting covar_rescaling_factor first and then multiplying, as done for other matrices above
            WW_mat = covar_rescale(WW_mat)
            
            #get covariance for maternalEffect 
            if bivariate is False:
                self._maternalCov = FixedCov(WW_mat)
            else: #TODO: have to double check this for bivariate!!
                C = FreeFormCov(2)
                self._maternalCov = KronCov(C, WW_mat, Iok = self.Iok_cov) # 2(N-x)-y , 2(N-x)-y
                self.nb_params = self.nb_params + 1                 
        else:
            self._maternalCov = None


        #5. Define overall covariance matrix as sum of genetic, environmental and cage covariance matrices
        if self._genoCov is None:
            if self._cageCov is None and self._maternalCov is None:
                self.covar = SumCov(self._envCov)
                print('in def VD, setting:  - out: genoCov,cageCov,maternalCov')
            elif self._cageCov is None:
                self.covar = SumCov(self._envCov, self._maternalCov)
                print('in def VD, setting: maternalCov - out: genoCov,cageCov')
            elif self._maternalCov is None:
                self.covar = SumCov(self._envCov, self._cageCov)
                print('in def VD, setting: cageCov - out: genoCov,maternalCov')
            else:
                self.covar = SumCov(self._envCov, self._cageCov, self._maternalCov)
                print('in def VD, setting: cageCov,maternalCov - out: genoCov')

        else:
            if self._cageCov is None and self._maternalCov is None:
                self.covar = SumCov(self._genoCov,self._envCov)
                print('in def VD, setting: genoCov - out: cageCov,maternalCov')
            elif self._cageCov is None:
                self.covar = SumCov(self._genoCov,self._envCov, self._maternalCov)
                print('in def VD, setting: genoCov,maternalCov - out: cageCov')
            elif self._maternalCov is None:
                self.covar = SumCov(self._genoCov,self._envCov, self._cageCov)
                print('in def VD, setting: genoCov,cageCov - out: maternalCov')
            else:
                self.covar = SumCov(self._genoCov,self._envCov,self._cageCov, self._maternalCov) # ALL are 2(N-x)-y , 2(N-x)-y
                print('in def VD, setting: genoCov,cageCov,maternalCov')
        #pdb.set_trace() #np.isnan(self.covar.K()).all() should be False; check dimensions: self.covar.K().shape is N-x,N-x in univar; 2(N-x)-y, 2(N-x)-y in bivar

        ## define gp
        self._gp = GP(covar=self.covar,mean=mean)


    def optimize(self, vc_init, vc_init_type, bivariate): #, seed):
        """optimises the covariance matrix = estimate variance components"""
        #1. initiate covariance matrices, _genoCov,_envCov,_maternalCov,_cageCov depending on the vc_init_type
        if vc_init_type == 'lowrank' or vc_init_type == 'diagonal': # univar only
            if bivariate is False:
                print("optimising lowrank or diagonal univariate")
                #below is for univar
                #'genoCov_K': genoCov_K, 'envCov_K': envCov_K, 'maternalCov_K': maternalCov_K, 'cageCov_K': cageCov_K
                #as soon as set them approximation is made so don't expect them to be the same
                #self._genoCov.setCovariance(vc_init._genoCov.C.K()) #is setting C covariance as defined in dirIndirCov_v2.py; recalls what defined in self._genoCov.C, that is a limix_core.covar.diagonal.DiagonalCov
                self._genoCov.setCovariance(vc_init['genoCov_K']) #is setting C covariance as defined in dirIndirCov_v2.py; recalls what defined in self._genoCov.C, that is a limix_core.covar.diagonal.DiagonalCov
                #as soon as set them 10-4 added to diagonal so don't expect them to be the same #not relevant here?
                #self._envCov.setCovariance(vc_init._envCov.C.K()) #is setting C covariance as defined in dirIndirCov_v2.py
                self._envCov.setCovariance(vc_init['envCov_K']) #is setting C covariance as defined in dirIndirCov_v2.py
                if self._maternalCov is not None: # Helene added this on 28/05/25 - otherwise could give problems if maternal not there
                    #self._maternalCov.scale =  vc_init._maternalCov.scale #Amelie added line for maternal on 26/04/2024
                    self._maternalCov.scale =  vc_init['maternalCov_K']
                if self._cageCov is not None: # Helene added this on 28/05/25 - otherwise could give problems if cage not there
                    #self._cageCov.scale =  vc_init._cageCov.scale
                    self._cageCov.scale =  vc_init['cageCov_K']
            else: #bivariate
                if vc_init_type == 'lowrank': 
                    sys.exit("lowrank vc_init_type not implemented yet for MT model")
                else:
                    print("optimising diagonal bivariate")
                    #as soon as set them approximation is made so don't expect them to be the same
                    #pdb.set_trace() #there was ISSUE HERE warning given only first time around but applies to cage etc as well. params not set as not implemented. BUG
                    # here self._genoCov and co. is a limix_core.covar.kronecker.KronCov that has no attribute setCovariance(); 
                    # while self._genoCov.C is a limix_core.covar.freeform.FreeFormCov that has setCovariance()
                    #self._genoCov.C.setCovariance(vc_init._genoCov.C.K()) # self._genoCov = limix_core.covar.kronecker.KronCov; self._genoCov.C = limix_core.covar.freeform.FreeFormCov
                    self._genoCov.C.setCovariance(vc_init['genoCov_K']) # self._genoCov = limix_core.covar.kronecker.KronCov; self._genoCov.C = limix_core.covar.freeform.FreeFormCov
                    #as soon as set them 10-4 added to diagonal so don't expect them to be the same #not relevant here?
                    #self._envCov.C.setCovariance(vc_init._envCov.C.K()) # as self._genoCov
                    self._envCov.C.setCovariance(vc_init['envCov_K']) # as self._genoCov
                    if self._maternalCov is not None: 
                        #self._maternalCov.C.setCovariance(vc_init._maternalCov.C.K()) # as self._genoCov #Amelie added line for maternal on 26/04/2024 
                        self._maternalCov.C.setCovariance(vc_init['maternalCov_K']) # as self._genoCov #Amelie added line for maternal on 26/04/2024 
                    if self._cageCov is not None:
                        #self._cageCov.C.setCovariance(vc_init._cageCov.C.K()) # as self._genoCov
                        self._cageCov.C.setCovariance(vc_init['cageCov_K']) # as self._genoCov

        elif vc_init_type == 'random_diagonal': # univar only ####VIGILA should check why only geno and env set when random diag
            if bivariate is True: # TODO: implement for bivar
                sys.exit("random diagonal vc_init_type not implemented yet for MT model")
            
            print("optimising random_diagonal univariate")
            #below is for univar
            
            self._genoCov.setCovariance(self.vc1_init.C.K()) #is setting C covariance as defined in dirIndirCov_v2.py
            #as soon as set them 10-4 added to diagonal so don't expect them to be the same #not sure if relevant here. since vc1 conforms to approximation maybe it is not changed
            self._envCov.setCovariance(self.vc1_init.C.K()) #is setting C covariance as defined in dirIndirCov_v2.py
            self._maternalCov.setRandomParams() #Amelie added line for maternal on 26/04/2024
            self._cageCov.setRandomParams()
        
        else: 
            if bivariate is True: #ie MT
                print("we're in MT mode. self.nb_params (out of DGE IGE DEE IEE cageEffect maternalEffect) is " + str(self.nb_params))

                #print("corrs initialised to 1./self.nb_params I think")
        #       if 'commons' in self.task or 'sexspe' in self.task: #for : initialise all correlations to 0
                #init_cov_g = (1./self.nb_params) *np.kron(np.eye(2), np.ones((2,2))) #blocks of size 2
                #init_cov_g = init_cov_g + 1e-4*np.ones(init_cov_g.shape) + np.eye(4)*1e-4
                #init_cov_e = (1./self.nb_params) *np.kron(np.eye(2), np.ones((2,2))) 
                #init_cov_e = init_cov_e + 1e-4*np.ones(init_cov_e.shape) + np.eye(4)*1e-4
                #init_cov_c = (1./self.nb_params) *np.ones((2,2)) + np.eye(2)*1e-4

        #       else:
        #       for model with 2 different traits: initialise all correlations to 0
                print("corrs initialised to 0 by default:")
                init_cov_g = (1./self.nb_params) *np.kron(np.eye(2), np.eye(2)) #diag
                init_cov_g = init_cov_g + 1e-4*np.ones(init_cov_g.shape) + np.eye(4)*1e-4
                init_cov_e = (1./self.nb_params) *np.kron(np.eye(2), np.eye(2))
                init_cov_e = init_cov_e + 1e-4*np.ones(init_cov_e.shape) + np.eye(4)*1e-4
                init_cov_c = (1./self.nb_params) *np.eye(2)
                init_cov_c = init_cov_c + 1e-4*np.ones(init_cov_c.shape) + np.eye(2)*1e-4
                init_cov_m = (1./self.nb_params) *np.eye(2)
                init_cov_m = init_cov_m + 1e-4*np.ones(init_cov_m.shape) + np.eye(2)*1e-4
                
                if self._genoCov is not None:
                    self._genoCov.C.setCovariance(init_cov_g) #if init_cov_g was 2 blocks of 2 only first block gets used if DGE only - clever
                    print("genoCov: \n", init_cov_g)
                self._envCov.C.setCovariance(init_cov_e) #never None
                print("envCov: \n", init_cov_e)
                if self._cageCov is not None:
                    self._cageCov.C.setCovariance(init_cov_c)
                    print("cageCov: \n", self._cageCov.C.K())
                if self._maternalCov is not None:
                    self._maternalCov.C.setCovariance(init_cov_m)
                    print("maternalCov: \n", self._maternalCov.C.K())
                
            else:
                print("we're in univariate mode. all params are initialised at random")

                self._gp.covar.setRandomParams()

                if isinstance(self._genoCov,DirIndirCov): #only in univar mode; will go through when DGE and IGE
                    init_corr = self._genoCov.C.K()[0,1]/np.sqrt(self._genoCov.C.K()[0,0]*self._genoCov.C.K()[1,1])
                    print('corr_Ads init is ' + str(init_corr))

        #2. optimization - keep calc_ste = False as we don't want the STE for the variance components: we want them for the proportions of phenotypic variance explained by xx
        #pdb.set_trace()
        print("about to optimize")
        #self.conv, self.info = self._gp.optimize(calc_ste = False) # before 09/19/23, without catching for optim error 
        #print("done optimizing") #
        try: # try to optimize
            self.conv, self.info = self._gp.optimize(calc_ste = False)
        # exceptions:
        #1. LinAlgError: if can't because of non-positive definite matrix - depends on initialization - want to try again
        #2. ValueError: if can't because infinite numbers - depends on initialization - want to try again
        except np.linalg.LinAlgError as LinAlgError: 
            #saving error only when corresponding to non-positive array - otherwise error raise
            if bool(re.search("\d-th leading minor of the array is not positive definite", str(LinAlgError))) is not True:
                raise LinAlgError
            else:
                print("Optimization was unsuccessful with error:\n\tLinAlgError('",LinAlgError,"')")
                self.optim_error = LinAlgError # Saving the type of error that is raised
                
        except ValueError as ValError:
            #saving error only when corresponding to non-positive array - otherwise error raise
            if bool(re.search("array must not contain infs or NaNs", str(ValError))) is not True:
                raise ValError
            else:
                print("Optimization was unsuccessful with error:\n\tValueError('",ValError,"')")
                self.optim_error = ValError
        else:
            self.optim_error = None
            print("Could optimize successfully")


    def build_output(self, calc_ste, DGE, IGE, IEE, cageEffect, maternalEffect, SimplifNonIdableEnvs, bivariate):
        """function to access estimated model specifics, variance components, standard errors"""
        
        ##### 1. Model specifics #####
        R = {}
        #whether the run converged
        R['conv'] = self.conv
        #should be small (e.g. < 10^-4)
        R['grad'] = self.info['grad']
        #Contrary to what it says, this is -LML
        R['LML']  = self._gp.LML()
        
        #number of focal animals
        R['sample_size1'] = self.focal_sample_size1
        R['sample_size2'] = self.focal_sample_size2 # None if univariate
        R['union_focal'] = self.focal_union # None if univariate
        R['inter_focal'] = self.focal_inter # None if univariate
        
        #number of _cm animals
        R['sample_size1_cm'] = self.real_sample_size_all1
        R['sample_size2_cm'] = self.real_sample_size_all2 # None if univariate
        R['union_cm'] = self.cm_union # None if univariate
        R['inter_cm'] = self.cm_inter # None if univariate
        
        #effect sizes of fixed effects in the model
        R['covariates_names'] = self.covariates_names
        R['covs_betas'] = self._gp.mean.b[:,0]


        ##### 2. Variance components and STEs #####
        #----- A. Getting Interpretable parameters - i.e. original VCs ------# 
        # Depending on the type of object of Cov, we retrieve the interpretable parameters in different ways:
        if self._genoCov is None: 
            genIntP = []
        elif type(self._genoCov) is FixedCov:
            genIntP = np.array([self._genoCov.scale])
        else: 
            genIntP = self._genoCov.getInterParams() #works for KronCov and DirIndirCov
        
        # if envCov is None: not possible, there's always some DEE noise!
        if type(self._envCov) is FixedCov: # that's when there's only DEE
            envIntP = np.array([self._envCov.scale])
        elif type(self._envCov) is SigmaRhoCov: # self._envCov is SigmaRhoCov when SimplifNonIdableEnvs is True - can retrieve sigma_sq with .scale and corr with .rho
            assert SimplifNonIdableEnvs, "SimplifNonIdableEnvs, check if everything with the environemnt is correct"
            envIntP = np.array([self._envCov.rho, self._envCov.scale]) # At this point have 2 intP in here: corr and var; var has to be the last
        else:
            envIntP = self._envCov.getInterParams() #that's when there is DEE and IEE or bivariate mode (KronCov)
        
        if maternalEffect:
            if type(self._maternalCov) is FixedCov:
                MatnIntP = np.array([self._maternalCov.scale])
            else:
                MatnIntP = self._maternalCov.getInterParams() #works for KronCov and DirIndirCov
        else:
            MatnIntP = []
            
        if cageEffect and not SimplifNonIdableEnvs: # if SimplifNonIdableEnvs is True, there's no cageCov! 
            if type(self._cageCov) is FixedCov:
                cageIntP = np.array([self._cageCov.scale])
            else:
                cageIntP = self._cageCov.getInterParams() #works for KronCov and DirIndirCov
        else:
            cageIntP = []
        
        # Defining the number of params for each effect - important to index them
        # Need to do add this to 'self', so that can retrieve them in the 'transformParams' without having to pass them as arguments 
        self.numP = {}
        self.numP['gen'] = len(genIntP)
        self.numP['env'] = len(envIntP)
        self.numP['matn'] = len(MatnIntP)
        self.numP['cage'] = len(cageIntP)
        
        # Concatenating Interpretable params in a single vector
        aP = np.concatenate([genIntP, envIntP, MatnIntP, cageIntP])


        #----- B. Get proportional variances and correlations, total genetic var and STE ------# 
        # 1. Getting params as proportional variances and correlations
        params = self.get_VCs(aP, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate)
        
        # 2. Getting total genetic variance
        if bivariate is False:
            gCov2 = (-999.)
            if self.numP['gen'] == 0:
                gCov1 = (-999.)
            else:
                gCov1 = 1/covar_rescaling_factor(self._genoCov.K())
        else:
            # Something is wrong with the number of samples in self._genoCov.K().shape and self._genoCov._K.shape # is this still the case - Helene 28/05/24
            if self.numP['gen'] ==0:
                gCov1 = (-999.)
                gCov2 = (-999.)
            else:
                N = self.focal_sample_size1 # number of focals in phenotype 1 # this has been calculated above
                assert self._genoCov.K().shape[0] == self.focal_sample_size1 + self.focal_sample_size2, "cannot extract correctly the genotype covariance for trait1 and trait2" 
                gCov1 = 1/covar_rescaling_factor(self._genoCov.K()[:N][:,:N]) # selecting upper left of the matrix - phenotype 1
                gCov2 = 1/covar_rescaling_factor(self._genoCov.K()[N:][:,N:]) # the rest is phenotype 2
                
        # 3. Getting STE and correlations between some params estimates, if want them 
        STEs = {}
        if calc_ste:
            assert not SimplifNonIdableEnvs, sys.exit("can't calculate STE of proportions of var explained with SimplifNonIdableEnvs") #really           
            STEs = self.get_STEs(aP, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate)
        else: # if no calc_ste true, returning "empty" matrices 
            STEs['GenSTE'] = np.full((4, 4), (-999.))
            STEs['EnvSTE'] = np.full((4, 4), (-999.)) 
            STEs['maternalSTE'] = np.full((2, 2), (-999.)) 
            STEs['cageSTE'] = np.full((2, 2), (-999.))
            STEs['tot_vSTE'] = np.array([(-999.), (-999.)])
            STEs['CorParams'] = np.full((8), (-999.))
            

        #----- B. Populating output ------# 
        # Genetic 
        R['prop_Ad1'] = params['GenP'][0,0]
        R['prop_As1'] = params['GenP'][2,2]
        R['prop_Ad2'] = params['GenP'][1,1]
        R['prop_As2'] = params['GenP'][3,3]

        R['corr_Ad1d2'] = params['GenP'][0,1]
        R['corr_Ad1s1'] = params['GenP'][0,2]
        R['corr_Ad2s1'] = params['GenP'][1,2]
        R['corr_Ad1s2'] = params['GenP'][0,3]
        R['corr_Ad2s2'] = params['GenP'][1,3]
        R['corr_As1s2'] = params['GenP'][2,3] 

        # Env
        R['prop_Ed1'] = params['EnvP'][0,0]
        R['prop_Es1'] = params['EnvP'][2,2]
        R['prop_Ed2'] = params['EnvP'][1,1]
        R['prop_Es2'] = params['EnvP'][3,3]

        R['corr_Ed1d2'] = params['EnvP'][0,1]
        R['corr_Ed1s1'] = params['EnvP'][0,2]
        R['corr_Ed2s1'] = params['EnvP'][1,2]
        R['corr_Ed1s2'] = params['EnvP'][0,3]
        R['corr_Ed2s2'] = params['EnvP'][1,3]
        R['corr_Es1s2'] = params['EnvP'][2,3]
        
        # maternal - dam
        R['prop_Dm1'] = params['maternalP'][0,0]
        R['corr_Dm1Dm2'] = params['maternalP'][0,1]
        R['prop_Dm2'] = params['maternalP'][1,1]

        # cage
        R['prop_C1'] = params['cageP'][0,0]
        R['corr_C1C2'] = params['cageP'][0,1]
        R['prop_C2'] = params['cageP'][1,1]
        
        # Total gen var
        R['tot_genVar1'] = gCov1
        R['tot_genVar2'] = gCov2
        
        R['total_var1'] = params['tot_var'][0]
        R['total_var2'] = params['tot_var'][1]

        # STE gen
        R['STE_Ad1'] = STEs['GenSTE'][0,0]
        R['STE_As1'] = STEs['GenSTE'][2,2]
        R['STE_Ad2'] = STEs['GenSTE'][1,1]
        R['STE_As2'] = STEs['GenSTE'][3,3]
        
        R['STE_Ad1d2'] = STEs['GenSTE'][0,1]
        R['STE_Ad1s1'] = STEs['GenSTE'][0,2]
        R['STE_Ad2s1'] = STEs['GenSTE'][1,2]
        R['STE_Ad1s2'] = STEs['GenSTE'][0,3]
        R['STE_Ad2s2'] = STEs['GenSTE'][1,3]
        R['STE_As1s2'] = STEs['GenSTE'][2,3]
        
        # STE env
        R['STE_Ed1'] = STEs['EnvSTE'][0,0]
        R['STE_Es1'] = STEs['EnvSTE'][2,2]
        R['STE_Ed2'] = STEs['EnvSTE'][1,1]
        R['STE_Es2'] = STEs['EnvSTE'][3,3]
        
        R['STE_Ed1d2'] = STEs['EnvSTE'][0,1]
        R['STE_Ed1s1'] = STEs['EnvSTE'][0,2]
        R['STE_Ed2s1'] = STEs['EnvSTE'][1,2]
        R['STE_Ed1s2'] = STEs['EnvSTE'][0,3]
        R['STE_Ed2s2'] = STEs['EnvSTE'][1,3]
        R['STE_Es1s2'] = STEs['EnvSTE'][2,3]
        
        # STE Maternal
        R['STE_Dm1'] = STEs['maternalSTE'][0,0]
        R['STE_Dm1Dm2'] = STEs['maternalSTE'][0,1]
        R['STE_Dm2'] = STEs['maternalSTE'][1,1]

        # STE cage # can't have STE for C1/C2
        R['STE_C1C2'] = STEs['cageSTE'][0,1]
        
        # STE totv - have to check if this is true
        R['STE_totv1'] = STEs['tot_vSTE'][0]
        R['STE_totv2'] = STEs['tot_vSTE'][1]
        
        # cor_params # corP_Ad1_As1, corP_Ed1_Es1, corP_Ed1_Dm1, corP_Es1_Dm1, corP_Ad2_As2, corP_Ed2_Es2, corP_Ed2_Dm2, corP_Es2_Dm2 
        R['corParams_Ad1_As1'] = STEs['CorParams'][0]
        R['corParams_Ed1_Es1'] = STEs['CorParams'][1]
        R['corParams_Ed1_Dm1'] = STEs['CorParams'][2]
        R['corParams_Es1_Dm1'] = STEs['CorParams'][3]
        
        R['corParams_Ad2_As2'] = STEs['CorParams'][4]
        R['corParams_Ed2_Es2'] = STEs['CorParams'][5]
        R['corParams_Ed2_Dm2'] = STEs['CorParams'][6]
        R['corParams_Es2_Dm2'] = STEs['CorParams'][7]

        R['time_exec'] = time.time() - start_time
    
        return R
      
      
    def get_VCs(self, aP, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate):
        """Function to get proportional variances and correlations from original VCs"""
        # 0. retrieving numGenP, numEnvP, numMatnP, numCageP for indexing and definition of number of effects
        #numGenP = self.numP['gen'] # This is not used here
        numEnvP = self.numP['env']
        numMatnP = self.numP['matn']
        numCageP = self.numP['cage']
        
        # 0b. 'transformParams' is the function to transform parameters from original variance components to prop variances and corrs, 
        #      and it is different between univariate and bivariate
        if bivariate is False:
            transformParams = self.transformParams_uni #transformParams here is a function, not a variable
        else:
            transformParams = self.transformParams_bi
          
        # A. Transform parameters to tensor and then new params
        aP_ = torch.tensor(aP, requires_grad=True, dtype=torch.float32) # 7 uni, 23 bi
        aPnew = transformParams(aP_).data.numpy().astype(np.float64) # 7 uni
        #if bivariate is True: #TOREMOVE
        #    #pdb.set_trace()
        #    aPnew_new = self.transformParams_bi_new(aP_).data.numpy().astype(np.float64) # 7 uni 
        #    assert (aPnew_new == aPnew).all(), "problem with the new transformParams_bi_new function"

        # B. Store new params for output 
        # From an array, storing in matrices - this is to make congruent univariate and bivariate
        GenP, EnvP, maternalP, cageP = self.paramsInMx(aPnew, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate) # out = [GenM, EnvM, DamM, CageM]
        
        # C. Since the last one is always vtot
        # we have to store the proportional for the last non-genetic component
        # calculate it as original_VC / vtot
        if bivariate is False:
            tot_v1 = aPnew[-1]
            tot_v2 = (-999)
            
            if numCageP == 1:
                assert cageP[0,0] == aPnew[-1], "vtot stored in Cage doesn't correspond to last interp param new (aPnew[-1])"
                cageP[0,0] = aP[-1] / aPnew[-1] # If cage effect, cage is the last 
                
            elif numCageP == 0 and numMatnP ==1:
                assert maternalP[0,0] == aPnew[-1], "vtot stored in Maternal doesn't correspond to last interp param new (aPnew[-1])"
                maternalP[0,0] = aP[-1] / aPnew[-1] # If no cage effects, and yes maternal, dam is the last
            
            elif numCageP == 0 and numMatnP == 0:
                if numEnvP ==1 or numEnvP == 2: # when SimplifNonIdableEnvs = True
                    assert EnvP[0,0] == aPnew[-1], "vtot stored in Env doesn't correspond to last interp param new (aPnew[-1])"
                    EnvP[0,0] = aP[-1] / aPnew[-1] # if no cage, no mat, and no IEE, DEE is the last - that is a general residual term
                else:
                    assert EnvP[2,2] == aPnew[-1], "vtot stored in Env doesn't correspond to last interp param new (aPnew[-1])"
                    EnvP[2,2] = aP[-1] / aPnew[-1] # Es1
        
        else: 
            # In Bivariate mode, last and 2-before-last (or 4-before-last if DEE and IEE) are always tot_v2 and tot_v1 respectively
            tot_v1 = aPnew[-3]
            tot_v2 = aPnew[-1] 
            
            if numCageP == 3: 
                assert cageP[0,0] == aPnew[-3] and cageP[1,1] == aPnew[-1], "vtot stored in Cage doesn't correspond to last interp param new"
                cageP[0,0] = aP[-3] / aPnew[-3] #C1 
                cageP[1,1] = aP[-1] / aPnew[-1] #C2
                
            elif numCageP == 0 and numMatnP ==3:
                assert maternalP[0,0] == aPnew[-3] and maternalP[1,1] == aPnew[-1], "vtot stored in Maternal doesn't correspond to last interp param new"
                maternalP[0,0] = aP[-3] / aPnew[-3] #Dm1 
                maternalP[1,1] = aP[-1] / aPnew[-1] #Dm2
            
            elif numCageP == 0 and numMatnP == 0:
                if numEnvP == 3: # Only one Env effect - only Ed1 and Ed2
                    assert EnvP[0,0] == aPnew[-3] and EnvP[1,1] == aPnew[-1], "vtot stored in Env doesn't correspond to last interp param new"
                    EnvP[0,0] = aP[-3] / aPnew[-3] # Ed1
                    EnvP[1,1] = aP[-1] / aPnew[-1] # Ed2
                else:
                    assert EnvP[2,2] == aPnew[-5] and EnvP[3,3] == aPnew[-1], "vtot stored in Env doesn't correspond to last interp param new"
                    EnvP[2,2] = aP[-5] / aPnew[-5] # Es1
                    tot_v1 = aPnew[-5]
                    EnvP[3,3] = aP[-1] / aPnew[-1] # Es2

        # D. Populating pars output with matrices and tot_Vs
        pars = {}
        pars['GenP']= GenP # 4x4 diag = prop_var_sq ; off-diag = corrs
        pars['EnvP']= EnvP # 4x4 diag = prop_var_sq ; off-diag = corrs
        pars['maternalP']= maternalP # 2x2 diag = prop_var_sq ; off-diag = corrs
        pars['cageP']= cageP # 2x2 diag = prop_var_sq ; off-diag = corrs
        pars['tot_var'] = np.array([tot_v1, tot_v2])
        
        return pars


    def get_STEs(self, aP, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate):
        """Function to get STE on proportional variances and correlations"""
        
        # 0. retrieving numGenP, numEnvP, numMatnP, numCageP for indexing and definition of number of effects
        numGenP = self.numP['gen'] 
        numEnvP = self.numP['env']
        numMatnP = self.numP['matn']
        numCageP = self.numP['cage']
        
        # 'transformParams' is the function to transform parameters from original variance components to prop variances and corrs, 
        # and it is different from univariate to bivariate
        if bivariate is False:
            transformParams = self.transformParams_uni
        else:
            transformParams = self.transformParams_bi

        # 1. Transform aP_ to tensor 
        aP_ = torch.tensor(aP, requires_grad=True, dtype=torch.float32) # 7 uni, 23 bi
        #aPnew = transformParams(aP_).data.numpy().astype(np.float64) # 7 uni # this is not essential..., used only if want to print later
        
        # 2. Compute C and transform to Cnew through Jacobean, using transforming function
        F = self._gp.covar.getFisherInf()
        C = la.pinv(F)
        J = torch.autograd.functional.jacobian(transformParams, aP_).data.numpy().astype(np.float64)
        #if bivariate is True: #TOREMOVE
        #    J_new = torch.autograd.functional.jacobian(self.transformParams_bi_new, aP_).data.numpy().astype(np.float64)
        #    assert (J_new == J).all(), "problem with the new transformParams_bi_new function"
        Cnew = np.dot(np.dot(J,C), J.T)
        
        # 3. Retrieving STE on new parameters from the diagonal of the Cnew 
        aPnew_se = np.sqrt(Cnew.diagonal())
        #for i in range(aPnew.shape[0]):
        #    print("param[",i,"]: ",aPnew[i], ' +/- ', aPnew_se[i], sep="")
            
        # 4. Preparing output with ste
        # From an array, storing in matrices - this is to make congruent univariate and bivariate
        GenSTE, EnvSTE, maternalSTE, cageSTE = self.paramsInMx(aPnew_se, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate) # out = [GenM, EnvM, DamM, CageM]
        
        # 5. Since the last one is always for vtot
        #    we have to store the STE vtot somewhere and null, -999.,  for the last non-genetic component
        #    if no cage effects, STE for totv1 and totv2 are in maternal; if no maternal, they are in env, either IEE if with IGE or DEE if no IGE
        if bivariate is False:
            tot_v1STE = aPnew_se[-1]
            tot_v2STE = (-999.)
            
            #pdb.set_trace()
            if numCageP == 1:
                assert cageSTE[0,0] == aPnew_se[-1], "vtot_STE stored in Cage doesn't correspond to last interp param new (aPnew_se[-1])"
                cageSTE[0,0] = (-999.) # If cage effect, cage is the last 
            
            elif numCageP == 0 and numMatnP ==1:
                assert maternalSTE[0,0] == aPnew_se[-1], "vtot_STE stored in Maternal doesn't correspond to last interp param new (aPnew_se[-1])"
                maternalSTE[0,0] = (-999.) # If no cage effects, and yes maternal, dam is the last
            
            elif numCageP == 0 and numMatnP ==0:
                if numEnvP ==1 or numEnvP ==2:
                    assert EnvSTE[0,0] == aPnew_se[-1], "vtot_STE stored in Env doesn't correspond to last interp param new (aPnew_se[-1])"
                    EnvSTE[0,0] = (-999.) # if no cage, no mat, and no IEE, DEE is the last - that is a general residual term
                else:
                    assert EnvSTE[2,2] == aPnew_se[-1], "vtot_STE stored in Env doesn't correspond to last interp param new (aPnew_se[-1])"
                    EnvSTE[2,2] = (-999.) # Es1
        else: 
            # In Bivariate mode, last and 2-before-last (or 4-before-last if DEE and IEE) are always tot_v2 and tot_v1 respectively
            tot_v1STE = aPnew_se[-3]
            tot_v2STE = aPnew_se[-1]
            
            if numCageP == 3: 
                assert cageSTE[0,0] == aPnew_se[-3] and cageSTE[1,1] == aPnew_se[-1], "vtot_STE stored in Cage doesn't correspond to last STE interp param new"
                cageSTE[0,0] = (-999.) #C1 
                cageSTE[1,1] = (-999.) #C2
            
            elif numCageP == 0 and numMatnP ==3:
                assert maternalSTE[0,0] == aPnew_se[-3] and maternalSTE[1,1] == aPnew_se[-1], "vtot_STE stored in Maternal doesn't correspond to last STE interp param new"
                maternalSTE[0,0] = (-999.) #Dm1 
                maternalSTE[1,1] = (-999.) #Dm2
            
            elif numCageP == 0 and numMatnP == 0:
                if numEnvP == 3: # Only one Env effect - only Ed1 and Ed2
                    assert EnvSTE[0,0] == aPnew_se[-3] and EnvSTE[1,1] == aPnew_se[-1], "vtot_STE stored in Env doesn't correspond to last STE interp param new"
                    EnvSTE[0,0] = (-999.) # Ed1
                    EnvSTE[1,1] = (-999.) # Ed2
                else:
                    #assert EnvSTE[2,2] == aPnew_se[-3] and EnvSTE[3,3] == aPnew_se[-1], "vtot_STE stored in Env doesn't correspond to last STE interp param new"
                    assert EnvSTE[2,2] == aPnew_se[-5] and EnvSTE[3,3] == aPnew_se[-1], "vtot_STE stored in Env doesn't correspond to last STE interp param new"
                    EnvSTE[2,2] = (-999.) # Es1
                    tot_v1STE = aPnew_se[-5]
                    EnvSTE[3,3] = (-999.) # Es2
        
        # 6. Preparing output with correlations between parameters
        CorParams = np.full((8), (-999.))  # corP_Ad1_As1, corP_Ed1_Es1, corP_Ed1_Dm1, corP_Es1_Dm1, corP_Ad2_As2, corP_Ed2_Es2, corP_Ed2_Dm2, corP_Es2_Dm2  # !!can't have cor_params with C
        
        if bivariate is False:
            if numGenP > 1:
                CorParams[0] = Cnew[0,2]/(np.sqrt(Cnew[0,0])*np.sqrt(Cnew[2,2])) # corP_Ad1_As1
            else:
                print('no correlation between gen params as there is ', numGenP)
                
            # Env eff and maternal effect
            if numCageP ==1:    
                if numEnvP ==3: 
                    CorParams[1] = Cnew[numGenP+0, numGenP+2]/(np.sqrt(Cnew[numGenP+0, numGenP+0])*np.sqrt(Cnew[numGenP+2, numGenP+2])) # corP_Ed1_Es1
                if numMatnP == 1: # indexes are relative to number of gen and env parameters
                    CorParams[2] = Cnew[numGenP+0, numGenP+numEnvP+0]/(np.sqrt(Cnew[numGenP+0, numGenP+0])*np.sqrt(Cnew[numGenP+numEnvP+0, numGenP+numEnvP+0])) # corP_Ed1_Dm1
                    if numEnvP ==3: 
                        CorParams[3] = Cnew[numGenP+0, numGenP+numEnvP+0]/(np.sqrt(Cnew[numGenP+2, numGenP+2])*np.sqrt(Cnew[numGenP+numEnvP+0, numGenP+numEnvP+0])) # corP_Es1_Dm1
            else:
                if numMatnP == 1 and numEnvP ==3: # indexes are relative to number of gen and env parameters
                    CorParams[1] = Cnew[numGenP+0, numGenP+2]/(np.sqrt(Cnew[numGenP+0, numGenP+0])*np.sqrt(Cnew[numGenP+2, numGenP+2])) # corP_Ed1_Es1
                    
        else: # For now getting only cor between params intra phenotypes, not inter - should we get inter as well?
            if numGenP > 3: 
                CorParams[0] = Cnew[0,5]/(np.sqrt(Cnew[0,0])*np.sqrt(Cnew[5,5])) # corP_Ad1_As1 #Ad1 = v[0] # As1= v[5]
                CorParams[4] = Cnew[2,9]/(np.sqrt(Cnew[2,2])*np.sqrt(Cnew[9,9])) # corP_Ad2_As2 #Ad2 = v[2] # As2= v[9]
                
            if numCageP == 3: # There is cage effect so I can calculate cor_params between the others
                if numEnvP > 3:
                    CorParams[1] = Cnew[numGenP+0,numGenP+5]/(np.sqrt(Cnew[numGenP+0,numGenP+0])*np.sqrt(Cnew[numGenP+5,numGenP+5])) # corP_Ed1_Es1
                    CorParams[5] = Cnew[numGenP+2,numGenP+9]/(np.sqrt(Cnew[numGenP+2,numGenP+2])*np.sqrt(Cnew[numGenP+9,numGenP+9])) # corP_Ed2_Es2
                if numMatnP == 3: # There is maternal effect 
                    CorParams[2] = Cnew[numGenP+0,numGenP+numEnvP+0]/(np.sqrt(Cnew[numGenP+0,numGenP+0])*np.sqrt(Cnew[numGenP+numEnvP+0,numGenP+numEnvP+0])) # corP_Ed1_Dm1
                    CorParams[6] = Cnew[numGenP+2,numGenP+numEnvP+0]/(np.sqrt(Cnew[numGenP+2,numGenP+2])*np.sqrt(Cnew[numGenP+numEnvP+0,numGenP+numEnvP+0])) # corP_Ed2_Dm2
                    if numEnvP > 3:
                        CorParams[3] = Cnew[numGenP+5,numGenP+numEnvP+0]/(np.sqrt(Cnew[numGenP+5,numGenP+5])*np.sqrt(Cnew[numGenP+numEnvP+0,numGenP+numEnvP+0])) # corP_Es1_Dm1
                        CorParams[7] = Cnew[numGenP+9,numGenP+numEnvP+0]/(np.sqrt(Cnew[numGenP+9,numGenP+9])*np.sqrt(Cnew[numGenP+numEnvP+0,numGenP+numEnvP+0])) # corP_Es2_Dm2
            else: # no cage effect - cannot have cor with maternal
                if numMatnP == 3 and numEnvP > 3: # There is maternal effect but no cage effect
                    CorParams[1] = Cnew[numGenP+0,numGenP+5]/(np.sqrt(Cnew[numGenP+0,numGenP+0])*np.sqrt(Cnew[numGenP+5,numGenP+5])) # corP_Ed1_Es1
                    CorParams[5] = Cnew[numGenP+2,numGenP+9]/(np.sqrt(Cnew[numGenP+2,numGenP+2])*np.sqrt(Cnew[numGenP+9,numGenP+9])) # corP_Ed2_Es2
                        
        # 7. Populating STE output with matrices
        ste = {}
        ste['GenSTE']= GenSTE # 4x4 diag = STE(prop_var_sq) ; off-diag = STE(corrs)
        ste['EnvSTE']= EnvSTE # 4x4 diag = STE(prop_var_sq) ; off-diag = STE(corrs)
        ste['maternalSTE']= maternalSTE # 2x2 diag = STE(prop_var_sq) ;  off-diag = STE(corr) 
        ste['cageSTE']= cageSTE # 2x2 diag = STE(tot_v) - have to check with Paolo if this is true ; off-diag = STE(corr_cage) 
        ste['tot_vSTE']= np.array([tot_v1STE, tot_v2STE])  
        ste['CorParams'] = CorParams # length 8

        return ste


    def transformParams_uni(self, x):
        # Transforms original_VCs into proportional variances and corrs
        #           x: v1gen, v12gen, v2gen, v1env, v12env, v2env, vcage 
        #     -> f(x): prop_v1gen, corr12gen, prop_v2gen, prop_v1env, corr12env, prop_v2env, **vtot**

        # NB: this actaully changes depending on which between DGE, IGE (IEE), maternalEffect, cageEffect are included ### 
        #     especially for the non-genetic components as the last element always needs to be vtot!
        
        # This function needs to take only ONE argument: **aP**
        #                       and output ONE object:   **aPnew** -> needed to calculate the STE with torch.autograd
        # Everything else needs to be defined already, e.g. genoCov-envCov ... and numGenP... or defined inside the function
        
        numGenP = self.numP['gen']
        numEnvP = self.numP['env']
        numMatnP = self.numP['matn']
        numCageP = self.numP['cage']
      
        # 1. total cov for each group of effects
        #   gCov
        if numGenP == 0:
            gCov = None
        elif numGenP == 1:
            gCov  = x[0] * torch.tensor(self._genoCov.K0, dtype=torch.float32)
        else:
            gCov  = x[0] * torch.tensor(self._genoCov._K, dtype=torch.float32)
            gCov += x[1] * torch.tensor(self._genoCov._KZ + self._genoCov._ZK, dtype=torch.float32)
            gCov += x[2] * torch.tensor(self._genoCov._ZKZ, dtype=torch.float32)
        
        #   eCov
        if numEnvP == 1: # If no IGE, hence no IEE
            eCov  = x[numGenP+0] * torch.tensor(self._envCov.K0, dtype=torch.float32) 
        elif numEnvP == 2: # When SimplifNonIdableEnvs = True
            print("WARNING: probably SimplifNonIdableEnvs = True, this has to be verified! ")
            # I copied the K function from SigmaRhoCov:
            #def K(self):
            #    RV = self.rho * sp.ones([self.dim, self.dim])
            #    sp.fill_diagonal(RV, 1)
            #    RV = self.scale * RV
            #    return RV * self.K0 
            # when have 2, means SimplifNonIdableEnvs and:  envIntP = np.array([self._envCov.rho, self._envCov.scale]) 
            eCov = x[numGenP+0] * torch.tensor(np.ones([self._envCov.dim, self._envCov.dim]), dtype=torch.float32) # RV = self._envCov.rho * sp.ones([self._envCov.dim, self._envCov.dim])
            #eCov == torch.tensor(RV, dtype=torch.float32)
            mask = torch.eye(self._envCov.dim) == 1 # Creating a mask for the diagonal 
            eCov[mask] = 1 # sp.fill_diagonal(RV, 1) # Filling diagonal with 1
            eCov = x[numGenP+1] * eCov # RV = self._envCov.scale * RV
            eCov = eCov * torch.tensor(self._envCov.K0, dtype=torch.float32) # RV = RV * self._envCov.K0
            # (eCov == torch.tensor(RV, dtype=torch.float32)).all()
        else:
            eCov  = x[numGenP+0] * torch.tensor(self._envCov._K, dtype=torch.float32)
            eCov += x[numGenP+1] * torch.tensor(self._envCov._KZ + self._envCov._ZK, dtype=torch.float32)
            eCov += x[numGenP+2] * torch.tensor(self._envCov._ZKZ, dtype=torch.float32)
        
        #   DmCov
        if numMatnP == 1: # we are in univariate, shouldn't be other option than 1
            DmCov = x[numGenP+numEnvP+0] * torch.tensor(self._maternalCov.K0, dtype=torch.float32)
        else:
            DmCov = None
        #   cCov
        if numCageP == 1: # we are in univariate, shouldn't be other option than 1
            cCov = x[numGenP+numEnvP+numMatnP+0] * torch.tensor(self._cageCov.K0, dtype=torch.float32)
        else:
            cCov = None
        
        # 2. the TOTAL PHENOTYPIC VARIANCE 
        #    env Cov always present
        vtot = 1 / self.gower_torch(eCov)
        
        #    add gen Cov if present
        if gCov is not None:
            vtot = vtot + 1. / self.gower_torch(gCov)
            #print("gCov = ",  1. / self.gower_torch(gCov))
        
        #    add maternal Cov if present
        if DmCov is not None:
            vtot = vtot + 1. / self.gower_torch(DmCov)
    
        #    add cage Cov if present
        if cCov is not None:
            vtot = vtot + 1. / self.gower_torch(cCov)
            
        print('vtot:', vtot.item())
        
        # 3. compile output now
        out = torch.zeros_like(x)
        # Gen eff 
        if numGenP == 0:
            next
        elif numGenP == 1: # There's only one Gen effect - either DGE or IGE, always first of interpret params
            out[0] = x[0] / vtot
        else: 
            out[0] = x[0] / vtot
            out[1] = x[1] / torch.sqrt(x[0]*x[2])
            out[2] = x[2] / vtot
        
        # Env eff
        if numEnvP == 1: # There's only one Env effect, always second of interpret params
            out[numGenP+0] = x[numGenP+0] / vtot
        elif numEnvP == 2:
            print("WARNING: probably SimplifNonIdableEnvs = True, this has to be verified! ")
            out[numGenP+0] = x[numGenP+0] # This should be the correlation already? 
            out[numGenP+1] = x[numGenP+1] / vtot
        else:
            out[numGenP+0] = x[numGenP+0] / vtot
            out[numGenP+1] = x[numGenP+1] / torch.sqrt(x[numGenP+0]*x[numGenP+2])
            out[numGenP+2] = x[numGenP+2] / vtot
        
        # Dam eff - maternal effect
        if numMatnP == 1:
            out[numGenP+numEnvP+0] = x[numGenP+numEnvP+0] / vtot
        
        # Cage eff - not existing, if present, it is vtot anyways
        if numCageP == 1: 
            out[numGenP+numEnvP+numMatnP+0] = vtot
        else:
          if numMatnP ==1: # if no cage effect, vtot is in maternal
              out[numGenP+numEnvP+0] = vtot
          elif numMatnP == 0:
              if numEnvP == 1: # if no cage effect, no maternal and no IEE, vtot is in DEE
                  out[numGenP+0] = vtot # var_Ed1
              elif numEnvP == 2: # in SimplifNonIdableEnvs = True, if no maternal, vtot is in DEE - i.e. second env param
                  out[numGenP+1] = vtot # var_Ed1
              else: # if no cage effect, no maternal and IEE, vtot is in IEE
                  assert numEnvP == 3, "something wrong with the number of environm params"
                  out[numGenP+2] = vtot # var_Es1

        return out


    def transformParams_bi(self, x):
        # Transforms original_VCs into proportional variances and corrs
        # x:
        #     v1gen, v12gen, v2gen, v13gen, v23gen, v3gen, v14gen, v24gen, v34gen, v4gen,
        #     v1env, v12env, v2env, v13env, v23env, v3env, v14env, v24env, v34env, v4env,
        #     v1dam, v12dam, v2dam,    
        #     v1cag, v12cag, v2cag
        # f(x):
        #     prop_v1gen, corr12gen, prop_v2gen, corr13gen, corr23gen, prop_v3gen, corr14gen, corr24gen, corr34gen, prop_v4gen,
        #     prop_v1env, corr12env, prop_v2env, corr13env, corr23env, prop_v3env, corr14env, corr24env, corr34env, prop_v4env,
        #     prop_v1dam, corr12dam, prop_v2dam,
        #     **v1tot**, corr12cag, **v2tot**
        
        # NB: this actaully changes depending on which between DGE, IGE (IEE), maternalEffect, cageEffect are included ### 
        #     especially for the non-genetic components as the last element always needs to be vtot!
        
        # This function needs to take only ONE argument: **aP**
        #                       and output ONE object:   **aPnew** -> needed to calculate the STE with torch.autograd
        # Everything else needs to be defined already, e.g. genoCov-envCov ... and numGenP... or defined inside this function
        numGenP = self.numP['gen']
        numEnvP = self.numP['env']
        numMatnP = self.numP['matn']
        numCageP = self.numP['cage']

        # 1. rebuild Cd,C2, ... as function of x 
        #    x -> 4x4 C as a torch tensor from x (now it is a freeform)
        #    x = [C11, C12, C22, C13, C23, C33, C14, C24, C34, C44]
        
        # Genetic Effects
        if numGenP ==0:
            Cgen=None
            next
        elif type(self._genoCov) is KronCov: # if there's one effect, there should be 3 parameter or 2 params if one cov set to 0, and genoCov be a KronCov
            assert numGenP == 2 or numGenP == 3, "something wrong with the type of genoCov and number of estimated params"
            if numGenP == 3: #bivar with no cov set to 0
                Cgen = x[[0,1,1,2]].reshape(2,2)
            elif numGenP == 2:
                Cgen = x[[0,2,2,1]].reshape(2,2) # Ad1 is in position 0 and Ad2 in position 1. using position 2 (not genetic) to fill up matrix before changing it to 0
                #Cgen[0,1] = 0
                #Cgen[1,0] = 0
                Cgen[0,1] = Cgen[1,0] = 0 # covariance to 0 as corr constrained to 0
        else:
            Cgen = x[[0,1,3,6,1,2,4,7,3,4,5,8,6,7,8,9]].reshape(4,4) 
        
        # Environmental effects
        if type(self._envCov) is KronCov: # if there's one effect, there should be 3 parameter
            assert numEnvP == 3, "something wrong with the type of envCov and number of estimated params"
            Cenv = x[[numGenP+0,numGenP+1,numGenP+1,numGenP+2]].reshape(2,2)
        else:
            Cenv = x[[numGenP+0,numGenP+1,numGenP+3,numGenP+6,
                      numGenP+1,numGenP+2,numGenP+4,numGenP+7,
                      numGenP+3,numGenP+4,numGenP+5,numGenP+8,
                      numGenP+6,numGenP+7,numGenP+8,numGenP+9]].reshape(4,4)
                      
        # Maternal effects
        if numMatnP == 3: # if it is a different number this doesn't work anymore, shouldn't happen anyways
            Cdam = x[[numGenP+numEnvP+0, numGenP+numEnvP+1,
                       numGenP+numEnvP+1, numGenP+numEnvP+2]].reshape(2,2)
        else:
            Cdam=None
        
        # Cage effects
        if numCageP == 3: # if it is a different number this doesn't work anymore, shouldn't happen anyways
            Ccage = x[[numGenP+numEnvP+numMatnP+0, numGenP+numEnvP+numMatnP+1,
                       numGenP+numEnvP+numMatnP+1, numGenP+numEnvP+numMatnP+2]].reshape(2,2)
        else:
            Ccage=None
    
        # 2. Reimplement the function dirindiCovMT.K() ... as need to be as torch.tensor
        def Cd(C): #Cd is a 2x2 D1 D2
            return C[:2,:2]
    
        def Cds(C): #Cds is 2x2 
            return C[:2,2:] #1,2
    
        def Cs(C):
            return C[2:,2:]
        
        def K(C, Cov, Iok=None): # for genoCov and envCov - dirIndirCov_mt.DirIndirCovMT
            #torch.kron(Cd(Cgen), torch.tensor(genoCov._K, dtype=torch.float32))
            RV  = torch.kron(Cd(C), torch.tensor(Cov._K, dtype=torch.float32).contiguous())
            RV += torch.kron(Cds(C), torch.tensor(Cov._KZ, dtype=torch.float32).contiguous())
            RV += torch.kron(Cds(C).T.contiguous(), torch.tensor(Cov._ZK, dtype=torch.float32).contiguous())
            RV += torch.kron(Cs(C), torch.tensor(Cov._ZKZ, dtype=torch.float32).contiguous())
            if Iok is not None:
                 RV = RV[Iok,:][:,Iok]
            return RV

        def K_cage(C, Cov, Iok=None): # for kronecker.KronCov - like cageCov
            R = torch.kron(C, torch.tensor(Cov.R, dtype=torch.float32).contiguous())
            if Iok is not None:
                R = R[Iok][:, Iok]
            return R
          
        print("Cgen = ", Cgen)
        print("Cenv = ", Cenv)
        print("Cdam = ", Cdam)
        print("Ccage = ", Ccage)
        
        # 3. total cov for each group of effects
        #    gCov
        if numGenP ==0:
            gCov= None
        elif type(self._genoCov) is KronCov: # if there's one effect, there should be 3 parameter or 2 parameters
            assert sum(self.Iok_cov) == self._genoCov.K().shape[0], "something wrong with dimensions of gCov"
            gCov = K_cage(Cgen, self._genoCov, self.Iok_cov) 
        else: # if there's one effect, there are 3 parameters minimum
            assert sum(self.Iok_cov) == self._genoCov.K().shape[0], "something wrong with dimensions of gCov"
            gCov = K(Cgen, self._genoCov, self.Iok_cov)
            
        #N = len(self.pheno_ID[self.Iok_pot2N[ : self.pheno_ID.shape[0], ]]) # Length of phenotype 1 
        #gCov1 = 1. / self.gower_torch(gCov[:N][:,:N])
        #gCov2 = 1. / self.gower_torch(gCov[N:][:,N:])
        #print("gCov 1 = ",gCov1)
        #print("gCov 2 = ",gCov2)
        
        #    eCov ... similar to gCov
        # There is always some environment - always minimum 3 parameters
        assert sum(self.Iok_cov) == self._envCov.K().shape[0], "something wrong with dimensions of eCov"
        if type(self._envCov) is KronCov: # if there's one effect, there should be 3 parameter
            eCov = K_cage(Cenv, self._envCov, self.Iok_cov)
        else:
            eCov = K(Cenv, self._envCov, self.Iok_cov)

        #    cDam
        if numMatnP == 3:
            assert sum(self.Iok_cov) == self._maternalCov.K().shape[0], "something wrong with dimensions of matnCov"
            dCov = K_cage(Cdam, self._maternalCov, self.Iok_cov)
        else: 
            dCov = None
        
        #    cCov
        if numCageP == 3:
            assert sum(self.Iok_cov) == self._cageCov.K().shape[0], "something wrong with dimensions of cCov"
            cCov = K_cage(Ccage, self._cageCov, self.Iok_cov)
        else: 
            cCov = None
            
        # 4. Get the total phenotypic variance  
        # a. env Cov always present
        totcov = eCov
        
        # b. add gen Cov if present
        if gCov is not None:
            totcov = totcov + gCov
        
        # c. add dam Cov (maternal) if present
        if dCov is not None:
            totcov = totcov + dCov
    
        # c. add cage Cov if present
        if cCov is not None:
            totcov = totcov + cCov
    
        # get number of samples to divide totcov in vtot1 and vtot2
        #N = len(self.pheno_ID[self.Iok_pot2N[ : self.pheno_ID.shape[0], ]]) # Length of phenotype 1
        assert totcov.shape[0] == self.focal_sample_size1 + self.focal_sample_size2, "cannot extract correctly the total covariance for trait1 and trait2" 
        N = self.focal_sample_size1
        print("size pheno 1, N = ", N)
        vtot1 = 1. / self.gower_torch(totcov[:N][:,:N]) # phenotype 1, up left triangle, dimension NxN
        vtot2 = 1. / self.gower_torch(totcov[N:][:,N:]) # phenotype 2, down right triangle, dimension full-N x full-N
        
        print('vtot1:', vtot1.item())
        print('vtot2:', vtot2.item())
        
        # 5. compile output now
        #    as function of Cgen/Cenv/Cdam/Ccage created above, as function of x
        #    that should be helpful in case of constrained corrs
        out = torch.zeros_like(x)
    
        # Genetic effects: 
        if numGenP ==0:
            next # no gen effect in out
        elif type(self._genoCov) is KronCov: # if there's one effect, there should be 3 parameter
            out[0] = Cgen[0,0] / vtot1 # var_Ad1 or var_As1
            out[1] = Cgen[0,1] / torch.sqrt(Cgen[0,0]*Cgen[1,1]) # cor_Ad1_d2 or cor_As1_s2 # this should be 0 when cov costrained to 0
            out[2] = Cgen[1,1] / vtot2 # var_Ad2 or var_As2
        else:
            out[0] = Cgen[0,0] / vtot1 # var_Ad1
            out[1] = Cgen[0,1] / torch.sqrt(Cgen[0,0]*Cgen[1,1]) # cor_Ad1_d2
            out[2] = Cgen[1,1] / vtot2 # var_Ad2
            
            out[3] = Cgen[0,2] / torch.sqrt(Cgen[0,0]*Cgen[2,2]) # cor_Ad1_s1
            out[4] = Cgen[1,2] / torch.sqrt(Cgen[1,1]*Cgen[2,2]) # cor_Ad2_s1
            out[5] = Cgen[2,2] / vtot1 # var_As1
            
            out[6] = Cgen[0,3] / torch.sqrt(Cgen[0,0]*Cgen[3,3]) # cor_Ad1_s2
            out[7] = Cgen[1,3] / torch.sqrt(Cgen[1,1]*Cgen[3,3]) # cor_Ad2_s2
            out[8] = Cgen[2,3] / torch.sqrt(Cgen[2,2]*Cgen[3,3]) # cor_As1_s2
            out[9] = Cgen[3,3] / vtot2 # var_As2 

        # Environmental effects
        if type(self._envCov) is KronCov: # if there's one effect, there should be 3 parameter
            out[numGenP+0] = Cenv[0,0] / vtot1 # var_Ed1 
            out[numGenP+1] = Cenv[0,1] / torch.sqrt(Cenv[0,0]*Cenv[1,1]) # cor_Ed1_d2
            out[numGenP+2] = Cenv[1,1] / vtot2 # var_Ed2 
        else:
            out[numGenP+0] = Cenv[0,0] / vtot1 # var_Ed1
            out[numGenP+1] = Cenv[0,1] / torch.sqrt(Cenv[0,0]*Cenv[1,1]) # cor_Ed1_d2
            out[numGenP+2] = Cenv[1,1] / vtot2 # var_Ed2
            
            out[numGenP+3] = Cenv[0,2] / torch.sqrt(Cenv[0,0]*Cenv[2,2]) # cor_Ed1_s1
            out[numGenP+4] = Cenv[1,2] / torch.sqrt(Cenv[1,1]*Cenv[2,2]) # cor_Ed2_s1
            out[numGenP+5] = Cenv[2,2] / vtot1 # var_Es1
            
            out[numGenP+6] = Cenv[0,3] / torch.sqrt(Cenv[0,0]*Cenv[3,3]) # cor_Ed1_s2
            out[numGenP+7] = Cenv[1,3] / torch.sqrt(Cenv[1,1]*Cenv[3,3]) # cor_Ed2_s2
            out[numGenP+8] = Cenv[2,3] / torch.sqrt(Cenv[2,2]*Cenv[3,3]) # cor_Es1_s2
            out[numGenP+9] = Cenv[3,3] / vtot2 # var_Es2 
        
        # Maternal effects
        if numMatnP == 3: # if it is a different number this doesn't work anymore, shouldn't happen anyways   
            out[numGenP+numEnvP+0] = Cdam[0,0] / vtot1 # var_Mat1
            out[numGenP+numEnvP+1] = Cdam[0,1] / torch.sqrt(Cdam[0,0]*Cdam[1,1]) # cor_Mat1_2
            out[numGenP+numEnvP+2] = Cdam[1,1] / vtot2 # var_Mat2
            
        # Cage effects
        if numCageP == 3: # if cage effect, vtot is in cage
            out[numGenP+numEnvP+numMatnP+0] = vtot1 # Last one is Vtot
            out[numGenP+numEnvP+numMatnP+1] = Ccage[0,1] / torch.sqrt(Ccage[0,0]*Ccage[1,1]) # cor_C1_2
            out[numGenP+numEnvP+numMatnP+2] = vtot2 # Last one is Vtot
        else:
          if numMatnP ==3: # if no cage effect, vtot is in maternal
              out[numGenP+numEnvP+0] = vtot1
              out[numGenP+numEnvP+2] = vtot2
          elif numMatnP ==0:
              if numEnvP == 3: # if no cage effect, no maternal and no IEE, vtot is in DEE
                  out[numGenP+0] = vtot1 # var_Ed1
                  out[numGenP+2] = vtot2 # var_Ed2
              else: # if no cage effect, no maternal and IEE, vtot is in IEE
                  assert numEnvP > 3, "something wrong with the number of environm params"
                  out[numGenP+5] = vtot1 # var_Es1
                  out[numGenP+9] = vtot2 # var_Es2
        
        return out


    def paramsInMx(self, v, DGE, IGE, IEE, maternalEffect, cageEffect, bivariate):
        """Function to go from array with all params together to matrices for each group of effects"""
        # This funciton is used to make congruent aPs of different length that depends on the model (uni or bi) and the presence/absence of effects
        # -> stores params in matrices per group of effects. If no param, then is -999
        numGenP = self.numP['gen']
        numEnvP = self.numP['env']
        numMatnP = self.numP['matn']
        numCageP = self.numP['cage']
        
        GenM = np.full((4, 4), (-999.))
        EnvM = np.full((4, 4), (-999.)) 
        DamM = np.full((2, 2), (-999.)) 
        CageM = np.full((2, 2), (-999.)) 
        
        if bivariate is False:
            # Gen eff
            if numGenP == 0:
                next
            elif numGenP == 1:
                if DGE and not IGE:
                    GenM[0,0] = v[0] # Ad1
                elif not DGE and IGE:
                    GenM[2,2] = v[0] # As1
                else: 
                    sys.exit("something wrong between gen effects (",",".join([DGE,IGE]),") and number of estimated params: ", numGenP)
            else:
                GenM[0,0] = v[0] # Ad1
                GenM[0,2] = GenM[2,0] = v[1] # Ad1s1
                GenM[2,2] = v[2] # As1
                
            # Env eff
            if numEnvP == 1: # if there is only one Env effect it is DEE; cannot have IEE only
                EnvM[0,0] = v[numGenP+0] # Ed1 #checked that went through here Amelie
            elif numEnvP == 2: 
                print("WARNING: probably SimplifNonIdableEnvs = True, this has to be verified! Env var stored as 'Ed1' and Env Corr stored as 'Ed1s1'")
                EnvM[0,0] = v[numGenP+1] # Ed1 / Es1 / C1 # That is vtot if no matn
                EnvM[0,2] = EnvM[2,0] = v[numGenP+0] # EnvCorr - I don't know what this corresponds to
            else:
                EnvM[0,0] = v[numGenP+0] # Ed1
                EnvM[0,2] = EnvM[2,0] = v[numGenP+1] # Ed1s1
                EnvM[2,2] = v[numGenP+2] # Es1
            
            # Dam eff - maternal effect
            if numMatnP == 1: 
                DamM[0,0] = v[numGenP+numEnvP+0]
            
            # Cage eff - not existing - last one always is vtot
            if numCageP == 1: 
                CageM[0,0] = v[numGenP+numEnvP+numMatnP+0] # vtot # last one always need to be vtot
                
        else: # this is for bivariate 
            # Gen eff
            if numGenP == 0:
                next
            elif numGenP == 2 or numGenP == 3: # when only 2 or 3 gen params, we have only 1 of the two gen effects
                if DGE and not IGE:
                    GenM[0,0] = v[0] # Ad1
                    GenM[0,1] = GenM[1,0] = v[1] # Ad1d2
                    GenM[1,1] = v[2] # Ad2
                elif not DGE and IGE:
                    GenM[2,2] = v[0] # As1
                    GenM[2,3] = GenM[3,2] = v[1] # As1s2
                    GenM[3,3] = v[2] # As2
                else: 
                    sys.exit("something wrong between gen effects (",",".join([DGE,IGE]),") and number of estimated params: ", numGenP)
            else:
                GenM[0,0] = v[0] # Ad1
                GenM[0,1] = GenM[1,0] = v[1] # Ad1d2
                GenM[1,1] = v[2] # Ad2
                
                GenM[0,2] = GenM[2,0] = v[3] # Ad1s1
                GenM[1,2] = GenM[2,1] = v[4] # Ad2s1
                GenM[2,2] = v[5] # As1
                
                GenM[0,3] = GenM[3,0] = v[6] # Ad1s2
                GenM[1,3] = GenM[3,1] = v[7] # Ad2s2
                GenM[2,3] = GenM[3,2] = v[8] # As1s2
                GenM[3,3] = v[9] # As2
                
            # Env effects
            if numEnvP == 3 :  # if there's one effect, there should be 3 parameter, and cannot be IEE only, so it is a case of DEE only
                 EnvM[0,0] = v[numGenP+0] # Ed1
                 EnvM[0,1] = EnvM[1,0] = v[numGenP+1] # Ed1d2
                 EnvM[1,1] = v[numGenP+2] # Ed2
            else:
                 EnvM[0,0] = v[numGenP+0] # Ed1
                 EnvM[0,1] = EnvM[1,0] = v[numGenP+1] # Ed1d2
                 EnvM[1,1] = v[numGenP+2] # Ed2
                 
                 EnvM[0,2] = EnvM[2,0] = v[numGenP+3] # Ed1s1
                 EnvM[1,2] = EnvM[2,1] = v[numGenP+4] # Ed2s1
                 EnvM[2,2] = v[numGenP+5] # Es1
                 
                 EnvM[0,3] = EnvM[3,0] = v[numGenP+6] # Ed1s2
                 EnvM[1,3] = EnvM[3,1] = v[numGenP+7] # Ed2s2
                 EnvM[2,3] = EnvM[3,2] = v[numGenP+8] # Es1s2
                 EnvM[3,3] = v[numGenP+9] # Es2
                 
            #maternal
            if numMatnP == 3: # If maternalEffect, always numMatnP == 3 because we are in bivariate
                DamM[0,0] = v[numGenP+numEnvP+0] #Dm1
                DamM[0,1] = DamM[1,0] = v[numGenP+numEnvP+1] #Dm1Dm2
                DamM[1,1] = v[numGenP+numEnvP+2] #Dm2
          
            #cage
            if numCageP == 3: # If CageEffect, always numCageP == 3 because we are in bivariate
                CageM[0,0] = v[numGenP+numEnvP+numMatnP+0] # vtot1 or STEvtot1
                CageM[0,1] = CageM[1,0] = v[numGenP+numEnvP+numMatnP+1]
                CageM[1,1] = v[numGenP+numEnvP+numMatnP+2] # vtot2 or STEvtot2
            # When no cage effect, vtot is stored either in DamM or in EnvM if no Maternal effect
            # Keep in mind this for when retrieving outpu
            
        out = [GenM, EnvM, DamM, CageM]
        
        return out


    def gower_torch(self, C):
         """This is the same function as 'covar_rescaling_factor' but written using torch instead of sp. """
         n = C.shape[0]
         P = torch.eye(n) - torch.ones((n,n)) / float(n)
         trPCP = torch.trace(torch.mm(P, torch.mm(C,P))) # or torch.mm(P, torch.mm(C,P)).diagonal().sum(0,keepdim=True)
         out = (n - 1) / trPCP
         return out


    def getOutput(self):
        """to get output without having to specify DGE, IGE, ...."""
        return self.output

    #def getToSave(self): # this split into 2, with implementation to save in case of optimization failing at first try
    #    return {'sampleID' : self.pheno_ID[self.Iok_N], 'pheno' : self.pheno,'covs' : self.covs,'covar_mat' : self.getDirIndirVar().K()}
    
    def getToSaveInfos(self):
        """Retrieve pheno_IDs, pheno, covs even if optimization failed"""
        #return {'sampleID_all' : self.sampleID_all,'pheno' : self.pheno,'covs' : self.covs}
        return {'sampleID' : self.pheno_ID[self.Iok_N], 'pheno' : self.pheno,'covs' : self.covs}

    def getToSaveCovar(self):
        """Retrieve covariance matrix - only when optimization successfull"""
        return {'covar_mat' : self.getDirIndirVar().K()}

    def getDirIndirVar(self):
        """to get overall, fitted covariance matrix"""
        return self.covar
      
    def getDirIndirVCinit(self):
        """to get cov matrices for vc_init"""
        if self._genoCov is not None:
            if type(self._genoCov) is FixedCov:
                genoCov_K = self._genoCov.scale 
            else:    
                genoCov_K = self._genoCov.C.K() #vc_init._genoCov.C.K()
        else:
            genoCov_K = None
        
        if self._envCov is not None:
            if type(self._envCov) is FixedCov:
                envCov_K = self._envCov.scale 
            else:    
                envCov_K = self._envCov.C.K() #vc_init._envCov.C.K()
        else:
            envCov_K = None
        
        if self._maternalCov is not None: 
            if type(self._maternalCov) is FixedCov:
                maternalCov_K = self._maternalCov.scale # vc_init._maternalCov.scale 
            else:
                maternalCov_K = self._maternalCov.C.K() # vc_init._maternalCov.C.K()
        else:
            maternalCov_K = None
            
        if self._cageCov is not None: 
            if type(self._cageCov) is FixedCov:
                cageCov_K = self._cageCov.scale # vc_init._cageCov.scale 
            else:
                cageCov_K = self._cageCov.C.K() # vc_init._cageCov.C.K()
        else:
            cageCov_K = None
        return {'genoCov_K': genoCov_K, 'envCov_K': envCov_K, 'maternalCov_K': maternalCov_K, 'cageCov_K': cageCov_K}
