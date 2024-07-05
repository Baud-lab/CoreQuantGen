#!/usr/bin/env python

### TODOS:
#a. Can change the structure of out_dir and subdirs that are created
#b. should check to have 'social_data_wMat' 'dirIndirVD_wMT_wMat' (or dirIndirVD_noMT_wMat) 'diagcov' 'dirIndirCov_mt' 'dirIndirCov_v2' in a subfolder called "classes"

#-----------------------------------#
#	  I. DEFINING COMMAND-LINE ARGS
#-----------------------------------#	 
# command line arguments are before importing modules so that can use --help without having to load all modules (which requires some time)
import argparse as argp 

# Function converting value to int when it is one, otherwise converted to string
def int_or_str(value):
    """This function converts value to integer if it is one, otherwise put it in string"""
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If it can't be converted to an integer, return it as a string
        return str(value)

parser = argp.ArgumentParser(description ='Analysis of phenotypic variance with univariate or bivariate model')
# Positional arguments - NEEDED : input path / pheno version / covs version / cage version / GRM version / mode
parser.add_argument('--input', help = 'path to input hdf5 file', required=True)
parser.add_argument('--phenos_v', help= 'type of PHENOTYPES to use, i.e. name of subgroup of `phenotypes` in .h5, e.g. simulations', required=True)
parser.add_argument('--covs_v', help= 'type of COVARIATES to use, i.e. name of subgroup of `covariates` in .h5, e.g. from_file, set `None` if dont want to use', required=True)
parser.add_argument('--cage_v', help= 'type of CAGE to use, i.e. name of subgroup of `cages` in .h5, e.g. real',required=True)
parser.add_argument('--dam_v', help= 'type of DAM to use, i.e. name of subgroup of `dam` in .h5, e.g. real, set `None` if dont want to use', required=True)
parser.add_argument('--grm_v', help= 'kinship type, e.g. Andres_kinship', required=True)
parser.add_argument('--analysis_type', help= 'set `VD` for heritability analysis or `null_covars_LOCO` for generating the covariance matrices of the null model for Leave One Chromosome Out GWAS', required=True)

# Optional arguments
parser.add_argument('-o','--out', help = 'path to folder in which create output folders and files [default="."]', default=".")
parser.add_argument('-c','--combins_path',nargs="?", help = 'path to file with pairs of phenotypes to compare (.csv), needed for bivariate')
parser.add_argument('-p','--pheno', type=int_or_str, help = 'Number of phenotypes col to analyse, or line to combins file with pair of phenotypes in case of bivar [default=1]', default=1)
parser.add_argument('-m','--model', help='type of model: univariate (uni) or bivariate (bi) [default=uni]', default="uni")
parser.add_argument('-e','--effects',help='effects - DGE,IGE,IEE,cageEffect,maternalEffect - to be included [default= None to all]', default=None)
parser.add_argument('-s','--subset', help='to handle optional subset on individuals [default=None]', default=None)
parser.add_argument('-z','--zero', help='which corr to set to 0: corr_Ad1d2 or (once implemented by Helene) other options [default=None]', default=None)
parser.add_argument('-C','--chrom', type = str, help = 'comma-separated string of chromosomes or list ')

args = vars(parser.parse_args())

#-------------------------#
#  II. IMPORTING MODULES
#-------------------------#	 
import os
import sys

# They social_data_wMat and dirIndirVD_wMT need to be in the same folder or in a "classes" subfolder
from classes.social_data_wMat import SocialData # if modifying this file, always update the latest version as here
#from classes.dirIndirVD_wMT_wMat import DirIndirVD 
from classes.dirIndirVD_noMT_wMat import DirIndirVD 
import pdb
import h5py
import re
import csv
import time
import pandas as pd
import numpy as np # need to generate random seed
import pickle
import gzip
import gc # needed at the end - gc.collect()

start_time = time.time()

if __name__=='__main__':
    ##################################################################
    ######################   A. GET ARGUMENTS   ######################
    ##################################################################
  
    #---------------------------------------------------------#
    #	       A1. VERSIONS FOR INPUT FILE = *.h5 file		  
    #---------------------------------------------------------#
  
    in_file=args['input'] # input file name
    phenos_version=args['phenos_v'] # e.g. phenos_version = 'simulations'
    covs_version=args['covs_v'] 
    if covs_version == "None":
      covs_version = None
    cage_version=args['cage_v'] 
    dam_version=args['dam_v'] 
    if dam_version == "None":
      dam_version = None
    GRM_version = args['grm_v'] # e.g. GRM_version = 'Andres_kinship'
  
    #-----------------------------------------------------------------#
    #  A2. SUBSET, if any - to handle optional subset on individuals
    #-----------------------------------------------------------------#	 
    subset=args['subset']
    if subset == "None":
      subset = None
    print("Subset: ",  subset)
    
    #---------------------------------------------------------#
    #				           A3. MODEL      						  
    #---------------------------------------------------------#

    m = args['model']
    if m == "bivariate" or m == "bi":
        model = "bi"
        print("model is BIVARIATE: ", model)
    
    elif m == "univariate" or m == "uni":
        model = "uni"
        print("model is UNIVARIATE: ", model)
        
    else:
        sys.exit("Need to know which model to use: " + "'"+ m +"' " + "is not valid" + "\n==> Use 'uni' / 'univariate' OR 'bi' / 'bivariate'")
   
        
    #---------------------------------------------------------#
    #			A4. EFFECTS TO INCLUDE in the model
    #---------------------------------------------------------#

    #DGE = "DGE" #set to "DGE" if you want to include DGE in your models, to None if you don't
    #IGE = "IGE" #"IGE" or None
    #IEE = "IEE" #"IEE" or None
    #cageEffect = "cageEffect" #"cageEffect" or None
    #maternalEffect = "maternalEffect" #"maternalEffect" or None
    # Setting all to None by default
    DGE = None
    IGE = None
    IEE = None
    cageEffect = None
    maternalEffect = None
    
    effects = [DGE, IGE, IEE, cageEffect, maternalEffect]

    if args['effects'] is not None:
        effects = args['effects'].split(",")
    
    if "IGE" in effects:
        IGE = "IGE"
        IEE = "IEE"
        
    if "DGE" in effects:
        DGE = "DGE"
        
    if "cageEffect" in effects:
        cageEffect = "cageEffect"
    
    if "maternalEffect" in effects:
        maternalEffect = "maternalEffect"
    if maternalEffect is None:
        dam_version = None
        
    print("Effects included in model: ", DGE, IGE, IEE, cageEffect, maternalEffect)

    #---------------------------------------------------------#
    #	     A5. GETTING PHENOTYPE NAME or COLUMN NUMBER    	  
    #---------------------------------------------------------#

    #BIVARIATE MODEL
    #few lines below go through a csv file that has on each row PAIRS OF PHENOTYPES saying which pairs of phenotypes you want to consider (doesnt have to be all combinations);
    #Can be INDXS as in the H5 phenotype matrix or phenotype name; NB: phenotypes cannot have name as an integer number!
    #for example it may look like:
    #   1,2
    #   blood.Na,body_weight
    #   5,8
    #pdb.set_trace()
    if model == "bi":
        assert args['combins_path'] is not None, 'Need combins file'
        assert isinstance(args['pheno'], int), 'We are in bivariate, need the line of combins file to analyse'
        # reading combins file
        dfcombins = pd.read_csv(args['combins_path'], header=None, sep=',')
        #pdb.set_trace() 
        #this works with combins file with pheno_names or numbers
        selected_pheno, selected_pheno_MT = dfcombins.iloc[args['pheno'] - 1] 
        selected_pheno = int_or_str(selected_pheno) # if int, NOT YET in python 0-indexing
        selected_pheno_MT = int_or_str(selected_pheno_MT) # if int, NOT YET in python 0-indexing
        
    elif model == "uni": #or more simply for UNIVARIATE
        #col = (args['pheno']-1) 
        #selected_pheno = args['pheno'] 
        selected_pheno = args['pheno'] # if int, NOT YET in python 0-indexing
        selected_pheno = int_or_str(selected_pheno)
        selected_pheno_MT = None
        
    #-----------------------------------------------------------------#
    #                 A6. DEFINE TYPE OF ANALYSIS  
    #-----------------------------------------------------------------#    

    analysis_type = args['analysis_type']
    assert analysis_type == 'VD' or analysis_type == 'null_covars_LOCO', 'set analysis_type argument to `VD` or `null_covars_LOCO`'
    assert not model == 'bi' or not analysis_type == 'null_covars_LOCO', "can't generate LOCO variances with bivariate model yet"
 
    #-----------------------------------------------------------------#
    #      A7. DEFINE list of chromosomes (if null_covars...)  
    #-----------------------------------------------------------------#    
    if analysis_type == 'null_covars_LOCO': 
        #pdb.set_trace()
        assert args['chrom'] is not None and len(args['chrom']) > 0, "need at least one chromosome for null_covars_LOCO"
        chrs = []
        for item in args['chrom'].split(","):
            if "-" in item:
                start, end = map(int, item.split("-"))
                chrs.extend(range(start, end+1))
            else:
                chrs.append(int(item))
    
    ### SETTING SEED TO MAKE IT RANDOM AND REPRODUCIBLE ###
    ### TODO: set "sid = None" if don't want to use the seed, it requires to comment two lines below with 'sid = sid+1'
    sid = np.random.randint(0, 10000) #SEED_test
    #sid=500 # to test specific #TODO: change this to above!
    print('initial seed set is:', sid)
    
    
    ################################################################
    ###########   B. PARSING INPUT AND DEFINING OUTPUT   ###########
    ################################################################
    #if True:
    #if analysis_type == 'VD': # Helene's improved code (TOREMOVE comment when done)
    #-----------------------------------------------------------------#
    #		  B1. PARSING INPUT .h5 with SocialData() 
    #-----------------------------------------------------------------#	 
    ## will now use code in social_data_wMat.py 
    # Order in SocialData (self, in_file=None, phenos_version = None,covs_version=None, cage_version=None, dam_version = None, GRM_version = None, subset = None, chrom = None)
    
    #pdb.set_trace()
    data = SocialData(in_file, phenos_version, covs_version, cage_version, dam_version, GRM_version, subset) 
    
    traits = data.get_trait(selected_pheno, selected_pheno_MT)
    trait1 = traits['trait']
    trait2 = traits['trait_MT'] # None if univariate
    # then will assert if this corresponds to 'trait' and 'trait_MT' of doto = data.get_data(selected_pheno, selected_pheno_MT); 
    print("traits are " + " and ".join(filter(None, [trait1,trait2])))

    #-----------------------------------------------------------------#
    #					       B2. DEFINING OUTPUT DIR
    #-----------------------------------------------------------------#	 
    ## NB: 1. cannot go before parsing input because trait1 and trait2 are output of SocialData.get_data
    ##     2. if not existing, CREATE A NEW DIR inside the dir given to --out option
    out_dir=args['out']
    if out_dir is None:
        out_dir = os.getcwd()
        
    # this is common to VD and null_covars_LOCO
    if model == "bi":
        outfile_dir = "".join([out_dir,"/",analysis_type,"/",model,'variate/',phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/',trait1,'/'])
        #VD_outfile_dir = "".join([out_dir,"/",model,'variate/',phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/',trait1,'/'])
    elif model == "uni":
        outfile_dir = "".join([out_dir,"/",analysis_type,"/",model,'variate/',phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/'])
        #VD_outfile_dir = "".join([out_dir,"/",model,'variate/',phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/'])
    
    # adding a subdir in 'null_covars_LOCO'
    if analysis_type == 'null_covars_LOCO':
        # line from amelie, doesn't have the model, don't think it is a problem if it's in there  
        #covar_outfile_dir = "".join([out_dir,"/null_covars_LOCO/",phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/',trait,'/'])
        if model == "uni":
            outfile_dir = "".join([outfile_dir,'/',trait1,'/'])
        elif model == "bi":
            sys.exit("can't generate LOCO variances with bivariate model yet")
            #VD_outfile_dir = "".join([VD_outfile_dir,'/',trait2,'/']) # TODO: not sure this is how we are going to implement it, but might be an idea
            
    # creating dir if not existing
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir, exist_ok=True) # exist_ok=True should avoid problem in case it already exists
    
    
    ################################################################
    ############   C. VARIANCE DECOMPOSITION ANALYSIS   ############
    ################################################################

    if analysis_type == 'VD': # 
        
        ### Generating file name  ###    NB: in VD depends on trait1, trait2 and corr0 - if any; in null_covars_LOCO depends on chr!!! 
        #VD_outfile_pickle="".join([VD_outfile_dir,"_".join(filter(None, [trait1,trait2])),".pkl"]) 
        
        corr0 = args['zero'] #Amelie
        #if corr0 is None:
        #if True: # this happens anyways # TOREMOVE: comment the if True and move back of one indent
        VD_outfile_name="".join([outfile_dir,"_".join(filter(None, [trait1,trait2]))]) 
        #VD_outfile_pickle="".join([VD_outfile_name, "_VC.pkl"])
        VD_outfile_pickle="".join([VD_outfile_name, "_VC.pkl.gz"])
          
        if corr0 is not None: # == 'corr_Ad1d2': # if there is corr0, add it to the filename and that is going to be it
            assert corr0 == 'corr_Ad1d2', "covariance class with corr other than Ad1d2 not yet implemented. Torna-hi mes tard." # TODO: check how to add this that should be possible in univariate: **or corr0 == 'corr_Ad1s1'**
            if corr0 == "corr_Ad1d2":
                 assert model == "bi" and DGE is not None, " ".join(["need bivariate and DGE to constrain", corr0, "to 0"])
            #elif corr0 == "corr_Ad1s1":
            #     assert DGE is not None and IGE is not None, "need DGE and IGE for " + corr0
            VD_outfile_name="_".join([VD_outfile_name,corr0,'zero']) # add at the end of the normal filename "_corr*_zero" 


        #-----------------------------------------------------------------#
        #           C1. VD in alternative or corr to 0 
        #-----------------------------------------------------------------# 
        ### 0. Get data from 'data' ###
        doto = data.get_data(selected_pheno, selected_pheno_MT) 
        assert doto['trait'] == trait1, "something wrong in parsing data, doto['trait'] and trait1 do not correspond"
        assert doto['trait_MT'] == trait2, "something wrong in parsing data, doto['trait_MT'] and trait2 do not correspond"

        ### 1. Get VC, alternative model, without constraint ###
        calc_ste_VD = True
        standardize_pheno_VD = True
        
        ## If the pipeline has never yet been run or if it has not been run successfully, we'll start trying to run VD up to 5 times
        if not os.path.isfile(VD_outfile_pickle): # pickle not existing, meaning we are running for the first time
            try_nb_VD = 1
            while try_nb_VD < 6: # SEED_test: increase the number of tries until one works/doens't work, the seed will be printed so that you can reproduce it
                
                #pdb.set_trace()
                # Running DirIndirVD, i.e. fitting of the model
                vc = DirIndirVD(pheno = doto['pheno'], pheno_ID = doto['pheno_ID'], covs = doto['covs'], covs_ID = doto['covs_ID'], covariates_names = doto['covariates_names'], kinship_all = doto['kinship_full'], kinship_all_ID = doto['kinship_full_ID'],  cage_all = doto['cage_full'], cage_all_ID = doto['cage_full_ID'], maternal_all = doto['maternal_full'], maternal_all_ID = doto['maternal_full_ID'], subset_IDs = doto['subset_IDs'], 
                                DGE = (DGE is not None), IGE = (IGE is not None), IEE = (IEE is not None), cageEffect = (cageEffect is not None), maternalEffect = (maternalEffect is not None), 
                                calc_ste=calc_ste_VD, subset_on_cage = False, SimplifNonIdableEnvs = False, vc_init_type = None, vc_init = None, seed = sid, standardize_pheno = standardize_pheno_VD) #standardize_pheno = True) #standardize_pheno = False) 
                # line that Amelie is using # difference standardize_pheno = False # TOREMOVE once checked is ok
                dirIndir_out = vc.getOutput() # getting output
                vc_covs = vc.getDirIndirVCinit() # getting covariance matrices used in vc_init
                
                try_nb_VD = try_nb_VD + 1
                # will check if vc was successfull, if not, will try again
                if "optim_error" in dirIndir_out.keys():
                    assert dirIndir_out['optim_error'] is not None, "optimization error is None, when expected to be something"
                    if try_nb_VD < 6:
                        print("Warning: optimization was not successful, gonna try for the ", try_nb_VD , " /5 time")
                    else:
                        print("Warning: optimization was not successful after 5 tries")
                        break
                    sid = sid+1   # SEED_test: change number for next test
                else:
                    print("Optimization successfull!")
                    break
                  
            # Saving vc to a pickle file at the end, whether it was successfull or not; this is to avoid fitting again in case want to run corr0
            # TODO: have to check how much space this takes - a lot!!!
            # Open a file to save the object to file name 
            #with open(VD_outfile_pickle, 'wb') as f:
            #pdb.set_trace()
            #VD_objects = {'vc_covs': {'_genoCov': vc._genoCov, '_envCov': vc._envCov, '_maternalCov': vc._maternalCov, '_cageCov': vc._cageCov}, } # still quite big because saving matrices
            #VD_objects = {'doto': doto, 'vc_covs': vc_covs, 'dirIndir_out' : dirIndir_out} # if saving doto as well, the size increases quite a lot depending on the GRM and the number of phenotypes...
            VD_objects = {'vc_covs': vc_covs, 'dirIndir_out' : dirIndir_out} # 
            with gzip.open(VD_outfile_pickle, "wb") as f:
            #with open(VD_outfile_pickle, 'wb') as f:
                 # Use pickle to serialize the object and save it to the file
                 #pickle.dump(vc, f)
                 pickle.dump(VD_objects, f)
                 print("Files saved to", f)

        else: # if existing already, loading from pickle
            # !!! WARNING: doto is not saved in the pickle, there might be inconsistency, might be a problem as there is not track of which *_version has been used to optimize vc
            #with open(VD_outfile_pickle, 'rb') as f:
            with gzip.open(VD_outfile_pickle, "rb") as f:
            # Use pickle to deserialize the object from the file
                #vc = pickle.load(f) # now I have vc from pickle
            #dirIndir_out = vc.getOutput() # now I have 'vc' and 'dirIndir_out' from alternative model previously optimised
                VD_objects = pickle.load(f)
            vc_covs = VD_objects['vc_covs']
            dirIndir_out = VD_objects['dirIndir_out']
            #pdb.set_trace()  
            
            if "optim_error" in dirIndir_out.keys(): # if the optimization wasn't successfull, will exit because can't do anything anyway
                sys.exit("after 5 times couldn’t optimize vc, change initial seed or increase nb of tries")
            else: 
                # NB: this might be a problem if change cage_version, dam_version or cov_version as there is no track anywhere... and socialData is created new at each time
                print("'vc' for alternative model already existed, loaded from ", VD_outfile_pickle)
                print("!!! WARNING !!!: 'doto' is not saved in the pickle, there might be inconsistency as there is not track of which *_version has been used to optimize vc;\
                    \n                  ***Delete pickle and run again if you are not sure***")
        
        
        ### 2. Get VC_lr, null model, with specific corr constrained to 0 ###
        # at this point I have 'vc' and 'dirIndir_out' either just created or loaded from the pickle
        if corr0 is not None: 
            # TODO: If want to set 'calc_ste' and 'standardize_pheno' differently than set for VD alternative, uncomment the following and set as desired
            #calc_ste_VD = True
            #standardize_pheno_VD = True

            #my_vc_init = vc # I imagine this to be the same for every case of corr constrained to 0, if not, move inside each specific corr
            my_vc_init = vc_covs
            if corr0 == "corr_Ad1d2": #or corr0 == "corr_Ad1s1": #TODO check if this want
                my_vc_init_type = 'diagonal' # this might change depending on the corr?
            else:
                sys.exit("covariance class with corr other than Ad1d2 not yet implemented. Torna-hi mes tard.")

            # do the same as done for vc - try to optimise max 5 times - 
            # NB: !! if tried more than once for alternative_vc, the seed changed! - which is not a problem, just bear in mind 
            try_nb_VD = 1
            while try_nb_VD < 6:
                print('in vc_lr')
                
                vc_lr = DirIndirVD(pheno = doto['pheno'], pheno_ID = doto['pheno_ID'], covs = doto['covs'], covs_ID = doto['covs_ID'], covariates_names = doto['covariates_names'], kinship_all = doto['kinship_full'], kinship_all_ID = doto['kinship_full_ID'],  cage_all = doto['cage_full'], cage_all_ID = doto['cage_full_ID'], maternal_all = doto['maternal_full'], maternal_all_ID = doto['maternal_full_ID'], subset_IDs = doto['subset_IDs'],
                                   DGE = (DGE is not None), IGE = (IGE is not None), IEE = (IEE is not None), cageEffect = (cageEffect is not None), maternalEffect = (maternalEffect is not None), 
                                   calc_ste=calc_ste_VD, subset_on_cage = False, SimplifNonIdableEnvs = False, vc_init_type = my_vc_init_type, vc_init = my_vc_init, seed = sid, standardize_pheno = standardize_pheno_VD)
                dirIndir_out_lr = vc_lr.getOutput() 
                
                try_nb_VD = try_nb_VD + 1
                # will check if vc was successfull, if not, will try again
                if "optim_error" in dirIndir_out.keys():
                    assert dirIndir_out['optim_error'] is not None, "optimization error is None, when expected to be something"
                    if try_nb_VD < 6:
                        print("Warning: optimization was not successful, gonna try for the ", try_nb_VD , " /5 time")
                    else:
                        print("Warning: optimization was not successful after 5 tries")
                        break
                    sid = sid+1   # SEED_test: change number for next test
                else:
                    print("Optimization successfull!")
                    break
        
        ## at this point I have:
        ##    'vc' and 'dirIndir_out'
        ##    'vc_lr' and 'dirIndir_out_lr' if corr0 not None; otherwise not existing
        
        #-----------------------------------------------------------------#
        #                    C2. GETTING THE OUTPUT 
        #-----------------------------------------------------------------# 
        
        ### 1. Define function to get the output written as want to 
        def writeDirIndir(out): # out = dirIndir_out...
            """Function to write estimates and ste in output from dirIndir_out"""
            
            #pdb.set_trace()
            if "optim_error" in out.keys():
                #here need to return a warning to let the user know but nothing more as we don't want the pipeline to fail in case optim was not successful
                print("Warning: after 5 tries optimization was still not successful; saving NAs to file")
                #print(dirIndir_out['optim_error'])
                # Has to correspond to as many NAs as in 'est' # TODO if changes
                est = (trait1, trait2) + ('NA',) * 43 # the number corresponds to the fields in est if successfull; thanks gemini for the solution
                
                # Has to correspond to as many NAs as in 'toWrite_ste' # TODO if changes
                ste = (trait1, trait2) + ('NA',) * 38 # the number corresponds to the fields in ste if successfull; thanks gemini for the solution
        
            else:
                #pdb.set_trace()
                est = (trait1, trait2, out['sample_size1'], out['sample_size1_cm'], out['sample_size2'], out['sample_size2_cm'], 
                  out['union_focal'], out['inter_focal'], out['union_cm'], out['inter_cm'], # added this on 6.06.24
                  ",".join(out['covariates_names']), out['conv'], out['LML'], # 13
                  # NB: the order of VCs in output is different than the order in the matrix, I kept the one that was in the code before
                  ## TODO: tot_env_var1, tot_env_var2, if care about it
          
                  out['prop_Ad1'], out['prop_Ad2'],
                  out['prop_As1'], out['prop_As2'], 
                  out['corr_Ad1d2'], out['corr_Ad1s1'], 
                  out['corr_Ad1s2'], out['corr_Ad2s1'],
                  out['corr_Ad2s2'], out['corr_As1s2'], # 10
          
                  out['prop_Ed1'], out['prop_Ed2'],
                  out['prop_Es1'], out['prop_Es2'], 
                  out['corr_Ed1d2'], out['corr_Ed1s1'],
                  out['corr_Ed1s2'], out['corr_Ed2s1'], 
                  out['corr_Ed2s2'], out['corr_Es1s2'], # 10
                  
                  out['prop_Dm1'], out['prop_Dm2'],
                  out['corr_Dm1Dm2'], # 3
                  
                  out['prop_C1'],  out['prop_C2'],
                  out['corr_C1C2'], # 3
                  
                  out['tot_genVar1'], out['tot_genVar2'],
                  out['total_var1'], out['total_var2'],  # 4 
                  )
                est = [-999 if x == -999 or x == -999.0 else x for x in est] # -999 are sometimes numbers, sometimes string - converting them to all the same 
              
                ste = (trait1, trait2, out['time_exec'], # 3 
                    out['STE_Ad1'], out['STE_Ad2'], 
                    out['STE_As1'], out['STE_As2'], 
                    out['STE_Ad1d2'],  out['STE_Ad1s1'], 
                    out['STE_Ad1s2'],  out['STE_Ad2s1'],
                    out['STE_Ad2s2'],  out['STE_As1s2'], # 10
            
                    out['STE_Ed1'], out['STE_Ed2'], 
                    out['STE_Es1'], out['STE_Es2'], 
                    out['STE_Ed1d2'], out['STE_Ed1s1'],
                    out['STE_Ed1s2'], out['STE_Ed2s1'], 
                    out['STE_Ed2s2'], out['STE_Es1s2'], # 10
                    
                    out['STE_Dm1'], out['STE_Dm2'],
                    out['STE_Dm1Dm2'], # 3
        
                    out['STE_C1C2'], # NB: because of the way we calculate STE we cannot get STE for C1/C2; 
                    
                    out['STE_totv1'], out['STE_totv2'], # 3
                    
                    out['corParams_Ad1_As1'], out['corParams_Ed1_Es1'], out['corParams_Ed1_Dm1'], out['corParams_Es1_Dm1'], 
                    out['corParams_Ad2_As2'], out['corParams_Ed2_Es2'], out['corParams_Ed2_Dm2'], out['corParams_Es2_Dm2'], # 8
                    )
                    
                #pdb.set_trace()
                ste = [-999 if x == -999 or x == -999.0 else x for x in ste] # -999 are sometimes numbers, sometimes string - converting them to all the same 
                
            return{'est': est, 
                   'ste': ste}


        ### 2. Use function to get output to write
        toWrite_est = writeDirIndir(dirIndir_out)['est']
        toWrite_ste = writeDirIndir(dirIndir_out)['ste']
        if corr0 is not None:
            toWrite_est_lr = writeDirIndir(dirIndir_out_lr)['est'] # dirIndir_out_lr exists only if corr0 is not None
            toWrite_ste_lr = writeDirIndir(dirIndir_out_lr)['ste']
        
        ### 3. Write output to files
        #pdb.set_trace()
        #print(toWrite_est)
        VD_outfile_est=open("".join([VD_outfile_name,'_est.txt']),'w')
        # TODO: evaluate if want to move creation of 'VD_outfile_est' file name (and then open here) to exit if exists
        VD_outfile_est.write("\t".join(str(e) for e in toWrite_est)+'\n')
        if corr0 is not None: # toWrite_est_lr exists only if corr0 is not None
            #if corr0 == 'corr_Ad1d2' or :
            VD_outfile_est.write("\t".join(str(e) for e in toWrite_est_lr)+'\n')
        VD_outfile_est.close()
        
        #print(toWrite_ste)
        VD_outfile_ste=open("".join([VD_outfile_name,'_STE.txt']),'w')
        # TODO: evaluate if want to move creation of 'VD_outfile_ste' file name (and then open here) to exit if exists
        VD_outfile_ste.write("\t".join(str(e) for e in toWrite_ste)+'\n')
        if corr0 is not None: # toWrite_est_lr exists only if corr0 is not None
            #if corr0 == 'corr_Ad1d2':
            VD_outfile_ste.write("\t".join(str(e) for e in toWrite_ste_lr)+'\n')
        VD_outfile_ste.close()

        ##pdb.set_trace()
        #print(toWrite_est)
        #VD_outfile_est=open("".join([VD_outfile_name,'_est.txt']),'w')
        #VD_outfile_est.write("\t".join(str(e) for e in toWrite_est)+'\n')
        #VD_outfile_est.close()
        #
        #print(toWrite_ste)
        #VD_outfile_ste=open("".join([VD_outfile_name,'_STE.txt']),'w')
        #VD_outfile_ste.write("\t".join(str(e) for e in toWrite_ste)+'\n')
        #VD_outfile_ste.close()
        print("Files with estimates and STE saved to", outfile_dir)


    ################################################################
    ###################   D. MAP LOCO ANALYSIS   ###################
    ################################################################
        
    elif analysis_type == 'null_covars_LOCO': 
        #-----------------------------------------------------------------#
        #      Calculate covariance matrix for GRM_LOCO of each chr
        #-----------------------------------------------------------------# 
        #chrs = list(range(1,21)) # this has been implemented as an argument
        #chrs.extend([23,26]) # chromosomes X and mito
        for chrom in  chrs:
            print('chromosome is ' + str(chrom))
            #pdb.set_trace()
            
            ### 1a. Get data from 'data' ###
            data = SocialData(in_file, phenos_version, covs_version, cage_version, dam_version, GRM_version, subset, chrom)
            doto = data.get_data(selected_pheno) # if int, NOT YET in python 0-indexing
            assert trait1 == doto['trait'], "something wrong in parsing data, doto['trait'] and trait1 do not correspond"
            #trait=doto['trait']
            print("trait in null_covars_LOCO is " + trait1)
            
            #out_dir=args['out']
            #if out_dir is None:
            #    out_dir = os.getcwd()
            #covar_outfile_dir = "".join([out_dir,"/null_covars_LOCO/",phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/',trait,'/'])
            #if not os.path.exists(covar_outfile_dir):
            #    os.makedirs(covar_outfile_dir, exist_ok=True) # exist_ok=True should avoid problem in case it already exists
            
            ### 1b. Define output file name ###                
            covar_outfile_name="".join([outfile_dir, trait1, '_chr',str(chrom),'.h5'])
            #comment out if decide to overwrite
            if os.path.exists(covar_outfile_name):
                continue
            print(covar_outfile_name)
            
            ### 2. Get covariance matrix ###
            try_nb_VD = 1
            #while try_nb_VD < 6:
            if 0:
                print('null_covars_LOCO covar')

                vc = DirIndirVD(pheno = doto['pheno'], pheno_ID = doto['pheno_ID'], covs = doto['covs'], covs_ID = doto['covs_ID'], covariates_names = doto['covariates_names'], kinship_all = doto['kinship_full'], kinship_all_ID = doto['kinship_full_ID'],  cage_all = doto['cage_full'], cage_all_ID = doto['cage_full_ID'], maternal_all = doto['maternal_full'], maternal_all_ID = doto['maternal_full_ID'], subset_IDs = doto['subset_IDs'], 
                                DGE = (DGE is not None), IGE = (IGE is not None), IEE = (IEE is not None), cageEffect = (cageEffect is not None), maternalEffect = (maternalEffect is not None), 
                                calc_ste=False, subset_on_cage = False, SimplifNonIdableEnvs = False, vc_init_type = None, vc_init = None, seed = sid, standardize_pheno = True)
                dirIndir_out = vc.getOutput()
                
                try_nb_VD = try_nb_VD + 1
                # will check if vc was successfull, if not, will try again
                if "optim_error" in dirIndir_out.keys():
                    assert dirIndir_out['optim_error'] is not None, "optimization error is None, when expected to be something"
                    if try_nb_VD < 6:
                        print("Warning: optimization was not successful, gonna try for the ", try_nb_VD , " /5 time")
                    else:
                        print("Warning: optimization was not successful after 5 tries")
                        break
                    sid = sid+1   # SEED_test: change number for next test
                else:
                    print("Null covar optimization for chr", chrom,"successfull!")
                    break
              
            ### 3. Save output ###
            # will save Infos whether optimization successful (after 5 tries) or not
            toSave_file = h5py.File(covar_outfile_name,'w')
            toSaveInfos = vc.getToSaveInfos()
            toSave_file.create_dataset(name = 'sampleID',data = toSaveInfos['sampleID'])
            toSave_file.create_dataset(name = 'pheno',data = toSaveInfos['pheno'])
            toSave_file.create_dataset(name = 'covs',data = toSaveInfos['covs'])
            # will save Covar if optimization was successful
            if "optim_error" in dirIndir_out.keys():
            	#here need to return a warning to let the user know but nothing more as we don't want the pipeline to fail in case optim was not successful
            	print("Warning: after 5 tries optimization was still not successful; have still saved sampleID, pheno and covs")				
            else:
            	toSaveCovar = vc.getToSaveCovar()
            	toSave_file.create_dataset(name = 'covar_mat',data = toSaveCovar['covar_mat'])
            
            toSave_file.close()
            gc.collect()


print("My program took", time.time() - start_time, "to run")




#############################
#if chrs is None:
#    print("chrs is ", chrs)
#else:
#    for chrom in chrs:
#        print("chrs is ", chrom)

