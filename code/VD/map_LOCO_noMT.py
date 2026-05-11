#!/usr/bin/env python
#mem usage 20GB

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
parser.add_argument('--directSNP_v', help= 'type of GENOTYPES, SNPs, i.e. name of subgroup of `genotypes`', required=True)
parser.add_argument('--covarDir', help= 'general directory where null covariance matrices are saved, ', required=True)

# Optional arguments
parser.add_argument('-o','--out', help = 'path to folder in which create output folders and files [default="."]', default=".")
parser.add_argument('-C','--chrom', type = str, help = 'comma-separated string of chromosomes or list ')
parser.add_argument('-p','--pheno', type=int_or_str, help = 'Number of phenotypes col to analyse, or line to combins file with pair of phenotypes in case of bivar [default=1]', default=1)
parser.add_argument('-m','--model', help='type of model: univariate (uni) or bivariate (bi) [default=uni]', default="uni")
parser.add_argument('-e','--effects',help='effects - DGE,IGE,IEE,cageEffect,maternalEffect - to be included [default= None to all]', default=None)
parser.add_argument('-g','--gwas', help='type of GWAS: dgeGWAS (dGWAS) or igeGWAS(iGWAS) [default=dGWAS]', default="dGWAS")
parser.add_argument('--socialSNP_v', help= 'type of SOCIAL GENOTYPES, SNPs, i.e. name of subgroup of `genotypes`', default=None)
parser.add_argument('-s','--subset', help='to handle optional subset on individuals [default=None]', default=None)


args = vars(parser.parse_args())
#args['input'] 
#args['phenos_v'] # e.g. phenos_version
#args['covs_v'] 
#args['cage_v'] 
#args['dam_v'] 
#args['grm_v'] 
#args['snp_v'] 
#args['covarDir']
#
#args['subset']
#args['model']
#args['pheno']
#args['effects']
#args['chrom'] 
#args['out']
#-------------------------#
#  II. IMPORTING MODULES
#-------------------------#	 

import sys
import os
import h5py
import numpy as np
#sys.path.insert(0,'/users/abaud/abaud/P50_HSrats/code/variance_decomposition/felipes_deblur/VDessential_wBivarPvalues/pysrc/')
from classes.social_data import SocialData # if modifying this file, always update the latest version as
#sys.path.insert(0,'/users/abaud/abaud/CFW/code/reproduce/LIMIX')
# runs GWAS without estimating the covariance structure: uses the total covariance matrix provided (built from DGE SGE DEE SEE and cage effects)
import pdb
import copy
import gc

from limix_core.gp import GP2KronSum
from limix_core.covar import FreeFormCov
from limix_lmm import LMM # need pip install limix-lmm



if __name__=='__main__':
    ##################################################################
    ######################   A. GET ARGUMENTS   ######################
    ##################################################################
    #---------------------------------------------------------#
    #	       A1. VERSIONS FOR INPUT FILE = *.h5 file		  
    #---------------------------------------------------------#
    #in_file = '/users/abaud/abaud/P50_HSrats/data/P50_rats_Rn7_Helene.h5'
    in_file=args['input'] # input file name
    # first fetch trait name
    #task = 'deblur_counts'
    #pheno_version=task
    phenos_version=args['phenos_v'] # e.g. phenos_version = 'simulations'
    
    #covs_version = None
    #cage_version = 'cage_TNfixed2'
    #dam_version = 'dams'
    covs_version=args['covs_v'] 
    if covs_version == "None":
      covs_version = None
    cage_version=args['cage_v'] 
    dam_version=args['dam_v'] 
    if dam_version == "None":
      dam_version = None

    #kinship_type = sys.argv[2]
    #GRM_version = kinship_type
    GRM_version = args['grm_v'] # e.g. GRM_version = 'Andres_kinship'
    direct_version= args['directSNP_v'] 
    social_version= args['socialSNP_v']
    if social_version == "None":
      social_version = None

    # directory where saved covar from null_covar... analysis
    covar_mainDir = args['covarDir']

    #-----------------------------------------------------------------#
    #  A2. SUBSET, if any - to handle optional subset on individuals
    #-----------------------------------------------------------------#	 
    #if len(sys.argv) == 4:
    #    subset = sys.argv[3]
    #else:
    #    subset = None
    subset=args['subset']
    if subset == "None":
      subset = None
    print("Subset: ",  subset)

    #---------------------------------------------------------#
    #				           A3. MODEL      						  
    #---------------------------------------------------------#
    # TODO: Not sure this is useful, are we doing GWAS bivariate?
    m = args['model']
    if m == "bivariate" or m == "bi":
        sys.exit("can't do GWAS bivariate yet")
        #model = "bi"
        #print("model is BIVARIATE: ", model)
    
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
    #then fetch pickled object # from amelie:
    #DGE = "DGE"
    #IGE = None
    #IEE = None
    #cageEffect = "cageEffect"
    #maternalEffect = "maternalEffect"

    effects = [DGE, IGE, IEE, cageEffect, maternalEffect]

    if args['effects'] is not None:
        effects = args['effects'].split(",")
    
    if "IGE" in effects:
        IGE = "IGE"
        IEE = "IEE"

    if "IEE" in effects:
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
    #	                  A5. TYPE OF GWAS    	  
    #---------------------------------------------------------#

    g = args['gwas']
    if g == "dgeGWAS" or g == "dGWAS":
        gwas = "dGWAS"
    elif g == "igeGWAS" or g == "iGWAS":
        gwas = "iGWAS"
    
    print("doing GWAS: ", gwas)

    #---------------------------------------------------------#
    #	     A6. GETTING PHENOTYPE NAME or COLUMN NUMBER    	  
    #---------------------------------------------------------#
    # TODO: for now leaving option about model commented 
    if model == "bi":
        sys.exit("can't do GWAS bivariate yet")
        #assert args['combins_path'] is not None, 'Need combins file'
        #assert isinstance(args['pheno'], int), 'We are in bivariate, need the line of combins file to analyse'
        ## reading combins file
        #dfcombins = pd.read_csv(args['combins_path'], header=None, sep=',')
        ##this works with combins file with pheno_names or numbers
        #selected_pheno, selected_pheno_MT = dfcombins.iloc[args['pheno'] - 1] 
        #selected_pheno = int_or_str(selected_pheno) # if int, NOT YET in python 0-indexing
        #selected_pheno_MT = int_or_str(selected_pheno_MT) # if int, NOT YET in python 0-indexing
        
    elif model == "uni": #or more simply for UNIVARIATE # move if uncomment, indent next 3 lines 
        selected_pheno = args['pheno'] # if int, NOT YET in python 0-indexing
        selected_pheno = int_or_str(selected_pheno)
        selected_pheno_MT = None

    #-----------------------------------------------------------------#
    #      A7. DEFINE list of chromosomes 
    #-----------------------------------------------------------------#    
    assert args['chrom'] is not None and len(args['chrom']) > 0, "need at least one chromosome for null_covars_LOCO"
    chrs = []
    for item in args['chrom'].split(","):
        if "-" in item:
            start, end = map(int, item.split("-"))
            chrs.extend(range(start, end+1))
        else:
            chrs.append(int(item))
    
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
    data = SocialData(in_file = in_file, phenos_version = phenos_version, covs_version = covs_version, cage_version = cage_version, dam_version = dam_version, GRM_version = GRM_version, sex_version = None, subset = subset, chrom = chrs[0])
    #data = SocialData(in_file, phenos_version, covs_version, cage_version, dam_version, GRM_version, subset, chrom=chrs[0]) 
    traits = data.get_trait(selected_pheno, selected_pheno_MT)
    trait1 = traits['trait']
    trait2 = traits['trait_MT'] # None if univariate
    # then will assert if this corresponds to 'trait' and 'trait_MT' of doto = data.get_data(selected_pheno, selected_pheno_MT); 
    print("traits are " + " and ".join(filter(None, [trait1,trait2])))

    ##chr below could be any - only used to get trait
    ##data = SocialData(task,kinship_type,subset,1)
    #data = SocialData(in_file, phenos_version, covs_version, cage_version, dam_version, GRM_version, subset) 
    #selected_pheno=int(sys.argv[1])
    #doto = data.get_data(selected_pheno) # if int, NOT YET in python 0-indexing
    #print("Different covariates actually used")
    #trait = doto['trait']
    #print(trait)
    
    #pdb.set_trace()
    #-----------------------------------------------------------------#
    #					       B2. DEFINING OUTPUT DIR
    #-----------------------------------------------------------------#	 
    ## NB: 1. cannot go before parsing input because trait1 and trait2 are output of SocialData.get_data
    ##     2. if not existing, CREATE A NEW DIR inside the dir given to --out option
    out_dir=args['out']
    if out_dir is None:
        out_dir = os.getcwd()
        
    if model == "bi": #e.g. out_dir = "/users/abaud/htonnele/CFW/output/" "pvalues_LOCO/univariate/dosages_pruned/noBatch/dosages_pruned_{effs}/"
        sys.exit("can't do GWAS bivariate yet")
        # TODO: have to check the one below
        #pvalues_file_dir = "".join([out_dir,"/pvalues_LOCO/",model,"variate/",genotypes_version,"/",phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, cageEffect, maternalEffect])),'/',trait1,'/'])
    elif model == "uni": # TODO: insert some 
        #pdb.set_trace()
        pvalues_file_dir = "".join([out_dir,"/pvalues_LOCO/",model,"variate/",".".join(filter(None,['',direct_version,social_version])),"/",phenos_version,"/",gwas,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, IEE, cageEffect, maternalEffect])),'/'])
        #pvalues_file_dir = "".join(['/users/abaud/abaud/P50_HSrats/output/pvalues_LOCO_unpruned/',task,"/",kinship_type,"_","_".join(filter(None, [subset, DGE, IGE, IEE, cageEffect,maternalEffect]))])
    if not os.path.exists(pvalues_file_dir):
        os.makedirs(pvalues_file_dir, exist_ok=True)
    pvalues_file_name = "".join([pvalues_file_dir,'/',trait1,'.h5'])
    if os.path.exists(pvalues_file_name): # commented for testing
        print("".join(["file ", pvalues_file_name, " already exists, exiting"]))
        sys.exit(0)
        
    ##try opening pvalues file early so that lmm doesnt run if file opening is going to fail...
    #pvalues_file_dir = "".join(['/users/abaud/abaud/P50_HSrats/output/pvalues_LOCO_unpruned/',task,"/",kinship_type,"_","_".join(filter(None, [subset, DGE, IGE, IEE, cageEffect,maternalEffect]))])
    #if not os.path.exists(pvalues_file_dir):
    #    os.makedirs(pvalues_file_dir, exist_ok=True)
    #pvalues_file_name = "".join([pvalues_file_dir,'/',trait,'.h5'])
    ##if os.path.exists(pvalues_file_name):
    ##   sys.exit(0)
    #pvalues_file = h5py.File(pvalues_file_name,'w')
    
    #e.g. covar_mainDir = "test/null_covars_LOCO/"
    covar_dir = "".join([covar_mainDir,"/",model,"variate/",phenos_version,"/",GRM_version,"_","_".join(filter(None, ['',subset, DGE, IGE, IEE, cageEffect, maternalEffect])),'/',trait1])
    if not os.path.exists(pvalues_file_dir):
        sys.exit("cannot find covar dir " + covar_dir)
    
    #covar_outfile_dir = "".join(["/users/abaud/abaud/P50_HSrats/output/null_covars_LOCO/",task,"/",kinship_type,"_","_".join(filter(None, [subset, DGE, IGE, IEE, cageEffect,maternalEffect]))])
    
    #genotype_file_name = '/users/abaud/abaud/P50_HSrats/data/P50_rats_Rn7.h5' # this became that is in in_file
    #genotype_file = h5py.File(genotype_file_name,'r')
    
    #will define one LMM per chr as covar changes for each cov
    #chrs = list(range(1,21))
    #chrs.extend([23,26]) # chromosomes X and mito
    
    pvalues_file = h5py.File(pvalues_file_name,'w')
    
    ### Loading sampleID pheno and covs, which are the same for all chromosomes, from one covar file ###
    chr = chrs[0]
    covar_file_name = "".join([covar_dir,"/",trait1,"_chr",str(chr),".h5"])
    covar_file = h5py.File(covar_file_name,'r')
    saved = {}
    #pdb.set_trace()
    saved['sampleID'] = covar_file['sampleID'].asstr()[:] # this are depending on sample size of the phenotype!
    saved['pheno'] = covar_file['pheno'][:]
    saved['covs'] = covar_file['covs'][:]
    #also get the covar_mat since the file is open - will have it already for chr1
    covar_mat = covar_file['covar_mat'][:] #saved['covar_mat'] = covar_file['covar_mat'][:] # was like this
    covar_file.close()
    
    ##import genotypes. direct and social geno already matched on rows and cols
    #genotype_file = h5py.File(in_file,'r')
    #geno = genotype_file['genotypes'][direct_version]
    #geno_matrix = geno['matrix'][:].T
    #geno_sample_ID = geno['row_header']['sample_ID'].asstr()[:]
    #
    #social_geno = genotype_file['genotypes'][social_version]
    #social_geno_matrix = social_geno['matrix'][:].T
    #social_geno_sample_ID = social_geno['row_header']['sample_ID'].asstr()[:]
    #
    #position = {
    #"chr" : geno['col_header']['chr'][:].astype(int),
    #"pos" : geno['col_header']['pos'][:].astype(int),
    #}
    #
    ##pdb.set_trace()
    #social_position = {
    #"chr" : social_geno['col_header']['chr'][:].astype(int),
    #"pos" : social_geno['col_header']['pos'][:].astype(int),
    #}
    #assert all(position['chr'] == social_position['chr']), "'chr' in direct and social genotypes don't correspond"
    #assert all(position['pos'] == social_position['pos']), "'pos' in direct and social genotypes don't correspond"
    #del social_position # don't need it, will use position for both
    #
    #genotype_file.close()
    
    ##match direct and social genotypes - necessary because TN rats not included in social genotypes table
    #Imatch0 = np.nonzero(geno_sample_ID[:,np.newaxis]==social_geno_sample_ID)
    #geno_sample_ID = geno_sample_ID[Imatch0[0]]
    #geno_matrix = geno_matrix[Imatch0[0],:]
    #social_geno_sample_ID = social_geno_sample_ID[Imatch0[1]]
    #social_geno_matrix = social_geno_matrix[Imatch0[1],:]
    
    
    ##match genotypes with the rest
    #Imatch = np.nonzero(saved['sampleID'][:,np.newaxis]==geno_sample_ID)
    #print("Imatch"+str(len(Imatch[0])))
    #saved['sampleID'] = saved['sampleID'][Imatch[0]]
    #saved['pheno'] = saved['pheno'][Imatch[0]]
    #saved['covs'] = saved['covs'][Imatch[0],:]
    ##matching of covar_mat will be done in for loop below
    #geno_matrix = geno_matrix[Imatch[1],:]
    #geno_sample_ID=geno_sample_ID[Imatch[1]]
    #social_geno_matrix = social_geno_matrix[Imatch[1],:]
    ##save the genotype matrices as will subset them later by chr
    #saved_geno_matrix = geno_matrix
    #saved_social_geno_matrix = social_geno_matrix
    #
    ##add chr and pos of all SNPs to output file (NOT by chr)
    #pvalues_file.create_dataset(name = 'chr',data = position['chr'])
    #pvalues_file.create_dataset(name = 'pos',data = position['pos'])


    for chr in chrs:
        #pdb.set_trace()
        print('chr is ' + str(chr))
        chrgroup = pvalues_file.create_group("".join(["chr",str(chr)])) 
        
        #import genotypes. direct and social geno already matched on rows and cols
        genotype_file = h5py.File(in_file,'r')
        geno = genotype_file['genotypes'][direct_version]["".join(['chr',str(chr)])]
        geno_matrix = geno['matrix'][:].T
        geno_sample_ID = geno['row_header']['sample_ID'].asstr()[:]

        position = {
        "chr" : geno['col_header']['chr'][:].astype(int),
        "pos" : geno['col_header']['pos'][:].astype(int),
        }
        
        if "IGE" in effects:
            assert social_version is not None, 'need the version of social genotypes when doing igeGWAS'
            social_geno = genotype_file['genotypes'][social_version]["".join(['chr',str(chr)])]
            social_geno_matrix = social_geno['matrix'][:].T
            social_geno_sample_ID = social_geno['row_header']['sample_ID'].asstr()[:]
          
            #pdb.set_trace()
            social_position = {
            "chr" : social_geno['col_header']['chr'][:].astype(int),
            "pos" : social_geno['col_header']['pos'][:].astype(int),
            }
            assert all(position['chr'] == social_position['chr']), "'chr' in direct and social genotypes don't correspond"
            assert all(position['pos'] == social_position['pos']), "'pos' in direct and social genotypes don't correspond"
            del social_position # don't need it, will use position for both
        
        genotype_file.close()
        
        #pdb.set_trace()
        #match direct and social genotypes - necessary because TN rats not included in social genotypes table
        if "IGE" in effects:
            Imatch0 = np.nonzero(geno_sample_ID[:,np.newaxis]==social_geno_sample_ID)
            geno_sample_ID = geno_sample_ID[Imatch0[0]]
            geno_matrix = geno_matrix[Imatch0[0],:]
            social_geno_sample_ID = social_geno_sample_ID[Imatch0[1]]
            social_geno_matrix = social_geno_matrix[Imatch0[1],:]
        
        
        #match genotypes with the rest
        Imatch = np.nonzero(saved['sampleID'][:,np.newaxis]==geno_sample_ID) 
        print("Imatch " + str(len(Imatch[0])))
        saved['sampleID'] = saved['sampleID'][Imatch[0]]
        saved['pheno'] = saved['pheno'][Imatch[0]]
        saved['covs'] = saved['covs'][Imatch[0],:]
        #matching of covar_mat will be done in for loop below
        geno_matrix = geno_matrix[Imatch[1],:]
        geno_sample_ID=geno_sample_ID[Imatch[1]]
        if "IGE" in effects:
            social_geno_matrix = social_geno_matrix[Imatch[1],:]
        #save the genotype matrices as will subset them later by chr
        saved_geno_matrix = geno_matrix
        if "IGE" in effects:
            saved_social_geno_matrix = social_geno_matrix
        
        #pdb.set_trace()
        #for chr 1 we already have covar_mat but get it for other chrs
        if chr != 1:
            print("getting covar matrix")
            covar_file_name = "".join([covar_dir,"/",trait1,"_chr",str(chr),".h5"])
            covar_file = h5py.File(covar_file_name,'r')
            covar_mat = covar_file['covar_mat'][:]
            covar_file.close()
        else:
            print("covar matrix for chr"+str(chr)+" already loaded")
    
        #we have not yet matched covar_mat (not for chr1 nor for the others)
        K = covar_mat[Imatch[0],:][:,Imatch[0]]
        
        #define the model
        Cg = FreeFormCov(1)
        Cn = FreeFormCov(1)
        A = np.eye(1)
        gp = GP2KronSum(Y=saved['pheno'], F=saved['covs'], A=A, Cg=Cg, Cn=Cn, R=K)    
        gp.covar.Cg.setCovariance(0.5 * np.ones([1, 1]))
        gp.covar.Cn.setCovariance(0.000001 * np.ones([1, 1]))
    
        #optimize for the first time - that's the null model
        gp.optimize()
        # pheno arg is y; covs arg is F; 3rd arg is Ki_dot as per https://github.com/limix/limix-lmm/blob/master/limix_lmm/lmm.py; set to covar.solve following Paolo's instructions :) will check results match published
        #set the LMM - to be honest not sure why we have to do gp and LMM in two steps
        lmm = LMM(saved['pheno'], saved['covs'], gp.covar.solve)
        #now that LMM is set in the future can only update the parts we want; pheno will stay as is
    
        #get ready for conditional analysis, if fitting DGE of SNP when testing IGE of SNP
        if gwas == "dGWAS":
          if "IGE" in effects: # testing DGE, conditioning with IGE 
              cond_geno_matrix = saved_social_geno_matrix[:,position['chr']==chr]
              test_geno_matrix = saved_geno_matrix[:,position['chr']==chr]
          else:
              test_geno_matrix = saved_geno_matrix[:,position['chr']==chr]
        elif gwas == "iGWAS": # testing IGE, conditioning with DGE 
            cond_geno_matrix = saved_geno_matrix[:,position['chr']==chr]
            test_geno_matrix = saved_social_geno_matrix[:,position['chr']==chr]

        #geno_matrix = saved_geno_matrix[:,position['chr']==chr]
        #social_geno_matrix = saved_social_geno_matrix[:,position['chr']==chr]
        
        # when doing gwas with IGE included need to condition on social/direct genotypes
        if "IGE" in effects: 
            print('adding social or direct genotypes as covariate')
            #need to declare pvalues and betas here and iterate through each SNP below because set of covariates is different for each SNP (direct genotypes change)
            #pvalues = [-1] * social_geno_matrix.shape[1]
            #betas = [-1] * social_geno_matrix.shape[1]
            pvalues = [-1] * test_geno_matrix.shape[1]
            betas = [-1] * test_geno_matrix.shape[1]
            for i in range(test_geno_matrix.shape[1]):
                #start by adding conditional genotype to experimental covariates
                covs = np.concatenate([saved['covs'], cond_geno_matrix[:,i:i+1]], axis=1)
                #update the null model with the new covariates
                lmm.F = covs
                lmm._fit_null()
                #retrieve genotypes of tested SNP
                snp = test_geno_matrix[:,i:i+1]
                #fit LMM
                try:
                    lmm.process(snp)
                except:
                    print('caught error on this chr')
                    continue
                
                #retrieve P value and beta for output
                #pvalues[i] = lmm.getPv()[0]
                ##betas[i] = lmm.getBetaSNP()[0]
                pvalues[i] = lmm.getPv()[0]
                betas[i] = lmm.getBetaSNP()[0]
        else:
            print('IGE is', IGE, 'proceeding to get pvalues and betas')
            try:
                lmm.process(geno_matrix)
            except:
                print('caught error on this chr')
                continue
            pvalues = lmm.getPv()
            betas = lmm.getBetaSNP()

        #pdb.set_trace()

        #once have looped through all SNPs on chr add to H5 file - one list element per chr
        chrgroup.create_dataset(name = "".join(['pvalues_chr',str(chr)]), data = pvalues)
        chrgroup.create_dataset(name = "".join(['chrs_chr',str(chr)]), data = position['chr'])
        chrgroup.create_dataset(name = "".join(['poss_chr',str(chr)]), data = position['pos'])
        chrgroup.create_dataset(name = "".join(['betas_chr',str(chr)]), data = betas)

        #once have looped through all SNPs on chr add to H5 file - one list element per chr
        #pvalues_file.create_dataset(name = "".join(['pvalues_chr',str(chr)]),data = pvalues)
        #pvalues_file.create_dataset(name = "".join(['betas_chr',str(chr)]),data = betas)
        gc.collect()
    
    pvalues_file.close()

print('allgood')

    ### This is how it was done with only conditioning option
    #for chr in chrs:
    #    print('chr is ' + str(chr))
    #    
    #    genotype_file = h5py.File(in_file,'r')
    #    geno = genotype_file['genotypes'][direct_version]["".join(['chr',str(chr)])]
    #    geno_matrix = geno['matrix'][:].T
    #    geno_sample_ID = geno['row_header']['sample_ID'][:]
    #    # social genotypes
    #    social_geno = genotype_file['genotypes'][social_version]["".join(['chr',str(chr)])]
    #    social_geno_matrix = social_geno['matrix'][:].T
    #    social_geno_sample_ID = geno['row_header']['sample_ID'][:]
    #    
    #    position = {
    #        "chr" : geno['col_header']['chr'][:],
    #        "pos"   : geno['col_header']['pos'][:]
    #    }
    #    genotype_file.close()
    #    
    #    #match genotypes with the rest
    #    pdb.set_trace()
    #    Imatch = np.nonzero(saved['sampleID'][:,np.newaxis]==geno_sample_ID)
    #    print("Number of individuals in sampleID and with genotypes: " + str(len(Imatch[0])))
    #    saved['sampleID'] = saved['sampleID'][Imatch[0]]
    #    saved['pheno'] = saved['pheno'][Imatch[0]]
    #    saved['covs'] = saved['covs'][Imatch[0],:]
    #    saved['covar_mat'] = saved['covar_mat'][Imatch[0],:][:,Imatch[0]]
    #    geno_matrix = geno_matrix[Imatch[1],:]
    #    
    #    geno_sample_ID=geno_sample_ID[Imatch[1]]
    #    social_geno_matrix = social_geno_matrix[Imatch[1],:]
    #
    #    K = saved['covar_mat']
    #    Cg = FreeFormCov(1)
    #    Cn = FreeFormCov(1)
    #    A = np.eye(1)
    # 
    #    gp = GP2KronSum(Y=saved['pheno'], F=saved['covs'], A=A, Cg=Cg, Cn=Cn, R=K)    
    #    gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
    #    gp.covar.Cn.setCovariance(0.000001 * np.ones((1, 1)))
    #    info_opt = gp.optimize(verbose=False)
    #
    #    lmm = LMM(saved['pheno'], saved['covs'], gp.covar.solve)
    #
    #    #print('additional noise VC is ')
    #    #print(Cn.K())
    #    try:
    #        lmm.process(geno_matrix)
    #        pvalues = lmm.getPv()
    #        betas = lmm.getBetaSNP()
    #        pvalues_file.create_dataset(name = "".join(['pvalues_chr',str(chr)]),data = pvalues)
    #        pvalues_file.create_dataset(name = "".join(['chrs_chr',str(chr)]),data = position['chr'])
    #        pvalues_file.create_dataset(name = "".join(['poss_chr',str(chr)]),data = position['pos'])
    #        pvalues_file.create_dataset(name = "".join(['betas_chr',str(chr)]),data = betas)
    #    except:
    #        print('caught error on this chr')
    #        continue
    #
    #    gc.collect()
    #
    #pvalues_file.close()
