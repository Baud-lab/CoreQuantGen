#in_file should be the only thing you need to change if you have followed my H5 file conventions - which you can guess from below
# PHENOTYPE are saved in the h5 element that has the same name as "task"
# <task>$row_header$sample_ID = sample ID of ind in phenotype
# <task>$col_header$phenotype_ID = phenotype names
# <task>$col_header$covariatesUsed = covariates used for each phenotype; commasep string
# <task>$matrix = actual matrix with values

# COVARIATES are in h5 in element called "covariates" - if different either change the script or your file
# covariates$row_header$sample_ID
# covariates$col_header$covariate_ID
# covariates$matrix = actual matrix with values

# SUBSETS_IDs - in h5 as "subsets" - if different either change the script or your file


import numpy as np
import h5py
#re for regular expressions
import re
import pdb

class SocialData():
    
    def __init__(self, in_file=None, phenos_version = None,covs_version=None, cage_version=None, dam_version = None, GRM_version = None, subset = None, chrom = None):
        #data = SocialData(in_file,task,covs_task,cage_task,kinship_type,subset) # order in exp_bivar impo!
        assert in_file is not None, 'Give me an input!'
        assert phenos_version is not None, 'Specify phenotype type! (phenos_version)'
        #assert covs_version is not None, 'Specify covariate type! (covs_version)' # This has been deprecated since, it is possible not to have covariates
        assert cage_version is not None, 'Specify cage type! (cage_version)'
        assert GRM_version is not None, 'Specify kinship type!' 
        
        self.phenos_version=phenos_version #this is tied to the type of phenotypes used and the normalisation procedure. 
        
        self.covs_version=covs_version #this is tied to the type of covs used and the normalisation procedure.
        self.cage_version=cage_version 
        self.dam_version=dam_version 
        self.in_file=in_file 
        self.GRM_version=GRM_version #GRM_version is based on the set of genotypes used (e.g. pruned round 8, pruned round 2 or unpruned round 1). this determines the GRM, the GRM_LOCO (and also the genotype table used in GWAS although for some reason this is read separately in teh GWAS code...hm hm)
        
        self.subset = subset # default is None -> i.e. all 
        self.chrom = chrom
        self.load()
    
    def load(self):
        
        #in_file = '/users/abaud/abaud/HSmice/data/HSmice_wPleth_REVISIONS.hdf5'
        #in_file = '/users/abaud/htonnele/HSmice/output/HSmice_univar_1200_IDn_seed90.h5'
        #in_file = '/users/abaud/htonnele/HSmice/output/HSmice_univar_sub50cages_1200_IDn_seed90.h5'
        print("Reading data from ", self.in_file)
        
        #f = h5py.File(in_file,'r')
        f = h5py.File(self.in_file,'r')
        
        #pdb.set_trace()
        # get phenotypes
        self.all_pheno = f['phenotypes'][self.phenos_version]['matrix'][:].T
        self.measures = f['phenotypes'][self.phenos_version]['col_header']['phenotype_ID'].asstr()[:] # as string
        self.pheno_ID = f['phenotypes'][self.phenos_version]['row_header']['sample_ID'].asstr()[:]

        # get covariates - NB: saved in 'covariates' element in h5
        if self.covs_version is None:
            print("'covs_version' set to None, all_covs2use to None, not using covariates")
            self.all_covs2use = None
        elif 'covariates' not in f.keys():
            print("No covariates in .h5, all_covs2use to None")
            self.all_covs2use = None
        else:
            self.all_covs2use = f['phenotypes'][self.phenos_version]['col_header']['covariatesUsed'].asstr()[:] 
            self.all_covs = f['covariates'][self.covs_version]['matrix'][:].T
            self.covariates = f['covariates'][self.covs_version]['col_header']['covariate_ID'].asstr()[:]
            self.covs_ID = f['covariates'][self.covs_version]['row_header']['sample_ID'].asstr()[:]
        
        # get cages - NB: we decided to dissociate information about the cage from information about the phenotypes
        self.cage_full = f['cages'][self.cage_version]['array'].asstr()[:] #self
        self.cage_full_ID = f['cages'][self.cage_version]['sample_ID'].asstr()[:]
        print('theres cage info in HDF5')
        
        # get maternal - dam 
        if self.dam_version is None:
            print("'dam_version' and/or 'maternalEffect' is None, not including Maternal effects")
            self.maternal_full = None
            self.maternal_full_ID = None
        else:
            self.maternal_full = f['dam'][self.dam_version]['array'].asstr()[:]
            self.maternal_full_ID = f['dam'][self.dam_version]['sample_ID'].asstr()[:]
            print('theres dam info in HDF5, including maternal effect')

        if len(self.all_pheno.shape)==1: #very unlikely given highly dimensional datasets we use but still...
            self.all_pheno = self.all_pheno[:,np.newaxis]

        if self.all_covs2use is not None:
            if len(self.all_covs.shape)==1: #very unlikely given highly dimensional datasets we use but still...
                self.all_covs = self.all_covs[:,np.newaxis]


        if self.chrom is not None: 
            print('social data in LOCO')
            #DO NOT swap [self.GRM_version] to before any GRM thing like below actually - no idea on what it means Helene 17/05/24
            self.kinship_full = f['GRMs_LOCO'][self.GRM_version][''.join(['chr',str(self.chrom)])]['matrix'][:]
            self.kinship_full_ID = f['GRMs_LOCO'][self.GRM_version][''.join(['chr',str(self.chrom)])]['row_header']['sample_ID'].asstr()[:]
        else:    
            self.kinship_full = f['GRM'][self.GRM_version]['matrix'][:]
            self.kinship_full_ID = f['GRM'][self.GRM_version]['row_header']['sample_ID'].asstr()[:]

        if self.subset is None:
            self.subset_IDs = self.kinship_full_ID
        else:
            self.subset_IDs = f['subsets'][self.subset].asstr()[:]
            
    def get_trait(self, selected_pheno, selected_pheno_MT = None):  #col,col_MT = None):
        #pdb.set_trace()
        # here define 'col', i.e. the column corresponding to selected_pheno
        if isinstance(selected_pheno, int):
            col = selected_pheno - 1 # Python 0-indexing
        else:
            col = np.where(self.measures == selected_pheno)[0] # This will output a number, index as in python
            assert col.size > 0, print(f"The phenotype '{selected_pheno}' doesn't exist")
            col = col[0]
        
        # here define 'col_MT', i.e. the column corresponding to selected_pheno_MT
        if isinstance(selected_pheno_MT, int):
            col_MT = selected_pheno_MT - 1 # Python 0-indexing
        elif isinstance(selected_pheno_MT, str):
            col_MT = np.where(self.measures == selected_pheno_MT)[0]
            assert col_MT.size > 0, print(f"The phenotype '{selected_pheno_MT}' doesn't exist")
            col_MT = col_MT[0]
        else:
            col_MT = None
        
        # defining trait
        self.trait = self.measures[col]
        
        if col_MT is None:
            #self.pheno = self.all_pheno[:,[col]] #using array as index permits keeping the dimensionality kind of
            self.trait_MT = None
        else:
            self.trait_MT = self.measures[col_MT]
            #self.pheno = np.concatenate([self.all_pheno[:,[col]], self.all_pheno[:,[col_MT]]],0) # now pheno is 2N while pheno_ID is still N

        return {'col': col,
                'col_MT': col_MT,
                'trait' : self.trait,
                'trait_MT' : self.trait_MT}

    
    def get_data(self, selected_pheno, selected_pheno_MT = None): #col,col_MT = None):
        #pdb.set_trace()
        
        cols = self.get_trait(selected_pheno, selected_pheno_MT)
        col= cols['col']
        col_MT = cols['col_MT']
        
        self.trait = cols['trait'] 
        self.trait_MT = cols['trait_MT'] # None in univariate

        if col_MT is None:
            self.pheno = self.all_pheno[:,[col]] #using array as index permits keeping the dimensionality kind of
        else:
            self.pheno = np.concatenate([self.all_pheno[:,[col]], self.all_pheno[:,[col_MT]]],0) # now pheno is 2N while pheno_ID is still N

        #that's if no covs in entire study
        if self.all_covs2use is None:
            self.covs_ID = None # This is defined above if there's 'covariates' group - here set to None 
            self.covs = None
            covariates_names = []
          
        else: #at this point it is possible trait1 and/or trait2 have empty covs2use
            covs2use = self.all_covs2use[col].split(',')
            Ic = np.zeros(self.covariates.shape[0],dtype=bool) 
            for cov in covs2use:
                if cov != '':
                    assert any(self.covariates==cov), 'covariate not in cov table'
                    Ic = np.logical_or(Ic,self.covariates==cov)
            
            covariates_names = self.covariates[Ic] # covariates_names will be empty list rather than None if no cov for that phenotype (col)
            print('Initial covs (for first phenotype) in social_data are ' + str(covariates_names))
            if any(Ic): #Ic has only 1 value as len is 1 # FOR AMELIE: This is not the case if covs are more than 1 - can we remove this (and at line 150 comment?)
                self.covs = self.all_covs[:,Ic] #self.covs is covs for phenotype in column col; since Ic created as an array no need for [Ic]
            else:
                self.covs = None
            
            if col_MT is not None: #should only go there is there are covs in the study
                #at this point covs is None or sthg; covs_ID is sthg; covariates_names is (potentially null) array        
                #FOR AMELIE: Does this refers to what is done with the traitspe_mean in dirIndirVD and can remove these comments?
                ## ???all intercepts now added in dirIndirVD??? not sure I'll check what that comment means
                ##add col for intercept then extend by pasting at the bottom a matrix of 0s of size that of the cov matrix (which has original row number and specific col number)
                ##no need for similar intercept for second pheno as would be linearly dependent
                
                if self.covs is not None:
                    self.covs = np.append(self.covs,np.zeros(self.covs.shape),0)
                #so for case where phenotype in col has no covs2use we now have a column of 1 then 0 of length 2xN
                #self.covs_ID = np.concatenate([self.covs_ID,self.covs_ID])
                
                # now doing 3 for second phenotype
                covs2use_MT = self.all_covs2use[col_MT].split(',')
                Ic_MT = np.zeros(self.covariates.shape[0],dtype=bool)
                for cov_MT in covs2use_MT:
                    if cov_MT != '': # FOR AMELIE: this was cov - makes more sense if cov_MT
                        assert any(self.covariates==cov_MT), 'covariate not in cov table'
                        Ic_MT = np.logical_or(Ic_MT,self.covariates==cov_MT)
                covariates_names_MT = self.covariates[Ic_MT]
                print('Initial covs in social_data for second phenotype are ' + str(covariates_names_MT))
                if any(Ic_MT): #Ic has only 1 value as len is 1 
                    covs_MT = self.all_covs[:,Ic_MT] #self.covs is covs for phenotype in column col
                    covs_MT = np.append(np.zeros(covs_MT.shape),covs_MT,0)
                else:
                    covs_MT = None
                if self.covs is not None and covs_MT is not None:
                    self.covs = np.append(self.covs,covs_MT,1)
                    covariates_names = np.append(covariates_names,covariates_names_MT)
                elif self.covs is None and covs_MT is not None:
                    self.covs = covs_MT
                    covariates_names = covariates_names_MT
            
            
        #here I decided to have covs_ID as None if covs is None, to be rigorous about what comesout of this script, even though later I will turn covs_ID to pheno_ID when I add the intercept
        #pdb.set_trace()
        if self.covs is None:
            self.covs_ID = None
            assert len(covariates_names)==0, "pb covariates_names"
            #assert covariates_names is None, "pb in covariates_names"
        
        return {'trait' : self.trait,
                'trait_MT' : self.trait_MT,
                'pheno' : self.pheno, # 2N
                'pheno_ID' : self.pheno_ID, #N 
                'covs' : self.covs, #2N
                'covs_ID' : self.covs_ID, #N and same as pheno_ID
                'covariates_names' : covariates_names,
                'kinship_type' : self.GRM_version, 
                'kinship_full' : self.kinship_full, #N_a x N_a
                'kinship_full_ID' : self.kinship_full_ID, #N_a
                'cage_full' : self.cage_full, #N_a
                'cage_full_ID' : self.cage_full_ID, #N_a
                'maternal_full' : self.maternal_full,
                'maternal_full_ID' : self.maternal_full_ID,
                'subset_IDs' : self.subset_IDs}






