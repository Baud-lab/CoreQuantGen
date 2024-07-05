#!/usr/bin/env Rscript

###########################################################################
### PARSING ESTIMATE AND STE AFTER EXP_BIVAR - BIVARIATE AND UNIVARIATE ###
###########################################################################

sourcedir ="functions/"
# Get function to prepare results, colest and colste --> change them there if change them in exp_bivar
source(file=file.path(sourcedir, "prepare_res_update.R"))

suppressMessages(library("optparse"))

option_list = list(make_option("--est", action="store", default=NA, type='character'
                               , help="Path to file with estimates [required]"),
                   make_option("--ste", action="store", default=NULL, type='character'
                               , help="Path to file with STE, if have them, [default= %default]"),
                   make_option("--out", action="store", default=NA, type='character',
                               help="Path to output RData [required]"),
                   make_option("--corr0", action="store", default=NULL, type='character',
                               help="the corr0 that have been constrained, est file should have 2 lines per (pair of) phenotype [default= %default]"))

opt = parse_args(OptionParser(option_list=option_list))

#opt = list(est = "~/PRJs/outputs/test_maternal/output_HT/VD/bivariate/simulations/Andres_kinship_DGE/pheno5_uni/pheno5_uni_pheno6_uni_corr_Ad1d2_zero_est.txt", 
#           ste = "~/PRJs/outputs/test_maternal/output_HT/VD/bivariate/simulations/Andres_kinship_DGE/pheno5_uni/pheno5_uni_pheno6_uni_corr_Ad1d2_zero_STE.txt", 
#           out = "test.Rdata", 
#           corr0 = "corr_Ad1d2")

est_file = opt$est
ste_file = opt$ste
outRdata = opt$out
corr0 = opt$corr0

### Reading estimates
est <- read.csv(file = est_file, sep = "\t", header = F)
if(length(colest) != ncol(est)){
  stop("Something wrong with number of columns and colnames")
}
colnames(est) <- colest
est = prepare_res(est, nocol = c("")) 
#est[1:10,]

### Reading STE
#ste_file = gsub("_est", "_STE", est_file)
if (! is.null(ste_file)){
  ste <- read.csv(file = ste_file,  sep = "\t", header = F)
  if(length(colste) != ncol(ste)){
    stop("Something wrong with number of columns and colnames")
  }
  colnames(ste) <- colste
  ste = prepare_res(ste, nocol = c(""))
  
  # gives problem with repeated values...
  #motch = match(est$taskID, ste$taskID) #[,"taskID"], ste[,"taskID"]) #mmmm not convinced
  #ste = ste[which(ste[,"taskID"] == est[,"taskID"]),] # mmmm not convinced
}else{
  message("No file for STE was provided")
  ste = NULL
}


### Merging the two
# I want to verify that for each phenotype I have the est full and constrained and the STE full and constrained
if (!is.null(corr0)){
  cat("analysing results for corr",corr0,"constrained to 0\n")
  row_constrained = which(est[,corr0] == 0)
  #row_full = as.numeric( rownames(est)[-row_constrained] )
  
  names_constrained = est[row_constrained,"taskID"]
  names_full = est[-row_constrained,"taskID"]
  # checking that for each constrained I have a full model - full model but not constrained is accepted at this point
  if(!all(names_constrained %in% names_full)) stop("don't have a full model for all corr constrained")
  # keeping only the ones that have the constrain (and ordering as row1:full, row2:constrained)
  est = est[which(est[,"taskID"] %in% names_constrained),]
  
  if (!is.null(ste)){
    # keeping only the ste that have the constrain (and ordering as row1:full, row2:constrained)
    ste = ste[which(ste[,"taskID"] %in% names_constrained),] 
    #NB: this might change if calculate the STE ?
    #if(any(!is.na(ste[row_constrained,"STE_Ad1d2"]))) stop("problem with STE at corr constrained, supposed to be NA") 
    # when STE is calculated in constrained as well:
    if(any(ste[row_constrained,"STE_Ad1d2"] != 0)) stop("problem with STE at corr constrained, supposed to be 0") 
    # check that I have the two datasets are in same order and same lenght
    if(any(est[,"taskID"] != ste[,"taskID"])) stop("estimates and ste don't match")
    
    # doing it again in case something got filtered out
    row_constrained = which(est[,corr0] == 0)
    # 
    VCs <- merge(x = est[-row_constrained,], y =ste[-row_constrained,], by=c("taskID","trait1","trait2"))
  } else{
    VCs <- est[-row_constrained,]
  }
  VCs[,"pv_chi2dof2"] = NA
  VCs[,'pv_chi2dof2'] = pchisq(2*(-est[-row_constrained,'LML']+est[row_constrained,'LML']),lower.tail = FALSE, df=1)
} else {
  if (!is.null(ste)){
    VCs <- merge(x = est, y =ste, by=c("trait1","trait2"))
  }else{
    VCs <- est
  }
}



### Saving as Rdata
message("saving est (and STE if present) to Rdata: ", outRdata)
save(VCs,
     file = outRdata, compress = T)
