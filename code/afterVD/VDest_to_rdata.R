#!/usr/bin/env Rscript

###########################################################################
### PARSING ESTIMATE AND STE AFTER EXP_BIVAR - BIVARIATE AND UNIVARIATE ###
###########################################################################

### Get function to prepare results, colest and colste --> change them there if change them in exp_bivar
##suppressMessages(library("here"))
##source(here("./code/afterVD/Rfun", "prepare_res.R"))

suppressMessages(library("optparse"))

option_list = list(make_option("--est", action="store", default=NA, type='character'
                               , help="Path to file with estimates [required]"),
                   make_option("--ste", action="store", default=NULL, type='character'
                               , help="Path to file with STE, if have them, [default= %default]"),
                   make_option("--out", action="store", default=NA, type='character',
                               help="Path to output RData [required]"),
                   make_option("--corr0", action="store", default="None", type='character',
                               help="the corr0 that have been constrained, est file should have 2 lines per (pair of) phenotype [default= %default]"),
                   make_option("--swap", action="store", default=FALSE, type='logical',
                               help="TRUE or FALSE for pheno1=trait2 and pheno2=trait1 [default= %default]"))

opt = parse_args(OptionParser(option_list=option_list))

### GETTING OPTIONS
est_file = opt$est
ste_file = opt$ste
outRdata = opt$out
if(opt$corr0 == "None"){corr0 = NULL}else{corr0 = opt$corr0}
swap=opt$swap

### DEFINE FUCNTIONS 
# My results
# Fx 1 : preparing results, from original dataset, removing unused columns (nocol)
# TODO: put this to source if it works !!!!!!
prepare_res = function(res, nocol = c('sample_size','sample_size_all','covariates_names', 'conv', 'LML')){
  # assigning taskID = name phenoytpe1_phenotype2
  #     TODO: check how it works with univariate, imagine it would be something like trait1_None, trait2_None...
  #     if want something different can do 
  #     if ("trait2" %in% nocol) {res[,"taskID"] = res[,"trait1"] # or res[,"taskID"] = paste0(res[,"trait1"], "univariate")}
  order_cols = c("taskID",colnames(res))
  res[,"taskID"] = paste(res[,"trait1"], res[,"trait2"], sep='_') # NB: like this also have the same name that is saved as _est.txt _STE.txt output files
  res = res[,order_cols]
  
  # filtering out unwanted columns
  res = res[,! colnames(res) %in% nocol]
  # removing columns with all -999 i.e. NAs
  NAs =  apply(res, 2, FUN = function(res) all(res == -999))
  res = res[!NAs]
  res[res==-999] = NA # see if want to try this too
  return(res)
}
colest = c('trait1', 'trait2', 'sample_size1', 'sample_size1_cm', 'sample_size2', 'sample_size2_cm', #6
           'union_focal', 'inter_focal', 'union_cm', 'inter_cm', #4
           'covariates_names', 'conv', 'LML', #3
           'prop_Ad1', 'prop_Ad2','prop_As1', 'prop_As2', #4
           'corr_Ad1d2', 'corr_Ad1s1', 'corr_Ad1s2', 'corr_Ad2s1','corr_Ad2s2', 'corr_As1s2', #6
           'prop_Ed1', 'prop_Ed2','prop_Es1', 'prop_Es2', #4
           'corr_Ed1d2', 'corr_Ed1s1', 'corr_Ed1s2', 'corr_Ed2s1', 'corr_Ed2s2', 'corr_Es1s2', #6
           'prop_Dm1', 'prop_Dm2', 'corr_Dm1Dm2', #3
           'prop_C1',  'prop_C2', 'corr_C1C2', #3
           'tot_genVar1', 'tot_genVar2', #2
           'total_var1', 'total_var2') #2

colste = c('trait1', 'trait2', 'time_exec',
           'STE_Ad1', 'STE_Ad2', 'STE_As1', 'STE_As2', 
           'STE_Ad1d2',  'STE_Ad1s1', 'STE_Ad1s2',  'STE_Ad2s1', 'STE_Ad2s2',  'STE_As1s2',
           'STE_Ed1', 'STE_Ed2', 'STE_Es1', 'STE_Es2', 
           'STE_Ed1d2', 'STE_Ed1s1','STE_Ed1s2', 'STE_Ed2s1', 'STE_Ed2s2', 'STE_Es1s2',
           'STE_Dm1', 'STE_Dm2', 'STE_Dm1Dm2',
           'STE_C1C2', # NB: because of the way we calculate STE we cannot get STE for C1/C2; 
           'STE_totv1', 'STE_totv2',
           'corParams_Ad1_As1', 'corParams_Ed1_Es1', 'corParams_Ed1_Dm1', 'corParams_Es1_Dm1', 
           'corParams_Ad2_As2', 'corParams_Ed2_Es2', 'corParams_Ed2_Dm2', 'corParams_Es2_Dm2')

ste_dict = c("STE_Ad1", "STE_Ad2", "STE_As1", "STE_As2", "STE_Ad1d2", "STE_Ad1s1", "STE_Ad1s2",
             "STE_Ad2s1", "STE_Ad2s2", "STE_As1s2", "STE_Ed1", "STE_Ed2", "STE_Es1", "STE_Es2", "STE_Ed1d2", "STE_Ed1s1",
             "STE_Ed1s2", "STE_Ed2s1", "STE_Ed2s2", "STE_Es1s2", 'STE_Dm1', 'STE_Dm2', 'STE_Dm1Dm2', "STE_C1C2", "STE_totv1", "STE_totv2") # NB: don't have STE_C1, STE_C2, have STE_totv instead
# STE_totv1 --> total_var1
names(ste_dict) = ste_dict

for (e in seq_along(ste_dict)){
  p = gsub("STE_", "", ste_dict[e])
  #print(p)
  if (length(grep("tot", p)) > 0) {n = gsub("totv", "total_var", p)}
  else if (nchar(p) < 4) {n = paste0("prop_", p)}
  else {n = paste0("corr_", p)}
  ste_dict[e] = n
}
rm(e,p,n)
cat("Dictionary STE_name - param_name created\n")

#### START WITH ANALYSIS
### Reading estimates
est = read.csv(file = est_file, sep = "\t", header = F)
if(length(colest) != ncol(est)){
  stop("Something wrong with number of columns and colnames")
}
colnames(est) = colest
est = prepare_res(est, nocol = c("")) 
#est[1:10,]

### Reading STE
#ste_file = gsub("_est", "_STE", est_file)
if (! is.null(ste_file)){
  ste = read.csv(file = ste_file,  sep = "\t", header = F)
  if(length(colste) != ncol(ste)){
    stop("Something wrong with number of columns and colnames")
  }
  colnames(ste) = colste
  ste = prepare_res(ste, nocol = c(""))
  
  # gives problem with repeated values...
  #motch = match(est$taskID, ste$taskID) #[,"taskID"], ste[,"taskID"]) #mmmm not convinced
  #ste = ste[which(ste[,"taskID"] == est[,"taskID"]),] # mmmm not convinced
}else{
  cat("No file for STE was provided\n")
  ste = NULL
}

swap_col = function(df){
  colnames(df) = gsub("C2C1", "C1C2", 
                        gsub("s2s1","s1s2", 
                             gsub("d2d1","d1d2", 
                                  gsub("3","2",
                                       gsub("4","1", 
                                            gsub("2","4", 
                                                 gsub("1","3",colnames(df))))))))
  return(df)
}

if(swap){
  cat("swapping columns names so that will be: pheno1=trait1 and pheno2=trait2", "\n")
  est = swap_col(est)
  est=est[, match(c("taskID",colest), colnames(est), nomatch = 0)]
  
  ste=swap_col(ste)
  ste=ste[,match(c("taskID",colste), colnames(ste), nomatch = 0)]
  
  if(!is.null(corr0)){
    corr0 = gsub("C2C1", "C1C2", 
                 gsub("s2s1","s1s2", 
                      gsub("d2d1","d1d2", 
                           gsub("3","2",
                                gsub("4","1", 
                                     gsub("2","4", 
                                          gsub("1","3", corr0)))))))
  }
}

### Merging the two
# I want to verify that for each phenotype I have the est full and constrained and the STE full and constrained
if (!is.null(corr0)){
  cat("analysing results for corr",corr0,"constrained to 0 or 1\n")
  cat("assuming all phenotypes are in order and each has first row alternative model, second row null model\n")
  row_constrained = seq(2,nrow(est),2)
  #row_constrained = which(est[,corr0] == 0)
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
    STE0 = gsub("corr_", "STE_", corr0)
    #NB: this might change if calculate the STE ?
    if(any(!is.na(ste[row_constrained,"STE_Ad1d2"]))) stop("problem with STE at corr constrained, supposed to be NA") 
    # when STE is calculated in constrained as well:
    #if(any(ste[row_constrained, STE0] != 0)) stop("problem with STE at corr constrained, supposed to be 0") 
    # check that I have the two datasets are in same order and same lenght
    if(any(est[,"taskID"] != ste[,"taskID"])) stop("estimates and ste don't match")
    
    # doing it again in case something got filtered out
    #row_constrained = which(est[,corr0] == 0) # doesn't work when constrain is 1
    
    VCs = merge(x = est[-row_constrained,], y =ste[-row_constrained,], by=c("taskID","trait1","trait2"))
    VCs0 = merge(x = est[row_constrained,], y =ste[row_constrained,], by=c("taskID","trait1","trait2"))
  } else{
    VCs = est[-row_constrained,]
  }
  VCs[,"pv_chi2dof1"] = NA
  VCs[,'pv_chi2dof1'] = pchisq(2*(-est[-row_constrained,'LML']+est[row_constrained,'LML']),lower.tail = FALSE, df=1)
} else {
  VCs0 = NULL
  if (!is.null(ste)){
    if(any(duplicated(est$taskID))){
      est$taskID[duplicated(est$taskID)] = paste0(est$taskID[duplicated(est$taskID)], 1:sum(duplicated(est$taskID)))
      ste$taskID[duplicated(ste$taskID)] = paste0(ste$taskID[duplicated(ste$taskID)], 1:sum(duplicated(ste$taskID)))
      }
    VCs = merge(x = est, y = ste, by = c("taskID","trait1","trait2"))
  }else{
    VCs = est
  }
}

res = list(VCs=VCs, VCs0=VCs0)
### Saving as Rdata
cat("saving est (and STE if present) to Rdata: ", outRdata,"\n")
save(res,
     file = outRdata, compress = T)


