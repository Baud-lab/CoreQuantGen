#!/usr/bin/env Rscript

###########################################################################
### PARSING ESTIMATE AND STE AFTER EXP_BIVAR - BIVARIATE AND UNIVARIATE ###
###########################################################################

# Get function to prepare results, colest and colste --> change them there if change them in exp_bivar
suppressMessages(library("here"))
source(here("./code/afterVD/Rfun", "prepare_res.R"))

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

est_file = opt$est
ste_file = opt$ste
outRdata = opt$out
corr0 = if(opt$corr0 == "None"){corr0 = NULL}else{corr0 = opt$corr0}
swap=opt$swap

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
  cat("No file for STE was provided\n")
  ste = NULL
}

swap_col = function(df){
  colnames(df) <- gsub("C2C1", "C1C2", 
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
  
  corr0 = gsub("C2C1", "C1C2", 
              gsub("s2s1","s1s2", 
                   gsub("d2d1","d1d2", 
                        gsub("3","2",
                             gsub("4","1", 
                                  gsub("2","4", 
                                       gsub("1","3", corr0)))))))
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
    
    VCs <- merge(x = est[-row_constrained,], y =ste[-row_constrained,], by=c("taskID","trait1","trait2"))
    VCs0 <- merge(x = est[row_constrained,], y =ste[row_constrained,], by=c("taskID","trait1","trait2"))
  } else{
    VCs <- est[-row_constrained,]
  }
  VCs[,"pv_chi2dof1"] = NA
  VCs[,'pv_chi2dof1'] = pchisq(2*(-est[-row_constrained,'LML']+est[row_constrained,'LML']),lower.tail = FALSE, df=1)
} else {
  VCs0 = NULL
  if (!is.null(ste)){
    VCs <- merge(x = est, y =ste, by=c("taskID","trait1","trait2"))
  }else{
    VCs <- est
  }
}

res = list(VCs=VCs, VCs0=VCs0)
### Saving as Rdata
cat("saving est (and STE if present) to Rdata: ", outRdata,"\n")
save(res,
     file = outRdata, compress = T)


