#!/usr/bin/env Rscript
suppressMessages(library("here"))
source(here("./code/preVD/Rfun", "simulations.R"))

# TODO:
# The dam for MATERNAL EFFECT comes from by a random metadata with 6000 rows; file path hardcoded in the script 
# -> should improve this

suppressMessages(library("MASS"))
suppressMessages(library("rhdf5"))
suppressMessages(library("optparse"))

option_list = list(make_option("--in_file", action="store", default=NA, type='character', 
                               help="Path to input hdf5 file"),
                   make_option("--GRM_version", action="store", default=NA, type='character', 
                               help="name of GRM version, i.e. subgroup after 'GRM/'"),
                   make_option("--cage_version", action="store", default=NA, type='character', 
                               help="name of cage version, i.e. subgroup after 'cages/'"),
                   make_option("--dam_version", action="store", default="None", type='character', 
                               help="name of dam version, i.e. subgroup after 'dam/'"),
                   make_option("--sex_version", action="store", default="None", type='character', 
                               help="name of cov version, i.e. subgroup after 'covariates/'"),
                   make_option("--out", action="store", default='.', type='character',
                               help="Path to output dir [default = %default]"),
                   make_option("--prefix", action="store", default=NULL, type='character',
                               help="Prefix of output file [default = %default]"),
                   make_option("--model", action="store", default="uni", type='character',
                               help="which model used to simulate, univariate or bivariate or sexvariate: [default = %default]"), 
                   make_option("--vars_file", action="store", default=NULL, type='character',
                               help="Path to file with list with vars and rhos, .csv
                   Possible vars: (NB: ***put 0 for those not used***)
                     1) sigma_sq_Ad1;  2) sigma_sq_Ad2;
                     3) sigma_sq_As1;  4) sigma_sq_As2;
                     5) rho_Ad1d2;     6) rho_As1s2;        
                     7) rho_Ad1s1;     8) rho_Ad2s2;
                     9) rho_Ad2s1;     10) rho_Ad1s2;
                     
                     11) sigma_sq_Ed1; 12) sigma_sq_Ed2;
                     13) sigma_sq_Es1; 14) sigma_sq_Es2;
                     15) rho_Ed1d2;    16) rho_Es1s2;
                     17) rho_Ed1s1;    18) rho_Ed2s2;
                     19) rho_Ed2s1;    20) rho_Ed1s2;
                     
                     21) sigma_sq_C1;  22) sigma_sq_C2;
                     23) rho_C
                     24) sigma_sq_Dm1; 25) sigma_sq_Dm2; 
                     26) rho_Dm"),
                   
                   make_option("--phenos", action="store", default=2, type='integer',
                               help="Number of phenotypes that want to simulate per each type of model, DGE, IGE or noise [default= %default]"),
                   make_option("--seed", action="store", default=5, type='integer',
                               help="A random number to change starting point of randomization [default= %default]"),
                   make_option("--subset", action="store", default=-1, type='double',
                               help="Number of CAGES to subset, use -1 for NONE [default= %default]"),
                   make_option("--sub_version", action="store", default="None", type='character', 
                               help="name of subset version, i.e. subgroup after 'subsets/', used in case want to simulate from specific subset of real data"),
                   make_option("--missing", action="store", default=0, type='character',
                               help="Number of missing values to include in simulations or name of real phenotypes from which take missing [default= %default]"),
                   make_option("--pheno_version", action="store", default=NULL, type='character', 
                               help="name of pheno version, i.e. subgroup after 'phenotypes/', used only in case of missing from real phenos")
)

opt = parse_args(OptionParser(option_list=option_list))

cat("Getting options", "\n")
HShdf = opt$in_file
outdir = opt$out
model = opt$model
GRM_version = opt$GRM_version
cage_version = opt$cage_version
dam_version = opt$dam_version
sex_version = opt$sex_version
sub_version = opt$sub_version ### have to align names BEFORE doing anything - have TODO this
# storing missing as numeric if number, else as character
if (!is.na(suppressWarnings(as.numeric(opt$missing)))) {
  missing <- as.numeric(opt$missing)  # Convert to numeric if possible
} else {
  missing <- opt$missing  # Otherwise, keep as character
}

get_vars = read.csv(opt$vars_file, header = T)
if(model == "uni"){
  check_vars=c('sigma_sq_Ad1', 'sigma_sq_As1', 'rho_Ad1s1', 'sigma_sq_Ed1', 'sigma_sq_Es1', 'rho_Ed1s1', 'sigma_sq_C1', 'sigma_sq_Dm1')
  if(!all(check_vars %in% colnames(get_vars))){
    stop(paste0("missing ", check_vars[! check_vars %in% colnames(get_vars)],", check the names or put 0 if don't want to include\n"))
  }
}else if(model == "bi"){
  check_vars=c('sigma_sq_Ad1', 'sigma_sq_Ad2', 'sigma_sq_As1', 'sigma_sq_As2', 
               'rho_Ad1d2', 'rho_Ad1s1', 'rho_Ad2s1', 'rho_Ad1s2', 'rho_Ad2s2', 'rho_As1s2', 
               'sigma_sq_Ed1', 'sigma_sq_Ed2', 'sigma_sq_Es1', 'sigma_sq_Es2', 
               'rho_Ed1d2', 'rho_Ed1s1', 'rho_Ed2s1', 'rho_Ed1s2', 'rho_Ed2s2', 'rho_Es1s2', 
               'sigma_sq_C1', 'sigma_sq_C2', 'rho_C', 'sigma_sq_Dm1', 'sigma_sq_Dm2', 'rho_Dm')
  if(!all(check_vars %in% colnames(get_vars))){
    #if(!all(check_vars %in% get_vars[,1])){
    stop(paste0("missing ", check_vars[! check_vars %in% colnames(get_vars)],", check the names or put 0 if don't want to include\n"))
  }
  
}
var_list = as.numeric(get_vars[1,])
names(var_list) = colnames(get_vars)

if(model == "sex"){nb_phenos = opt$phenos*2}else{nb_phenos = opt$phenos}
seed = opt$seed
nb_cages_to_keep = opt$subset
### DONE WITH OPTIONS ###


h5 = h5read(HShdf,"/")

#---- LOADING GRM ----
# load(file.path(datadir, "HSmice_kinship.RData")); GRM = kin
cat("Loading GRM", "\n")
GRM = h5$GRM[[GRM_version]]$matrix
colnames(GRM) = rownames(GRM) = h5$GRM[[GRM_version]]$row_header$sample_ID
#str(GRM)


#---- LOADING METADATA INFO - FOR CAGES  ----

## W - cage assignment matrix
cat("Loading cages", "\n")
cages <- h5$cages[[cage_version]]$array
names(cages) <- h5$cages[[cage_version]]$sample_ID
str(cages)
cages <- cages[rownames(GRM)]
cages <- cages[!is.na(cages)]
str(cages)

# all.equal(as.character(rownames(GRM)), names(cages))
#cage_names <- as.character(unique(cages))

#---- LOADING METADATA INFO - FOR DAM - if any  ----
dam_version
if(dam_version != "None"){
  cat("Loading dam", "\n")
  dam <- h5$dam[[dam_version]]$array
  names(dam) <- h5$dam[[dam_version]]$sample_ID
  dam <- dam[rownames(GRM)]
  #str(dam)
}else{
  cat("Creating maternal matrix", "\n")
  metadata_file="/users/abaud/htonnele/PRJs/outputs/test_maternal/input/metadata_16S.txt" #TODO: change this one if want to, it is used to create 'dam'
  metadata <- read.csv(metadata_file, sep="\t", quote = '"')
  #dam <- sample(na.omit(metadata$dam), nrow(GRM), replace =T)
  dam <- na.omit(metadata$dam)[1:nrow(GRM)]
  #dam <- metadata$dam[1:nrow(GRM)] # having NA gives a problem later on
  names(dam) <- rownames(GRM)
  dam_version = "simulated"
}



#---- KEEPING ONLY SUBSET -----
if(sub_version != "None"){
  subID = h5$subsets[[sub_version]]
  subID = subID[subID %in% rownames(GRM)]
  subID = subID[subID %in% names(cages)] #
  subID = subID[subID %in% names(dam)]  # if dam
  ##### align GRM
  GRM = GRM[subID,subID]
  ##### align cages
  cages = cages[subID]
  ##### align dam
  dam = dam[subID]; 
}


#str(rownames(GRM))
stopifnot(all.equal(as.character(rownames(GRM)), names(cages)))
stopifnot(all.equal(as.character(rownames(GRM)), names(dam)))

cage_names <- as.character(unique(cages))
#str(cage_names)


### ## W - cage assignment matrix
### cat("Loading cages", "\n")
### #cages <- h5$data_bcNcovariates$rows_subjects$cage
### #names(cages) <- h5$data_bcNcovariates$rows_subjects$outbred
### cages <- h5$cages[[cage_version]]$array
### names(cages) <- h5$cages[[cage_version]]$sample_ID
### 
### cages <- cages[rownames(GRM)]
### # all.equal(as.character(rownames(GRM)), names(cages))
### cage_names <- as.character(unique(cages))

#### Subsetting cages ####
#TODO NB: change name of output file
set.seed(seed) # line 48: seed = opt$seed
if(nb_cages_to_keep == -1){
  cat("No subset", "\n")
}else if(nb_cages_to_keep > length(cage_names)) {
  stop("Subset number is bigger than total number of cages, please select one smaller than: ", length(cage_names))
}else if(nb_cages_to_keep == length(cage_names)){
  warning("Subset is the actual total number of cages: ", nb_cages_to_keep, " = ", length(cage_names))
}else if(nb_cages_to_keep > 0 ){
  cat("Subset = ", as.integer(nb_cages_to_keep), " cages", "\n")
  cage_names <- sample(cage_names, size=nb_cages_to_keep) # line 49: nb_cages_to_keep = opt$subset
  cages <- cages[cages %in% cage_names]
  GRM <- GRM[names(cages), names(cages)]
  dam <- dam[names(cages)]
  #covs <- covs[names(cages),]
}else {
  stop("Invalid number, please use -1 for no subset or the number of cages to subset")}
#stop("stopping for now")
#### end of subset

cat("Creating cage matrix", "\n")
W <- matrix(0, nrow= dim(GRM)[1], ncol = length(cage_names), dimnames = list(rownames(GRM), cage_names))

for (mouse in rownames(W)){
  cage = cages[mouse]
  W[mouse,cage] = 1 
}

## C - Cagemate matrix - method 1 
C = W %*% t(W) # NB: this is slower cos need to do the calculation 

# Get cage distribution
sums = apply(C, MAR= 1, FUN=sum)
cat("cage distribution is:", "\n")
table(sums, useNA = "always")


#---- Z = C with diag = 0 ---- 
Z <- C; 
diag(Z) = 0 # Same order as GRM already since comes from C

#---- I - diag = 1, rest = 0 ----
# TODO: now I have nrow = GRM_tot --> may wanna change this to something else like only the number of focals...
I = diag(nrow = nrow(GRM), ncol= ncol(GRM))
dimnames(I) = dimnames(GRM) # directly same order as GRM


## Matn - Dam matrix - 
W_mat <- matrix(0, nrow= dim(GRM)[1], ncol = length(dam), dimnames = list(rownames(GRM), dam))

for (mouse in rownames(W_mat)){
  dam_oi = na.omit(dam)[mouse]
  W_mat[mouse,dam_oi] = 1 
}

# Dm matrix
Dm = W_mat %*% t(W_mat) # NB: this is slower cos need to do the calculation

sums = apply(Dm, MAR= 1, FUN=sum)
cat("dam distribution is", "\n")
table(sums, useNA = "always")


## ---- List of Matrices ----
cat("Dim GRM: ", dim(GRM)[1], ",",  dim(GRM)[2], "\n")  # line 9 
cat("Dim Z: ", dim(Z)[1], ",",  dim(Z)[2], "\n")  # [0,1; diag = 0]   # line 45
cat("Dim I: ", dim(I)[1], ",",  dim(I)[2], "\n") # [diag=1, 0]  # line 54
cat("Dim C: ", dim(C)[1], ",",  dim(C)[2], "\n") # [0,1; diag = 1]   # line 34
cat("Dim Dm: ", dim(Dm)[1], ",",  dim(Dm)[2], "\n") # [0,1; diag = 1]   # line 34


###### 
# simulations uni
if(model == "uni"){
  set.seed(seed)
  cat("\nStarting simulations for", model, "\n")
  cat("\t using params1 only\n")
  sim = sim_uni(nb_phenos, GRM, Z, I, C, Dm,
                sigma_sq_AD = var_list["sigma_sq_Ad1"], sigma_sq_AS = var_list["sigma_sq_As1"], rho_A = var_list["rho_Ad1s1"],
                sigma_sq_ED = var_list["sigma_sq_Ed1"], sigma_sq_ES = var_list["sigma_sq_Es1"], rho_E = var_list["rho_Ed1s1"],
                sigma_sq_C = var_list["sigma_sq_C1"], sigma_sq_Dm = var_list["sigma_sq_Dm1"])
  cat("Finished simulations for ", model,"\n\n")
}else if (model == "bi" | model == "sex"){
  set.seed(seed)
  half_phenos = nb_phenos/2
  cat("\nStarting simulations for", model, "\n")
  sim = sim_bi(half_phenos, GRM, Z, I, C, Dm, 
               sigma_sq_AD = c(var_list["sigma_sq_Ad1"], var_list["sigma_sq_Ad2"]), sigma_sq_AS = c(var_list["sigma_sq_As1"], var_list["sigma_sq_As2"]),
               rho_AD = var_list["rho_Ad1d2"], rho_AS = var_list["rho_As1s2"], rho_A1 = var_list["rho_Ad1s1"], rho_A2 = var_list["rho_Ad2s2"], rho_AD2_AS1 = var_list["rho_Ad2s1"], rho_AD1_AS2 = var_list["rho_Ad1s2"], 
               sigma_sq_ED = c(var_list["sigma_sq_Ed1"], var_list["sigma_sq_Ed2"]), sigma_sq_ES = c(var_list["sigma_sq_Es1"], var_list["sigma_sq_Es2"]), 
               rho_ED = var_list["rho_Ed1d2"], rho_ES = var_list["rho_Es1s2"], rho_E1 = var_list["rho_Ed1s1"], rho_E2 = var_list["rho_Ed2s2"], rho_ED2_ES1 = var_list["rho_Ed2s1"], rho_ED1_ES2 = var_list["rho_Ed1s2"],
               sigma_sq_C = c(var_list["sigma_sq_C1"], var_list["sigma_sq_C2"]),
               rho_C = var_list["rho_C"], 
               sigma_sq_Dm = c(var_list["sigma_sq_Dm1"], var_list["sigma_sq_Dm2"]),
               rho_Dm = var_list["rho_Dm"])
  cat("Finished simulations for ", model,"\n\n")
}


### HERE have the simulated phenotypes #####
phenos <- sim$sims

#---- GET SEX COVARIATE - if needed -----
if(model == "sex"){
  cat("in sexvariate mode, rearranging phenotypes\n")
  #or_cov = h5read(h5, paste0("/sex_cov/", sex_version))
  sex = h5$sex_cov[[sex_version]]$array
  names(sex) = h5$sex_cov[[sex_version]]$sample_ID
  table(sex) # orh5$covariates$noBatch$col_header$covariate_ID # "Sex" is the first one
  #    1    2 
  #  951  918  CFW
  # 1196 1252  HSmice
  sex = sex[rownames(phenos)]
  cat(table(sex), "\n")
  
  # order so that have all males/females together
  sex = sort(sex)
  sorted_IDs = names(sex)
  
  phenos = phenos[sorted_IDs, ]
  
  f = names(sex[which(sex == 1)]) # in CFW and HSmice 1 corresponds to females
  m = names(sex[which(sex == 2)]) # in CFW and HSmice 2 corresponds to males
  cat("n females:", length(f), "; n males: ", length(m), "\n")
  
  # Adding male/female normal mask  # this can be skipped since will be removed anyways
  ##phenos[f, seq(2, ncol(phenos), 2)] = -999
  ##phenos[m, seq(1, ncol(phenos), 2)] = -999
  ##colnames(phenos)[seq(1, ncol(phenos), 2)] = paste0("pheno", seq(1, ncol(phenos)/2, 1), ".f")
  ##colnames(phenos)[seq(2, ncol(phenos), 2)] = paste0("pheno", seq(1, ncol(phenos)/2, 1), ".m") 
  
  # building phenotypes as real ones
  ##pre_phenos = phenos # uncomment to check rbind with lines below
  phenos = rbind(phenos[f, seq(1, ncol(phenos), 2), drop=F], phenos[m, seq(2, ncol(phenos), 2), drop=F]) 
  colnames(phenos) = paste0("pheno", seq(1, ncol(phenos), 1))
  
  # checking that the rbind works
  ##c = c(1,2)
  ##for(n in 1:ncol(phenos)){
  ##  stopifnot(table(phenos[,n] == pre_phenos[,c[1]]) == table(c(rep(T, length(f)), rep(F, length(m)))) ) # check male pheno 
  ##  stopifnot(table(phenos[,n] == pre_phenos[,c[2]]) == table(c(rep(T, length(m)), rep(F, length(f)))) ) # check female pheno
  ##  cat("pheno ", n, " ok \n")
  ##  c = c+2
  ##}
}

if( !is.double(missing)){
  #pheno_version = opt$pheno_version
  pheno_version = unlist(strsplit(opt$pheno_version,","))[1]
  misspheno = unlist(strsplit(missing, ","))
  cat("Introducing missing values from realdata, from phenos: ", missing, "\n")
  
  if(!pheno_version %in% names(h5$phenotypes)){
    warning(paste0("there is no phenotype version called ", pheno_version, ", not introducing NAs"))
  }else{
    realPheno = h5$phenotypes[[pheno_version]]$matrix; 
    colnames(realPheno) = h5$phenotypes[[pheno_version]]$col_header$phenotype_ID; 
    rownames(realPheno) = h5$phenotypes[[pheno_version]]$row_header$sample_ID
    
    realPheno = realPheno[rownames(phenos), ]
    #if(!is.null(sub_version)){
    #  realPheno = realPheno[sub_ids, ]
    #  stopifnot(rownames(realPheno) == rownames(phenos))
    #}
    
    NAt1 = which(realPheno[,misspheno[1]] == -999, arr.ind = T)
    t1 = seq(1,ncol(phenos), 2)
    stopifnot(all.equal(rownames(phenos[NAt1,]), names(NAt1)))
    phenos[NAt1,t1] = -999
    
    NAt2 = which(realPheno[,misspheno[2]] == -999, arr.ind = T)
    t2 = seq(2,ncol(phenos), 2)
    stopifnot(all.equal(rownames(phenos[NAt2,]), names(NAt2)))
    phenos[NAt2,t2] = -999
  }
  #stop("Need a number of randomly introduces missing value, '0' for NONE") 
}else if (missing == 0){
  cat("no missing values introduced", "\n")
} else{
  missing = round(missing)
  if (model == "bi"){
    add_missing <- function(n_miss, ncols, phenos){
      n_miss = round(n_miss)
      cat("Introducing ",n_miss," missing values in ", ncols, " phenotypes", "\n")
      col_idxs = sample(ncol(phenos), ncols)
      for (c in col_idxs){
        idxs = sample(rownames(phenos), n_miss)
        phenos[idxs,c] = -999
      }
      return(phenos)
    }
    N = nrow(phenos)
    even <- seq(1,ncol(phenos), 2)
    odd <- seq(2,ncol(phenos), 2)
    phenos[,even] <- add_missing(N/2, ncol(phenos)/10, phenos[,even]) # 10/100 with 1/2 missing 
    phenos[,odd] <- add_missing(N*(4/5), ncol(phenos)/50, phenos[,odd]) # 2/100 with many many missing
    phenos[,apply(phenos, MARGIN = 2, function(x){all(x != -999)})] <- add_missing(N/3, ncol(phenos)*3/10, phenos[,apply(phenos, MARGIN = 2, function(x){all(x != -999)})])
  }
  rest_missing = missing - length(which(phenos == -999))
  if (rest_missing > 0){
    cat("Introducing ", rest_missing, " missing values at random", "\n")
    idxs = sample(prod(dim(phenos)), rest_missing)
    phenos[idxs] = -999
  }
}

# Plot the distribution of missing values per phenotypes
#true_phenos = h5$data_bcNcovariates$array
#length(which(true_phenos == -999)) / length(true_phenos == -999)
#pct_NAs = apply(true_phenos, MARGIN = 2, function(x){length(which(x == -999))/nrow(true_phenos)})
#freqs_NAs = hist(pct_NAs, plot = F)$counts / ncol(true_phenos); 
#hist(pct_NAs, main = "distribution of missing in HSmice phenotypes")

#TOCHECK IF OK TO REMOVE THIS:
colnames(phenos) <- paste0(colnames(phenos), ".", model,".s",seed)
params <- sim$params
vars <- sim$sample_vars
rhos <- sim$rhos
times <- sim$times


## write it to a hdf5 file!
#sim_types = paste0(substr(unique(unlist(sapply(strsplit(colnames(all_sims),"_"), "[", 2))), 1,1), collapse = "")
#sim_types
nb_cages = ncol(W) # dim(W)
#fid=file.path(outdir, paste0("HSmice_",ncol(phenos),model,"_",nb_cages,"cages","_seed",seed,".h5"))
fid=file.path(outdir, paste0(opt$prefix,"_", ncol(phenos), model,"_", nb_cages,"cages","_seed",seed,".h5"))
#fid
cat("Saving files to ", fid, "\n")
h5createFile(fid)

######## 1. PHENOTYPES # to be saved in format:  f['phenotypes'][self.phenos_version]['matrix'][:].T
str(phenos)
h5createGroup(fid,"phenotypes")
h5createGroup(fid,"phenotypes/mockphenos") #  simulations will correspond to "phenos_version" -> changed to "mockphenos"
#Sample IDs - group $row_headers
h5createGroup(fid,"phenotypes/mockphenos/row_header")
max_size <- max(nchar(rownames(phenos)))
h5createDataset(file=fid, dataset="phenotypes/mockphenos/row_header/sample_ID",
                dims=dim(phenos)[1], storage.mode='character', size=max_size)
h5write(obj=rownames(phenos), file=fid, name="phenotypes/mockphenos/row_header/sample_ID")

# Phenotype IDs
h5createGroup(fid,"phenotypes/mockphenos/col_header")
max_size <- max(nchar(colnames(phenos)))
h5createDataset(file=fid, dataset="phenotypes/mockphenos/col_header/phenotype_ID",
                dims=dim(phenos)[2], storage.mode='character', size=max_size)
h5write(obj=colnames(phenos), file=fid, name="phenotypes/mockphenos/col_header/phenotype_ID")
# Used covariates per pheno if any
#if(!is.null(covs)){
#  #h5createGroup(fid,"simulations/col_header") # this already exists
#  max_size <- max(nchar(usedCovs))
#  h5createDataset(file=fid, dataset="phenotypes/mockphenos/col_header/covariatesUsed",
#                  dims=length(usedCovs), storage.mode='character', size=max_size)
#  h5write(obj=usedCovs, file=fid, name="phenotypes/mockphenos/col_header/covariatesUsed")
#}
# Phenotype matrix
h5write(obj=phenos,file=fid,name="phenotypes/mockphenos/matrix")

## reading phenos
# fid = "/users/abaud/htonnele/HSmice/output/simulations/bivariate/sub100/D1t1/IG1_IG2/0.35/HSmice_bi_100IGE_100cages_seed40.h5"
# h5 = h5read(fid,"/")
# phenos= h5$simulations$matrix; colnames(phenos) <- h5$simulations$col_header$phenotype_ID; rownames(phenos) <- h5$simulations$row_header$sample_ID

######## 1b. subset - if any # # to be saved in format:  self.subset_IDs = f['subsets'][self.subset].asstr()[:]
if (sub_version != "None"){
  sub_ids = h5$subsets[[sub_version]]
  cat("including subset version: ", sub_version)
  str(sub_ids)
  h5createGroup(fid,"subsets")
  
  max_size = max(nchar(sub_ids))
  h5createDataset(file=fid, dataset=paste0("subsets/", sub_version), #"subsets/include",
                  dims=length(sub_ids), storage.mode='character', size=max_size)
  h5write(obj=as.character(sub_ids), file=fid, name=paste0("subsets/", sub_version)) #"cages/real/sample_ID")
}

######## 2. COVARIATES - deprecated # to be saved in format:  f['covariates'][self.covs_version]['matrix'][:].T
#if(!is.null(covs)){
#  str(covs)
#  h5createGroup(fid,"covariates")
#  h5createGroup(fid,"covariates/from_file")
#  #Sample IDs - group $row_headers
#  h5createGroup(fid,"covariates/from_file/row_header")
#  max_size <- max(nchar(rownames(covs)))
#  h5createDataset(file=fid, dataset="covariates/from_file/row_header/sample_ID",
#                  dims=dim(covs)[1], storage.mode='character', size=max_size)
#  h5write(obj=rownames(covs), file=fid, name="covariates/from_file/row_header/sample_ID")
#  #Cov names
#  h5createGroup(fid,"covariates/from_file/col_header")
#  max_size <- max(nchar(colnames(covs)))
#  h5createDataset(file=fid, dataset="covariates/from_file/col_header/covariate_ID",
#                  dims=dim(covs)[2], storage.mode='character', size=max_size)
#  h5write(obj=colnames(covs), file=fid, name="covariates/from_file/col_header/covariate_ID")
#  #Covs values
#  h5write(obj=covs,file=fid,name="covariates/from_file/matrix")
#}

######## 2b. SEXCOV - if any # # to be saved in format:   f['sex_cov'][self.sex_version]['array'].asstr()[:] #self
if (model == "sex"){
  str(sex)
  h5createGroup(fid,"sex_cov")
  h5createGroup(fid,paste0("sex_cov/", sex_version))
  
  # Sex_cov - `sex`
  max_size = max(nchar(sex))
  h5createDataset(file=fid, dataset=paste0("sex_cov/", sex_version, "/array"), 
                  dims=length(sex), storage.mode='character', size=max_size)
  h5write(obj=as.character(sex), file=fid, name= paste0("sex_cov/", sex_version, "/array")) 
  # Sex_IDs - `names(sex)`
  max_size = max(nchar(names(sex)))
  h5createDataset(file=fid, dataset=paste0("sex_cov/", sex_version, "/sample_ID"), 
                  dims=length(sex), storage.mode='character', size=max_size)
  h5write(obj=as.character(names(sex)), file=fid, name=paste0("sex_cov/", sex_version, "/sample_ID")) 
}

######## 3. CAGES - to be saved in format:  f['cages'][self.cage_version]['array'].asstr()[:] #self
h5createGroup(fid,"cages")
#h5createGroup(fid,"cages/real")
h5createGroup(fid,paste0("cages/", cage_version))

# Cages - `cages`
max_size = max(nchar(cages))
h5createDataset(file=fid, dataset=paste0("cages/", cage_version, "/array"), #"cages/real/array",
                dims=length(cages), storage.mode='character', size=max_size)
h5write(obj=as.character(cages), file=fid, name= paste0("cages/", cage_version, "/array")) #"cages/real/array")
# Cages_IDs - `names(cages)`
max_size = max(nchar(names(cages)))
h5createDataset(file=fid, dataset=paste0("cages/", cage_version, "/sample_ID"), #"cages/real/sample_ID",
                dims=length(cages), storage.mode='character', size=max_size)
h5write(obj=as.character(names(cages)), file=fid, name=paste0("cages/", cage_version, "/sample_ID")) #"cages/real/sample_ID")


######## 4. DAM - to be saved in format:  f['dam'][self.dam_version]['array'].asstr()[:] #self
h5createGroup(fid,"dam")
h5createGroup(fid, paste0("dam/", dam_version))

# Dams - `dam`
max_size = max(nchar(dam))
h5createDataset(file=fid, dataset=paste0("dam/", dam_version, "/array"),
                dims=length(dam), storage.mode='character', size=max_size)
h5write(obj=as.character(dam), file=fid, name=paste0("dam/", dam_version, "/array"))
# Dams_IDs - `names(dam)`
max_size = max(nchar(names(dam)))
h5createDataset(file=fid, dataset=paste0("dam/", dam_version, "/sample_ID"),
                dims=length(dam), storage.mode='character', size=max_size)
h5write(obj=as.character(names(dam)), file=fid, name=paste0("dam/", dam_version, "/sample_ID"))


######## 5. GRM -  to be saved in format:  f['GRM'][self.GRM_version]['matrix'][:]
h5createGroup(fid,"GRM")
#h5createGroup(fid,"GRM/Andres_kinship")
h5createGroup(fid, paste0("GRM/", GRM_version) )

h5createGroup(fid, paste0("GRM/", GRM_version,"/row_header") ) #"GRM/Andres_kinship/row_header")
max_size = max(nchar(rownames(GRM)))
h5createDataset(file=fid, dataset= paste0("GRM/", GRM_version,"/row_header/sample_ID"), #"GRM/Andres_kinship/row_header/sample_ID",
                dims=dim(GRM)[1], storage.mode='character', size=max_size)
h5write(obj=rownames(GRM), file=fid, name= paste0("GRM/", GRM_version,"/row_header/sample_ID")) #"GRM/Andres_kinship/row_header/sample_ID")
h5write(obj=GRM,file=fid,name= paste0("GRM/", GRM_version,"/matrix")) #"GRM/Andres_kinship/matrix")


######## Set params # params
str(params)
h5createGroup(fid,"sim_params")

max_size= max(nchar(colnames(params)))
h5createDataset(file = fid,"sim_params/col_header", 
                dims=dim(params)[2], storage.mode='character', size=max_size)
h5write(obj=colnames(params), file=fid, name="sim_params/col_header")

max_size= max(nchar(rownames(params)))
h5createDataset(file = fid,"sim_params/row_header", 
                dims=dim(params)[1], storage.mode='character', size=max_size)
h5write(obj=rownames(params), file=fid, name="sim_params/row_header")

#h5write(obj=params,file=fid,name="sim_params/matrix")
h5write(obj=as.matrix(params),file=fid,name="sim_params/matrix")



######## Sample Variances  
str(vars)
h5createGroup(fid,"variances")

max_size= max(nchar(colnames(vars)))
h5createDataset(file = fid,"variances/col_header", 
                dims=dim(vars)[2], storage.mode='character', size=max_size)
h5write(obj=colnames(vars), file=fid, name="variances/col_header")

max_size= max(nchar(rownames(vars)))
h5createDataset(file = fid,"variances/row_header", 
                dims=dim(vars)[1], storage.mode='character', size=max_size)
h5write(obj=rownames(vars), file=fid, name="variances/row_header")

#h5write(obj=vars,file=fid,name="variances/matrix")
h5write(obj=as.matrix(vars),file=fid,name="variances/matrix")


######## Rhos 
str(rhos)
h5createGroup(fid,"rhos")

max_size= max(nchar(names(rhos)))
h5createDataset(file = fid,"rhos/names", 
                dims=length(rhos), storage.mode='character', size=max_size)
h5write(obj=names(rhos), file=fid, name="rhos/names")


#h5write(obj=all_rhos,file=fid,name="rhos/matrix")
h5write(obj=as.matrix(rhos),file=fid,name="rhos/matrix")



######## Times of execution 
str(times)
h5createGroup(fid,"exec_times")

max_size= max(nchar(names(times)))
h5createDataset(file = fid,"exec_times/names", 
                dims=length(times), storage.mode='character', size=max_size)
h5write(obj=names(times), file=fid, name="exec_times/names")

h5write(obj=as.matrix(times),file=fid,name="exec_times/matrix")

#h5dump(fid, load = F)
