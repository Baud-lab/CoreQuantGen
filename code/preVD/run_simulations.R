#!/usr/bin/env Rscript
## suppressMessages(library("here"))
## source(here("./code/preVD/Rfun", "simulations.R"))

# TODO:
# The dam for MATERNAL EFFECT comes from by a random metadata with 6000 rows; file path hardcoded in the script 
# -> should improve this
#BiocManager::install("rhdf5", update=FALSE, ask=FALSE)

suppressMessages(library("MASS"))
suppressMessages(library("rhdf5"))
#suppressMessages(library("hdf5r"))
suppressMessages(library("optparse"))
suppressMessages(library("matrixcalc"))

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

#### FUNCTIONS USED IN SIMULATIONS
# Fx.1 ####
build_covar_2dim<- function(variances, rho){
  C = matrix(0, nrow = 2, ncol = 2)
  C[1,1] = variances[1] # sigma_sq, upper left
  C[2,2] =  variances[2] # sigma_sq, lower right
  C[1,2] = C[2,1] = sqrt(prod(variances))*rho
  return(C)
}


# Fx.2 ####
build_covar_var <- function(var_D, var_S, rhos){
  # var_D = sigma_sd_AD[1], sigma_sd_AD[2]
  # var_S = sigma_sd_AS[1], sigma_sd_AS[2]
  # rhos = rho_AD, rho_AS, 
  #        rho_A1, rho_A2, 
  #        rho_AD2_AS1, rho_AD1_AS2
  cat("\nCalculating covariance DIRECT effect - 'C_D' - DE(pheno 1) vs DE(pheno 2):\n\tvars = ", paste(var_D, collapse = ","), "\n\trho = ", rhos[1], "\n")
  C_D= build_covar_2dim(var_D, rhos[1])
  cat("\nCalculating covariance INDIRECT effect - 'C_S'- IE(pheno 1) vs IE(pheno 2):\n\tvars = ",  paste(var_S, collapse = ","), "\n\trho = ", rhos[2], "\n")
  C_S= build_covar_2dim(var_S, rhos[2])
  
  cat("\nCalculating covariance PHENO 1 - 'cov1' - DE(pheno 1) vs IE(pheno 1):\n\tvars = ", var_D[1], ", ", var_S[1], "\n\trho = ", rhos[3], "\n")
  cov_1 = rhos[3]*sqrt(var_D[1] * var_S[1])
  cat("\nCalculating covariance PHENO 2 - 'cov2'- DE(pheno 2) vs IE(pheno 2):\n\tvars = ", var_D[2], ", ",var_S[2], "\n\trho = ", rhos[4], "\n")
  cov_2 = rhos[4]*sqrt(var_D[2] * var_S[2])
  
  cat("\nCalculating covariance 'cov_D2_S1' - DE(pheno 2) vs IE(pheno 1):\n\tvars = ", var_D[2],", ",var_S[1], "\n\trho = ", rhos[5], "\n")
  cov_D2_S1 = rhos[5]*sqrt(var_D[2] * var_S[1])
  cat("\nCalculating covariance 'cov_D1_S2' - DE(pheno 1) vs IE(pheno 2):\n\tvars = ", var_D[1],", ",var_S[2], "\n\trho = ", rhos[6], "\n")
  cov_D1_S2 = rhos[6]*sqrt(var_D[1] * var_S[2])
  
  covs_SD = c(cov_1, cov_D2_S1, cov_D1_S2, cov_2)
  #cat("\nCovariance matrix DS - 'C_DS' - composed as: ", \n\t[cov_1 = ",cov_1,",\t\tcov_D2_S1 = ", cov_D2_S1, "\n\t cov_D1_S2 = ",cov_D1_S2,",\tcov_2 = ", cov_2, "]")
  cat("\nCovariance matrix DS - 'C_DS' - composed as", "\n")
  print(matrix(c("cov_1", "cov_D2_S1", "cov_D1_S2", "cov_2"), nrow= 2, ncol= 2))
  C_DS = matrix(covs_SD, 
                nrow= 2, ncol= 2)
  print(C_DS)
  
  cov_matrices = list("C_D" = C_D, "C_S" = C_S, "C_DS" = C_DS)
  return(cov_matrices)
}


# Fx.3 ####
# SAMPLE VAR - calculate sample variance of a matrix
sample_var <- function(M){
  # check if matrix
  stopifnot(is.matrix(M))
  # check dimensions
  if(dim(M)[1] != dim(M)[2]){stop("Different nrow and ncol")}
  n = dim(M)[1]
  vecti = matrix(1,ncol=1,nrow=n)
  p = diag(n) - (vecti%*%t(vecti)) / n
  num=sum(diag(p%*%M%*%p))
  denom=(n-1)
  vari = num/denom
  return(vari)
}


# Fx.4 ####
# simulate univariate - 
# if don't want one effect, set sigma_sq to 0
sim_uni <- function(nb_phenos, GRM, Z, I, C, Dm,
                    sigma_sq_AD = NULL, sigma_sq_AS = NULL, rho_A = NULL, 
                    sigma_sq_ED = NULL, sigma_sq_ES = NULL, rho_E = NULL, 
                    sigma_sq_C = NULL, sigma_sq_Dm = NULL){
  # Warnings
  if(is.null(sigma_sq_AD)){stop("if don't want to simulate DGE, set 'sigma_sq_AD' = 0")}
  if(is.null(sigma_sq_AS)){stop("if don't want to simulate IGE, set 'sigma_sq_AS' = 0")}
  if(is.null(rho_A)){stop("if don't want to simulate IGE, set 'rho_A' = 0")}
  
  if(is.null(sigma_sq_ED)){stop("if don't want to simulate DGE, set 'sigma_sq_ED' = 0")}
  if(is.null(sigma_sq_ES)){stop("if don't want to simulate IGE, set 'sigma_sq_ES' = 0")}
  if(is.null(rho_E)){stop("if don't want to simulate IGE, set 'rho_E' = 0")}
  
  if(is.null(sigma_sq_C)){stop("if don't want to simulate CE, set 'sigma_sq_C' = 0")}
  if(is.null(sigma_sq_Dm)){stop("if don't want to simulate MaternalE, set 'sigma_sq_Dm' = 0")}
  
  # Calculating sigmas 
  sigma_ADS = rho_A * sqrt(prod(sigma_sq_AD,sigma_sq_AS))
  sigma_EDS = rho_E* sqrt(prod(sigma_sq_ED,sigma_sq_ES)) # 0.05
  
  #### Calculating sample variance
  ## A. Genetic components ######
  start <- proc.time()[3]; 
  cat("Calculating sample variance of GENETIC components", "\n")
  # calculating sv_GRM if sigma_sq_AD or sigma_ADS
  if(sigma_sq_AD != 0 | sigma_ADS != 0){
    sv_GRM = sample_var(GRM)
  }else{
    sv_GRM = 0
  }
  # calculating sv_Z_GRM_Zt if sigma_sq_AS or sigma_ADS
  if(sigma_sq_AS != 0 | sigma_ADS != 0){
    sv_Z_GRM_Zt = sample_var(Z%*%GRM%*%t(Z))
  }else{
    sv_Z_GRM_Zt = 0
  }
  
  # Scaling only when necessary, i.e. variance component ≠ 0; otherwise setting to 0
  # 1. sigma_sq_AD = DGE
  if (sigma_sq_AD != 0 ){
    H1 = GRM / sv_GRM
    sv_H1= sample_var(H1) # to calculate proportional
    
    dge = sigma_sq_AD*H1
  } else{
    cat("\t not including DGE in simulations", "\n")
    sv_H1 = 0 
    dge = 0
  }
  # 2. sigma_ADS = covariance DGE - IGE 
  if(sigma_ADS != 0){
    H2 = GRM / sqrt( sv_GRM * sv_Z_GRM_Zt )
    sv_termH2 = sample_var(H2%*%t(Z) + Z%*%t(H2))
    
    dge_ige = sigma_ADS*(H2%*%t(Z) + Z%*%t(H2))
  }else{
    cat("\t no covariance DGE_IGE", "\n")
    sv_termH2 = 0
    dge_ige = 0 
  }
  # 3. sigma_sq_AS = IGE
  if(sigma_sq_AS != 0){
    H3 = GRM / sv_Z_GRM_Zt
    sv_termH3 = sample_var(Z%*%H3%*%t(Z))
    
    ige = sigma_sq_AS*(Z%*%H3%*%t(Z))
  }else{
    cat("\t not including IGE in simulations", "\n")
    sv_termH3 = 0 
    ige = 0 
  }
  
  cat("Calculating covariance from GENETIC components", "\n")
  cov_gen = dge + dge_ige + ige
  #sigma_sq_AD*H1 + sigma_ADS*(H2%*%t(Z) + Z%*%t(H2)) + sigma_sq_AS*(Z%*%H3%*%t(Z))
  
  
  ## B. Environment components ######
  cat("Calculating sample variance of ENVIRONMENT components", "\n")
  # calculating sv_I if sigma_sq_ED or sigma_EDS
  if(sigma_sq_ED != 0 | sigma_EDS != 0){
    sv_I = sample_var(I) # DEE
  }else{
    sv_I = 0 
  }
  # calculating sv_Z_I_Zt if sigma_sq_ES or sigma_EDS
  if(sigma_sq_ES != 0 | sigma_EDS != 0){
    sv_Z_I_Zt = sample_var(Z%*%I%*%t(Z)) # IEE
  } else{
    sv_Z_I_Zt = 0
  }
  
  # Scaling only when necessary, i.e. variance component ≠ 0; otherwise setting to 0
  # 1. sigma_sq_ED = DEE
  if (sigma_sq_ED != 0 ){
    I1 = I #### 
    sv_I1 = sample_var(I1)
    
    dee = sigma_sq_ED*I1
  } else{
    cat("\t not including DEE in simulations", "\n")
    sv_I1 = 0 
    dee = 0
  }
  # 2. sigma_EDS = covariance DEE - IEE 
  if(sigma_EDS != 0){
    I2 = I / sqrt( sv_I * sv_Z_I_Zt )
    sv_termI2 = sample_var(I2%*%t(Z) + Z%*%t(I2))
    
    dee_iee = sigma_EDS*(I2%*%t(Z)  + Z%*%t(I2))
  }else{
    cat("\t no covariance DEE_IEE", "\n")
    sv_termI2 = 0
    dee_iee = 0 
  }
  # 3. sigma_sq_ES = IEE
  if(sigma_sq_ES != 0){
    I3 = I / sv_Z_I_Zt
    sv_termI3 = sample_var(Z%*%I3%*%t(Z))
    
    iee = sigma_sq_ES*(Z%*%I3%*%t(Z))
  }else{
    cat("\t not including IEE in simulations", "\n")
    sv_termI3 = 0
    iee = 0 
  }
  
  # 4. sigma_sq_C = CE
  if(sigma_sq_C != 0){
    sv_C = sample_var(C)
    #Scaling of CAGE can be done in two ways:
    # 1. scaling directly C = WxW_t ( = WxIxW_t, see below)
    # 2. scaling I in WxIxW_t (where I is an identity matrix with dim ncol(W) x ncol(W) ) 
    ## Way 1: 
    #C = W %*% t(W) # NB: this is slower cos need to do the calculation 
    #sv_C = sample_var(C)
    #C_sc = C / sv_C
    #way1 = sigma_sq_C*C_sc
    #
    ## way 2: 
    #I_w = matrix(0, nrow=ncol(W), ncol=ncol(W))
    #diag(I_w) = 1
    #W_I_Wt = W%*%I_w%*%t(W)
    #all.equal(C, W_I_Wt)
    #
    #sv_W_I_Wt = sample_var(W_I_Wt)
    #I4 = I_w / sv_W_I_Wt
    #
    #way2= sigma_sq_C*(W%*%I4%*%t(W))
    #
    #all.equal(way1, way2) ## TRUE 
    C_sc = C / sv_C 
    sv_termC_sc = sample_var(C_sc)
    
    ce = sigma_sq_C*C_sc
  }else{
    cat("\t not including CE in simulations", "\n")
    sv_C = 0
    sv_termC_sc = 0 
    ce = 0 
  }
  
  # 5. sigma_sq_Dm = ME
  if( sigma_sq_Dm != 0 ){
    sv_Dm = sample_var(Dm)
    Dm_sc = Dm / sv_Dm
    sv_termDm_sc = sample_var(Dm_sc)
    
    me = sigma_sq_Dm*Dm_sc
  }else{
    cat("\t not including ME in simulations", "\n")
    sv_Dm = 0
    sv_termDm_sc = 0
    me = 0
  }
  
  cat("Calculating covariance from ENVIRONMENT components", "\n")
  cov_env = dee + dee_iee + iee + ce + me 
  #sigma_sq_ED*I1 + sigma_EDS*(I2%*%t(Z)  + Z%*%t(I2)) + sigma_sq_ES*(Z%*%I3%*%t(Z)) + sigma_sq_C*C_sc + sigma_sq_Dm*Dm_sc
  
  ## C. phenotypic covariance ####
  start <- proc.time()[3]; 
  cat("Calculating sample variance of covariance matrix for phenotypes N x N", "\n")
  cov_y = cov_gen + cov_env
  #cov_y = round(cov_y, 10) # this was just to compare to the results from teh old function
  var_cov_y = sample_var(cov_y)
  end <- proc.time()[3]; time_cov_y = end - start; cat(time_cov_y, "\n") # time_2
  
  
  # Simulating phenotypes
  phenos=nb_phenos
  start <- proc.time()[3]; 
  cat("Simulating with MVN of ", phenos, " phenos for ", dim(cov_y)[1]," individs", "\n")
  if(phenos == 1){
    sims = matrix(t(mvrnorm(n = phenos, # the number of samples required = the number of phenos
                            mu = rep(0, dim(cov_y)[1]), # a vector giving the means of the variables - all 0
                            Sigma = cov_y)))
  }else{
    sims = t(mvrnorm(n = phenos, # the number of samples required = the number of phenos
                     mu = rep(0, dim(cov_y)[1]), # a vector giving the means of the variables - all 0
                     Sigma = cov_y))
  }
  colnames(sims) = paste0("pheno", 1:phenos)
  end <- proc.time()[3]; time_mvn = end - start; cat(time_mvn, "\n") # time_3
  
  
  # Storing results
  # #ls(pattern="var_")
  cat("Creating vars dataframe phenotypes", "\n")
  vars <- matrix(ncol=5, nrow = 9)
  rownames(vars) <- c("var_Ad1", "var_Ad1s1", "var_As1", 
                      "var_Ed1", "var_Ed1s1", "var_Es1", 
                      "var_C1", "var_Dm1", "var_y1")
  colnames(vars) <- c("set_params", "var_term", "prop_params", "sampVar_noScaled", "sampVar_Scaled")
  vars[,"set_params"] <- c(sigma_sq_AD, rho_A, sigma_sq_AS, 
                           sigma_sq_ED, rho_E, sigma_sq_ES, 
                           sigma_sq_C, sigma_sq_Dm, NA)
  vars[,"var_term"] <- c(sigma_sq_AD*sv_H1, sigma_ADS*sv_termH2, sigma_sq_AS*sv_termH3,
                         sigma_sq_ED*sv_I1, sigma_EDS*sv_termI2, sigma_sq_ES*sv_termI3,
                         sigma_sq_C*sv_termC_sc,  sigma_sq_Dm*sv_termDm_sc,
                         var_cov_y)
  # vars[,"var_term"] <- c(sigma_sq_AD*sample_var(H1), sigma_ADS*sample_var(H2%*%t(Z) + Z%*%t(H2)), sigma_sq_AS*sample_var(Z%*%H3%*%t(Z)),
  #                        sigma_sq_ED*sample_var(I1), sigma_EDS*sample_var(I2%*%t(Z) + Z%*%t(I2)), sigma_sq_ES*sample_var(Z%*%I3%*%t(Z)),
  #                        sigma_sq_C*sample_var(C%*%I4), 
  #                        var_cov_y)
  for(i in 1:(nrow(vars)-1)){
    vars[i, "prop_params"] = vars[i,"var_term"] / var_cov_y
  }
  vars[,"sampVar_noScaled"] <- c(sv_GRM, sample_var(GRM%*%t(Z) + Z%*%t(GRM)), sv_Z_GRM_Zt, 
                                 sv_I, sample_var(I%*%t(Z) + Z%*%t(I)), sv_Z_I_Zt, 
                                 sv_C, sv_Dm, 
                                 NA)
  vars[,"sampVar_Scaled"] <- c(sv_H1, sv_termH2, sv_termH3,
                               sv_I1, sv_termI2, sv_termI3,
                               sv_termC_sc, sv_termDm_sc,
                               NA)
  
  cat("Storing RHOS", "\n")
  # rhos <- c("rho_AD_12" = rho_AD, "rho_AS_12" = rho_AS, "rho_ADS_1" = rho_A1, "rho_ADS_2" = rho_A2,"rho_AD2_AS1" = rho_AD2_AS1, "rho_AD1_AS2" = rho_AD1_AS2, 
  #           "rho_ED_12" = rho_ED, "rho_ES_12" = rho_ES, "rho_EDS_1" = rho_E1, "rho_EDS_2" = rho_E2, "rho_ED2_ES1" = rho_ED2_ES1, "rho_ED1_ES2" = rho_ED1_ES2,
  #           "rho_C" = rho_C )
  rhos <- c("corr_Ad1s1" = as.numeric(rho_A), 
            "corr_Ed1s1" = as.numeric(rho_E))
  
  
  cat("Storing EXEC TIMES", "\n")
  times = c("cov_y_calc" = time_cov_y, "mvrnorm" = time_mvn)
  
  cat("Saving results", "\n")
  result = list("sims" = sims,
                "params" = vars[,c("set_params","var_term","prop_params")],
                "sample_vars" = vars[,c("sampVar_noScaled","sampVar_Scaled")],
                "rhos" = rhos,
                "times" = times)
  
  return(result)
}


# Fx. 5 ####
# simulate bivariate - include CE and ME
sim_bi <- function(half_phenos, GRM, Z, I, C, Dm, 
                   sigma_sq_AD = NULL, rho_AD = NULL, sigma_sq_AS = NULL, rho_AS = NULL, rho_A1 = NULL, rho_A2 = NULL, rho_AD2_AS1 = NULL, rho_AD1_AS2 = NULL, 
                   sigma_sq_ED = NULL, rho_ED = NULL, sigma_sq_ES = NULL, rho_ES = NULL, rho_E1 = NULL, rho_E2 = NULL, rho_ED2_ES1 = NULL, rho_ED1_ES2 = NULL,
                   sigma_sq_C = NULL, rho_C = NULL, sigma_sq_Dm = NULL, rho_Dm = NULL){ 
  ## Checking for cage - the one always needed
  if(is.null(sigma_sq_C)){stop("if don't want to simulate CE, set 'sigma_sq_C' = c(0,0)")}
  if(is.null(rho_C)){stop("if don't want to simulate corr CE, set 'rho_C' = 0")}
  if(is.null(sigma_sq_Dm)){stop("if don't want to simulate MaternalE, set 'sigma_sq_Dm' = c(0,0)")}
  if(is.null(rho_Dm)){stop("if don't want to simulate corr MaternalE, set 'rho_Dm' = 0")}
  
  if(is.null(sigma_sq_AD) | length(sigma_sq_AD) != 2){stop("if don't want to simulate DGE, set 'sigma_sq_AD' = c(0,0)")}
  if(is.null(sigma_sq_ED) | length(sigma_sq_ED) != 2){stop("if don't want to simulate DGE, set two 'sigma_sq_ED' = c(0,0)")}
  if(is.null(rho_AD) | is.null(rho_ED)){stop("if don't want to simulate DGE, set 'rho_AD' = 0 and 'rho_ED' = 0")}
  
  if(is.null(sigma_sq_AS) | length(sigma_sq_AS) != 2){stop("if don't want to simulate IGE, set two 'sigma_sq_AS' = c(0,0)")}
  if(is.null(rho_AS)){stop("if don't want to simulate IGE, set 'rho_AS' = 0")}
  if(is.null(rho_A1) | is.null(rho_A2)){stop("if don't want to simulate IGE, set 'rho_A1' = rho(Ad1, As1) = 0 and 'rho_A2' = rho(Ad2, As2) = 0")}
  if(is.null(rho_AD2_AS1) | is.null(rho_AD1_AS2)){stop("if don't want to simulate IGE, set 'rho_AD2_AS1' = 0 and 'rho_AD1_AS2' = 0")}
  
  if(is.null(sigma_sq_ES | length(sigma_sq_ES) != 2)){stop("if don't want to simulate IGE, set two 'sigma_sq_ES' = c(0,0)")}
  if(is.null(rho_ES)){stop("if don't want to simulate IGE, set 'rho_ES'")}
  if(is.null(rho_E1) | is.null(rho_E2)){stop("if don't want to simulate IGE, set 'rho_E1' = rho(Ed1, Es1) = 0 and 'rho_E2' = rho(Ed2, Es2) = 0")}
  if(is.null(rho_ED2_ES1) | is.null(rho_ED1_ES2)){stop("if don't want to simulate IGE, set 'rho_ED2_ES1' = 0 and 'rho_ED1_ES2' = 0")}
  
  
  ### Calculating covariance matrices
  # Cov matrix for CAGE component
  C_C =  build_covar_2dim(sigma_sq_C, rho_C)
  
  # Cov matrix for MATERNAL component
  C_Dm =  build_covar_2dim(sigma_sq_Dm, rho_Dm)
  
  # Covariance matrices for GENETIC components
  covs_A = build_covar_var(var_D = sigma_sq_AD, 
                           var_S = sigma_sq_AS, 
                           rhos = c(rho_AD, rho_AS, rho_A1, rho_A2, rho_AD2_AS1, rho_AD1_AS2))
  C_AD = covs_A$C_D
  C_AS = covs_A$C_S
  C_ADS = covs_A$C_DS
  C_ADS_T = t(C_ADS)
  
  # Covariance matrices for ENVIRONMENTAL components
  covs_E = build_covar_var(var_D = sigma_sq_ED, 
                           var_S = sigma_sq_ES, 
                           rhos = c(rho_ED, rho_ES, rho_E1, rho_E2, rho_ED2_ES1, rho_ED1_ES2))
  
  C_ED = covs_E$C_D
  C_ES = covs_E$C_S
  C_EDS = covs_E$C_DS
  C_EDS_T = t(C_EDS)
  
  #### Calculating sample variance
  ## A. Genetic components ######
  start <- proc.time()[3]; 
  cat("Calculating sample variance of GENETIC components", "\n")
  # calculating sv_GRM if C_AD or C_ADS (1 term)
  if(any(C_AD != 0) | any(C_ADS != 0)){
    sv_GRM = sample_var(GRM)
  }else{
    sv_GRM = 0
  }
  # calculating sv_Z_GRM_Zt if C_ADS or C_AS (2-4 term)
  if(any(C_ADS != 0) | any(C_AS != 0)){
    sv_Z_GRM_Zt = sample_var(Z%*%GRM%*%t(Z))
  }else{
    sv_Z_GRM_Zt = 0
  }
  
  # Scaling only when necessary, i.e. variance component ≠ 0; otherwise setting to 0
  # 1. C_AD = DGE
  if (any(C_AD != 0 )){
    H1 = GRM / sv_GRM
    sv_H1= sample_var(H1) # to calculate proportional
    
    dge = C_AD %x% H1
  } else{
    cat("\t not including DGE in simulations", "\n")
    sv_H1 = 0 
    dge = 0
  }
  # 2. C_ADS = covariance DGE - IGE 
  if(any(C_ADS != 0)){
    H2 = GRM / sqrt( sv_GRM * sv_Z_GRM_Zt )
    #H2 = H3 = GRM / sqrt( sv_GRM * sv_Z_GRM_Zt )
    #sv_termH2 = sample_var(H2%*%t(Z) + Z%*%t(H2))
    sv_termH2 = sample_var(H2%*%t(Z))
    
    dge_ige = C_ADS %x% (H2 %*% t(Z)) + C_ADS_T %x% (Z %*% t(H2))
  }else{
    cat("\t no covariance DGE_IGE", "\n")
    sv_termH2 = 0
    dge_ige = 0 
  }
  
  # 3. C_AS = IGE
  if (any(C_AS != 0)){
    H3 = GRM / sv_Z_GRM_Zt
    sv_termH3 = sample_var(Z%*%H3%*%t(Z))
    
    #ige = C_AS %x% (Z%*%H4%*%t(Z)) 
    ige = C_AS %x% (Z%*%H3%*%t(Z)) 
  }else{
    cat("\t not including IGE in simulations", "\n")
    sv_termH3 = 0 
    ige = 0 
  }
  
  cat("Calculating covariance from GENETIC components", "\n")
  #stopifnot(is.positive.semi.definite(round(dge, 10)))
  #stopifnot(is.positive.semi.definite(round(dge_ige, 10)))
  #stopifnot(is.positive.semi.definite(round(ige, 10)))
  cov_gen = dge + dge_ige + ige
  
  ## B. Environment components ######
  cat("Calculating sample variance of ENVIRONMENT components", "\n")
  # calculating sv_I if C_ED or C_EDS
  if(any(C_ED != 0 ) | any(C_EDS != 0)){
    sv_I = sample_var(I) # DEE
  }else{
    sv_I = 0 
  }
  # calculating sv_Z_I_Zt if C_EDS or C_ES
  if(any(C_EDS != 0) | any(C_ES != 0)){
    sv_Z_I_Zt = sample_var(Z%*%I%*%t(Z)) # IEE
  } else{
    sv_Z_I_Zt = 0
  }
  
  # Scaling only when necessary, i.e. variance component ≠ 0; otherwise setting to 0
  # 1. C_ED = DEE
  if (any(C_ED != 0 )){
    I1 = I / sv_I # this is actually equal to I itself since is an Identity matrix
    sv_I1 = sample_var(I1)
    
    dee = C_ED %x% I1
  } else{
    cat("\t not including DEE in simulations", "\n")
    sv_I1 = 0 
    dee = 0
  }
  
  # 2. C_EDS = covariance DEE - IEE 
  if(any(C_EDS != 0)){
    I2 = I / sqrt( sv_I * sv_Z_I_Zt )
    sv_termI2 = sample_var(I2%*%t(Z))
    
    dee_iee = C_EDS %x% (I2 %*% t(Z)) + C_EDS_T %x% (Z %*% t(I2))
  }else{
    cat("\t no covariance DEE_IEE", "\n")
    sv_termI2 = 0
    dee_iee = 0 
  }
  
  # 3. C_ES = IEE
  if(any(C_ES != 0)){
    I3 = I / sv_Z_I_Zt
    sv_termI3 = sample_var(Z%*%I3%*%t(Z))
    
    iee = C_ES %x% (Z%*%I3%*%t(Z))
  }else{
    cat("\t not including IEE in simulations", "\n")
    sv_termI3 = 0
    iee = 0 
  }
  
  # 4. C_C = CE
  if(any(C_C != 0)){
    sv_C = sample_var(C)
    C_sc = C / sv_C 
    sv_termC_sc = sample_var(C_sc)
    
    ce = C_C %x% C_sc
  }else{
    cat("\t not including CE in simulations", "\n")
    sv_C = 0
    sv_termC_sc = 0 
    ce = 0 
  }
  
  # 5. C_Dm = MtnE
  if( any(C_Dm != 0 )){
    sv_Dm = sample_var(Dm)
    Dm_sc = Dm / sv_Dm
    sv_termDm_sc = sample_var(Dm_sc)
    
    me = C_Dm %x% Dm_sc
  }else{
    cat("\t not including ME in simulations", "\n")
    sv_Dm = 0
    sv_termDm_sc = 0
    me = 0
  }
  
  cat("Calculating covariance from ENVIRONMENT components", "\n")
  #stopifnot(is.positive.semi.definite(round(dee,10)))
  #stopifnot(is.positive.semi.definite(round(dee_iee,10)))
  #stopifnot(is.positive.semi.definite(round(iee,10)))
  #stopifnot(is.positive.semi.definite(round(ce,10)))
  #stopifnot(is.positive.semi.definite(round(me,10)))
  cov_env = dee + dee_iee + iee + ce + me 
  
  ## C. phenotypic covariance for PHENOTYPES 2y ==> 2N x 2N ####
  start <- proc.time()[3]; 
  cat("Calculating sample variance of covariance matrix for phenotypes N x N", "\n")
  cov_y = cov_gen + cov_env
  #cov_y = round(cov_y, 10) # this was to check with old function
  #var_cov_y = sample_var(cov_y)
  end <- proc.time()[3]; time_cov_y = end - start; cat(time_cov_y, "\n") # time_2
  
  ## Check sigma is positive semi definite - checked all above already
  #stopifnot(is.positive.semi.definite(round(cov_y,10))) 
  
  ### Simulating phenotypes
  # Obtain a 2N x nb_phenos matrix - N = n of individ
  cat("Simulating with MVN of ", half_phenos, " x2 phenos for ", dim(cov_y)[1]/2," individs", "\n")
  start <- proc.time()[3]; 
  if(half_phenos == 1){
    sims = matrix(t(mvrnorm(n = half_phenos, # the number of samples required = half of number of phenos that want to
                            mu = rep(0, dim(cov_y)[1]), # a vector giving the means of the variables - all 0
                            Sigma = cov_y)))
  }else{
    sims = t(mvrnorm(n = half_phenos, # the number of samples required = half of number of phenos that want to
                     mu = rep(0, dim(cov_y)[1]), # a vector giving the means of the variables - all 0
                     Sigma = cov_y))
  }
  end <- proc.time()[3]; time_mvn = round(end - start, 4); cat(time_mvn, "\n") # time_3
  
  N = dim(GRM)[1]
  # Converting 2N x half_phenos matrix in a N x 2*nb_phenos matrix
  cat("Splitting phenotypes", "\n")
  Y <- matrix( unlist(apply(sims, MARGIN = 2, #1 rows, 2 cols
                            FUN = function(x){y1 <- x[1:N]; y2 <- x[(N+1):(N*2)]; return(list("y1"=y1, "y2"=y2))})),  
               nrow = dim(GRM)[1])
  #c(dim(GRM)[1], half_phenos*2) )
  rownames(Y) = rownames(GRM)
  colnames(Y) = paste0("pheno", 1:(half_phenos*2))
  
  # Overall
  cat("Calculating sample variance of intermediate terms", "\n")
  # var_cov_y = sample_var(cov_y) # This is the sample_var of the matrix 2N x 2N
  cov_y1 <- cov_y[1:N,1:N] # Cov matrix of pheno 1 (taking upper left matrix of the 2N x 2N)
  cov_y2 <- cov_y[(N+1):(N*2),(N+1):(N*2)]# Cov matrix of pheno 2 (taking lower right matrix of the 2N x 2N)
  sv_cov_y1 = sample_var(cov_y1)
  sv_cov_y2 = sample_var(cov_y2)
  
  
  #ls(pattern="var_")
  cat("Storing VARIANCES", "\n")
  vars <- matrix(ncol=5, nrow = 18)
  #rownames(vars) <- c("DGE_1", "DGE_2", "DG2_IG1","DG1_IG2", "IGE_1","IGE_2",
  #                    "DEE_1", "DEE_2","DE2_IE1", "DE1_IE2","IEE_1", "IEE_2",
  #                    "CE_1", "CE_2","var_y1", "var_y2")
  #rownames(vars) <- c("var_Ad1", "var_Ad2", "var_Ad2s1","var_Ad1s2", "var_As1","var_As2",
  #"var_Ed1", "var_Ed2","var_Ed2s1", "var_Ed1s2","var_Es1", "var_Es2",
  #"var_C1", "var_C2","var_Dm1","var_Dm2", "var_y1", "var_y2")
  rownames(vars) <- c("var_Ad1", "var_Ad2", "var_Ad2s1","var_Ad1s2", "var_As1","var_As2",
                      "var_Ed1", "var_Ed2","var_Ed2s1", "var_Ed1s2","var_Es1", "var_Es2",
                      "var_C1", "var_C2", "var_Dm1", "var_Dm2","var_y1", "var_y2")
  
  colnames(vars) <- c("set_params", "var_term","prop_params", "sampVar_noScaled", "sampVar_Scaled")
  vars[,"set_params"] <- c(sigma_sq_AD[1], sigma_sq_AD[2], C_ADS[1,2],  C_ADS[2,1],sigma_sq_AS[1], sigma_sq_AS[2], 
                           sigma_sq_ED[1], sigma_sq_ED[2], C_EDS[2,1], C_EDS[1,2], sigma_sq_ES[1], sigma_sq_ES[2], 
                           sigma_sq_C[1], sigma_sq_C[2], sigma_sq_Dm[1], sigma_sq_Dm[2], NA, NA)
  
  vars[,"var_term"] <- c(rep(sample_var(C_AD)*sv_H1, 2), sample_var(C_ADS)*sv_termH2, sample_var(C_ADS_T)*sv_termH2, rep(sample_var(C_AS)*sv_termH3, 2),
                         rep(sample_var(C_ED)*sv_I1, 2), sample_var(C_EDS)*sv_termI2, sample_var(C_EDS_T)*sv_termI2, rep(sample_var(C_ES)*sv_termI3, 2),
                         rep(sample_var(C_C)*sv_termC_sc, 2), rep(sample_var(C_Dm)*sv_termDm_sc, 2), sv_cov_y1, sv_cov_y2)
  
  # Adding proportional terms 
  to_y1 <- rownames(vars)[grep("1", rownames(vars))]
  to_y2 <- rownames(vars)[grep("2", rownames(vars))]
  
  for(i in 1:(nrow(vars)) ){
    if(rownames(vars)[i] %in% to_y1){
      vars[i, "prop_params"] = vars[i,"set_params"] / sv_cov_y1 # This was initially "var_terms" - not sure why
    }
    if(rownames(vars)[i] %in% to_y2){
      vars[i, "prop_params"] = vars[i,"set_params"] / sv_cov_y2 # This was initially "var_terms" - not sure why
    }
    #vars[i, "prop_params"] = vars[i,"var_term"] / var_cov_y
  }
  
  
  # rep(sample_var(GRM%*%t(Z)),2) cos sample_var(GRM%*%t(Z)) == sample_var(Z%*%t(GRM)) is T
  vars[,"sampVar_noScaled"] <- c(rep(sv_GRM,2), rep(sample_var(GRM%*%t(Z)),2), rep(sv_Z_GRM_Zt,2),
                                 rep(sv_I,2), rep(sample_var(I%*%t(Z)), 2), rep(sv_Z_I_Zt, 2),
                                 rep(sv_Dm, 2), rep(sv_C, 2), NA, NA)
  
  vars[,"sampVar_Scaled"] <- c(rep(sv_H1, 2), sv_termH2, sv_termH2, rep(sv_termH3, 2),
                               rep(sv_I1, 2), sv_termI2, sv_termI2, rep(sv_termI3, 2),
                               rep(sv_termDm_sc, 2), rep(sv_termC_sc, 2), NA,NA)
  
  cat("Storing RHOS", "\n")
  # rhos <- c("rho_AD_12" = rho_AD, "rho_AS_12" = rho_AS, "rho_ADS_1" = rho_A1, "rho_ADS_2" = rho_A2,"rho_AD2_AS1" = rho_AD2_AS1, "rho_AD1_AS2" = rho_AD1_AS2, 
  #           "rho_ED_12" = rho_ED, "rho_ES_12" = rho_ES, "rho_EDS_1" = rho_E1, "rho_EDS_2" = rho_E2, "rho_ED2_ES1" = rho_ED2_ES1, "rho_ED1_ES2" = rho_ED1_ES2,
  #           "rho_C" = rho_C )
  rhos <- c("corr_Ad1d2" = rho_AD, "corr_As1s2" = rho_AS, "corr_Ad1s1" = rho_A1, "corr_Ad2s2" = rho_A2,"corr_Ad2s1" = rho_AD2_AS1, "corr_Ad1s2" = rho_AD1_AS2, 
            "corr_Ed1d2" = rho_ED, "corr_Es1s2" = rho_ES, "corr_Ed1s1" = rho_E1, "corr_Ed2s2" = rho_E2, "corr_Ed2s1" = rho_ED2_ES1, "corr_Ed1s2" = rho_ED1_ES2,
            "corr_Dm1Dm2" = rho_Dm,"corr_C1C2" = rho_C)
  
  #cat("Storing EXEC TIMES")
  #times = c("sampVar_noScaled" = time_sv_basic, "cov_y_calc" = time_cov_y, "mvrnorm" = time_mvn,"sampVar_Scaled" = time_sv_int)
  cat("Storing EXEC TIMES", "\n")
  times = c("cov_y_calc" = time_cov_y, "mvrnorm" = time_mvn)
  
  ## Saving results to return
  cat("Saving results", "\n")
  result = list("sims" = Y,
                "params" = vars[,c("set_params", "var_term","prop_params")],
                "sample_vars" = vars[,c("sampVar_noScaled","sampVar_Scaled")], 
                "rhos" = rhos, 
                "times" = times)
  
  ### Consider doing like in univar: 
  # result = list("sims" = sims,
  #               "params" = vars[,c("set_params","var_term","prop_params")],
  #               "sample_vars" = vars[,c("sampVar_noScaled","sampVar_Scaled")])
  return(result)
}


### START WITH SIMULATIONS
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
