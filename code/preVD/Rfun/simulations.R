#### FUNCTIONS USED IN SIMULATIONS
library("matrixcalc")
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
