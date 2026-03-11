#!/usr/bin/env Rscript

# Read simulations
library(rhdf5)
# file where where simulations

args = commandArgs(trailingOnly=TRUE)

#fid = "/users/abaud/htonnele/HSmice/output/HSmice_bivar_600_IDn_seed50.h5" # bivar
#fid = "/users/abaud/htonnele/HSmice/output/simulations/test1/univariate/HSmice_200I_549cages_seed55.h5" # univar
#fid = "/users/abaud/htonnele/HSmice/output/simulations/test0/univariate/HSmice_200D_549cages_seed23.h5"
#fid = "/users/abaud/htonnele/HSmice/output/simulations/univariate/CE_DGE/CE/0.2/HSmice_uni_200IGE_549cages_seed21.h5"
#fid="/nfs/users/abaud/htonnele/HSmice/tests/HSmice_bi_20IGE_100cages_seed301.h5"
fid=args[1]
#h5dump(fid, load=F)

h5 = h5read(fid,"/")

# Getting RHOS
## univariate - stored cov instead of rho -> have to calculate
params =  h5$sim_params$matrix; rownames(params) = h5$sim_params$row_header; colnames(params) = h5$sim_params$col_header
#params

rhos = h5$rhos$matrix; rownames(rhos) = h5$rhos$names; colnames(rhos) = "set_params"
#rhos
rhos = cbind(rhos, "var_term" = rep(NA, nrow(rhos)), "prop_params"= rep(NA, nrow(rhos)))
all = rbind(params, rhos)
all
write.table(all, file=args[2], quote=F, sep = "\t")
