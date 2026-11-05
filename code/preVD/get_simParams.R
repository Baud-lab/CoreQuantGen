#!/usr/bin/env Rscript

# Read simulations
library(rhdf5)
# file where where simulations

args = commandArgs(trailingOnly=TRUE)

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
