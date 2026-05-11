#!/usr/bin/env Rscript

library(rhdf5)
args = commandArgs(trailingOnly=TRUE)

h5file = args[1]
phenov = args[2]
outfile = gsub("\\.h5", paste0("_", phenov, "_list.txt"), h5file)

pheno_names = h5read(h5file, paste0("/phenotypes/", phenov, "/col_header/phenotype_ID"))

write.table(pheno_names, file = outfile, quote = F, col.names = F, row.names = F)
