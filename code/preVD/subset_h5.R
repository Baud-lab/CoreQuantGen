#!/usr/bin/env Rscript
# subsetting dataset for a trial
# versions (subgroup names) are taken from the original h5
# required groups: phenotypes, cages, GRM; optional: covariates, dam, sex_cov, subsets, GRMs_LOCO
# e.g. Rscript subset_h5.R --h5file ./HSmice.h5 --out ./HSmice_mini200.h5 --n 200 --n_phenos 5
suppressMessages(library("rhdf5"))
suppressMessages(library("optparse")) # ADDED: options from command line

# CHANGED: options from command line instead of hard-coded
option_list = list(make_option("--h5file", action="store", default=NA, type='character',
                               help="Path to original .h5 file [required]"),
                   make_option("--out", action="store", default=NA, type='character',
                               help="Path to subsetted .h5 file to write [required]"),
                   make_option("--n", action="store", default=200, type='integer',
                               help="Number of individuals to keep [default= %default]"),
                   make_option("--n_phenos", action="store", default=5, type='integer',
                               help="Number of phenotypes to keep, chosen randomly [default= %default]"))
opt = parse_args(OptionParser(option_list=option_list))
if(is.na(opt$h5file) | is.na(opt$out)){stop("--h5file and --out are required")}
if(file.exists(opt$out)){stop("output file already exists: ", opt$out)}

# read .h5
# CHANGED: read only one version per group (the first one, if there are more), instead of the whole file
#          the list `h5` keeps the same structure as h5read(opt$h5file, "/"), with one version per group
#h5 = h5read(opt$h5file, "/")
content = h5ls(opt$h5file, recursive = 2)
h5 = list()
used_groups = c("phenotypes", "covariates", "cages", "dam", "GRM", "GRMs_LOCO", "sex_cov", "subsets")
other_groups = setdiff(content$name[content$group == "/"], used_groups)
if(length(other_groups) > 0){cat("NB: not copied:", other_groups, "\n")}
for(g in intersect(content$name[content$group == "/"], used_groups)){
  if(g == "subsets"){ # subsets has no versions: all subsets are kept
    h5[[g]] = h5read(opt$h5file, g)
    next
  }
  versions = content$name[content$group == paste0("/", g)]
  if(length(versions) > 1){cat("NB:", g, "has", length(versions), "versions (", versions, "), using:", versions[1], "\n")}
  h5[[g]] = list()
  h5[[g]][[versions[1]]] = h5read(opt$h5file, paste0(g, "/", versions[1]))
}
h5closeAll()
#names = h5ls(opt$h5file, recursive = F)$name

# sample samples IDs so that have cages with more than 1
# 1. subsetting cages
cat("starting from subsetting cages:\n")
orcages = h5$cages[[names(h5$cages)]]$array;
names(orcages) = h5$cages[[names(h5$cages)]]$sample_ID
samplesIDs = h5$cages[[names(h5$cages)]]$sample_ID
if(opt$n > length(samplesIDs)){stop("--n (", opt$n, ") is larger than the number of individuals (", length(samplesIDs), ")")} # ADDED
set.seed(100)
subcage = sample(unique(orcages), opt$n / ceiling(mean(table(orcages))))
cages = orcages[orcages %in% subcage]
cages = cages[seq_len(min(length(cages), opt$n))] # ADDED: in case selected cages have more than n individuals
length(cages)
# top up to the number of IDs wanted
cages = c(cages, orcages[sample(samplesIDs[!samplesIDs %in% names(cages)], opt$n - length(cages))] )
length(cages)
table(table(cages))
# get subIDs to filter all the others
subIDs = names(cages)

# 2. subsetting grm
cat("subsetting GRM:\n")
grm = h5$GRM[[names(h5$GRM)]]$matrix; rownames(grm) = colnames(grm) = h5$GRM[[names(h5$GRM)]]$row_header$sample_ID
grm[1:10,1:10]
grm = grm[subIDs, subIDs]
dim(grm)

# 3. subsetting phenotypes
cat("subsetting phenotypes:\n")
phenos = h5$phenotypes[[names(h5$phenotypes)]]$matrix;
rownames(phenos) = h5$phenotypes[[names(h5$phenotypes)]]$row_header$sample_ID
# selecting only n_phenos phenotypes, randomly
subpheno = sample(1:ncol(phenos), min(opt$n_phenos, ncol(phenos))) # CHANGED: n_phenos from options
phenos = phenos[subIDs, subpheno, drop = FALSE] # CHANGED: drop = FALSE so that it stays a matrix with 1 phenotype
head(phenos)

# CHANGED: version from the h5 (was data_bcNcovariates) and cbind so that it stays a matrix with 1 phenotype
phenocols = do.call(cbind, lapply(h5$phenotypes[[names(h5$phenotypes)]]$col_header, function(n) n[subpheno]))
colnames(phenocols)

# to check
#subpheno
#orpheno = h5$phenotypes[[names(h5$phenotypes)]]$matrix;
#rownames(orpheno) = h5$phenotypes[[names(h5$phenotypes)]]$row_header$sample_ID;
#colnames(orpheno) = h5$phenotypes[[names(h5$phenotypes)]]$col_header$phenotype_ID
#head(orpheno[subIDs, subpheno])

# 4. subsetting covariates
if(any(names(h5) == "covariates")){ # ADDED: covariates are optional
  cat("subsetting covariates:\n")
  covs = h5$covariates[[names(h5$covariates)]]$matrix;
  rownames(covs) = h5$covariates[[names(h5$covariates)]]$row_header$sample_ID
  colnames(covs) = h5$covariates[[names(h5$covariates)]]$col_header$covariate_ID
  # ADDED: keep only covariates used by the selected phenotypes, if covariatesUsed is there; otherwise keep all
  if("covariatesUsed" %in% colnames(phenocols)){
    covused = unique(unlist(strsplit(phenocols[,"covariatesUsed"], ",")))
    covused = intersect(covused, colnames(covs))
  }else{
    covused = colnames(covs)
  }
  covs = covs[subIDs, covused, drop = FALSE]
  head(covs)
}

# 5. subsetting dam
if(any(names(h5) == "dam")){
  cat("subsetting dam:\n")
  dam = h5$dam[[names(h5$dam)]]$array;
  names(dam) = h5$dam[[names(h5$dam)]]$sample_ID
  dam = dam[subIDs]
  print(table(dam))
}

# 6. subsetting sex_cov
if(any(names(h5) == "sex_cov")){
  cat("subsetting sex_cov:\n")
  sex_cov = h5$sex_cov[[names(h5$sex_cov)]]$array;
  names(sex_cov) = h5$sex_cov[[names(h5$sex_cov)]]$sample_ID
  sex_cov = sex_cov[subIDs]
  print(table(sex_cov))
}

# 7. ADDED: subsetting subsets - each subset keeps only the selected IDs; empty subsets are dropped
if(any(names(h5) == "subsets")){
  cat("subsetting subsets:\n")
  subsets = lapply(h5$subsets, function(s) s[s %in% subIDs])
  subsets = subsets[sapply(subsets, length) > 0]
  print(sapply(subsets, length))
}

# 8. ADDED: subsetting GRMs_LOCO - one GRM per chromosome
if(any(names(h5) == "GRMs_LOCO")){
  cat("subsetting GRMs_LOCO:\n")
  grms_loco = lapply(h5$GRMs_LOCO[[names(h5$GRMs_LOCO)]], function(g){
    m = g$matrix; rownames(m) = colnames(m) = g$row_header$sample_ID
    m[subIDs, subIDs]
  })
  print(names(grms_loco))
}

# rewrite to subset file
fid = opt$out
cat("storing file to:", fid, "\n")

h5createFile(fid)

######## 1. PHENOTYPES # to be saved in format:  f['phenotypes'][self.phenos_version]['matrix'][:].T
str(phenos)
pheno_version = names(h5$phenotypes)
h5createGroup(fid,"phenotypes")
h5createGroup(fid,paste0("phenotypes/",pheno_version,"/")) # noBatch = pheno_version
#Sample IDs - group $row_headers
h5createGroup(fid,paste0("phenotypes/",pheno_version,"/row_header"))
max_size = max(nchar(rownames(phenos)))
h5createDataset(file=fid, dataset=paste0("phenotypes/",pheno_version,"/row_header/sample_ID"),
                dims=dim(phenos)[1], storage.mode='character', size=max_size)
h5write(obj=rownames(phenos), file=fid, name=paste0("phenotypes/",pheno_version,"/row_header/sample_ID") )

# Phenotype IDs
str(phenocols[,"phenotype_ID"])
h5createGroup(fid,paste0("phenotypes/",pheno_version,"/col_header"))
max_size = max(nchar(phenocols[,"phenotype_ID"]))
h5createDataset(file=fid, dataset=paste0("phenotypes/",pheno_version,"/col_header/phenotype_ID"),
                dims=dim(phenos)[2], storage.mode='character', size=max_size)
h5write(obj=phenocols[,"phenotype_ID"], file=fid, name=paste0("phenotypes/",pheno_version,"/col_header/phenotype_ID"))

# Phenotype matrix
h5write(obj=phenos,file=fid,name=paste0("phenotypes/",pheno_version,"/matrix"))

# Macrophenotypes, if any
if("macrophenotypes" %in% colnames(phenocols)){
  str(phenocols[,"macrophenotypes"])
  max_size = max(nchar(phenocols[,"macrophenotypes"]))
  h5createDataset(file=fid, dataset=paste0("phenotypes/",pheno_version,"/col_header/macrophenotypes"),
                  dims=length(phenocols[,"macrophenotypes"]), storage.mode='character', size=max_size)
  h5write(obj=phenocols[,"macrophenotypes"], file=fid, name=paste0("phenotypes/",pheno_version,"/col_header/macrophenotypes"))
}

# Used covariates per pheno, if any
if("covariatesUsed" %in% colnames(phenocols)){ # ADDED: only if present
  str(phenocols[,"covariatesUsed"])
  max_size = max(nchar(phenocols[,"covariatesUsed"]))
  h5createDataset(file=fid, dataset=paste0("phenotypes/",pheno_version,"/col_header/covariatesUsed"),
                  dims=length(phenocols[,"covariatesUsed"]), storage.mode='character', size=max_size)
  h5write(obj=phenocols[,"covariatesUsed"], file=fid, name=paste0("phenotypes/",pheno_version,"/col_header/covariatesUsed"))
}

h5ls(fid)
## reading phenos
# fid = ".../...h5"
# h5 = h5read(fid,"/")
# phenos= h5$...$matrix; colnames(phenos) = h5$...$col_header$phenotype_ID; rownames(phenos) = h5$...$row_header$sample_ID


######## 2. COVARIATES # to be saved in format:  f['covariates'][self.covs_version]['matrix'][:].T
if(any(names(h5) == "covariates")){ # ADDED: covariates are optional
  str(covs)
  cov_version = names(h5$covariates)
  h5createGroup(fid,"covariates")
  h5createGroup(fid, paste0("covariates/",cov_version,"/"))
  #Sample IDs - group $row_headers
  h5createGroup(fid,paste0("covariates/",cov_version,"/row_header"))
  max_size = max(nchar(rownames(covs)))
  h5createDataset(file=fid, dataset=paste0("covariates/",cov_version,"/row_header/sample_ID"),
                  dims=dim(covs)[1], storage.mode='character', size=max_size)
  h5write(obj=rownames(covs), file=fid, name=paste0("covariates/",cov_version,"/row_header/sample_ID"))
  #Cov names
  h5createGroup(fid,paste0("covariates/",cov_version,"/col_header"))
  max_size = max(nchar(colnames(covs)))
  h5createDataset(file=fid, dataset=paste0("covariates/",cov_version,"/col_header/covariate_ID"),
                  dims=dim(covs)[2], storage.mode='character', size=max_size)
  h5write(obj=colnames(covs), file=fid, name=paste0("covariates/",cov_version,"/col_header/covariate_ID"))
  #Covs values
  h5write(obj=covs,file=fid,name=paste0("covariates/",cov_version,"/matrix"))
  #h5ls(fid)
}

######## 3. CAGES - to be saved in format:  f['cages'][self.cage_version]['array'].asstr()[:] #self
str(cages)
length(unique(cages))
cage_version = names(h5$cages)
h5createGroup(fid,"cages")
h5createGroup(fid, paste0("cages/",cage_version, "/"))

# Cages - `cages`
max_size = max(nchar(cages))
h5createDataset(file=fid, dataset=paste0("cages/",cage_version, "/array"),
                dims=length(cages), storage.mode='character', size=max_size)
h5write(obj=as.character(cages), file=fid, name=paste0("cages/",cage_version, "/array"))
# Cages_IDs - `names(cages)`
max_size = max(nchar(names(cages)))
h5createDataset(file=fid, dataset=paste0("cages/",cage_version, "/sample_ID"),
                dims=length(cages), storage.mode='character', size=max_size)
h5write(obj=as.character(names(cages)), file=fid, name=paste0("cages/",cage_version, "/sample_ID"))


######## 4. DAM - to be saved in format:  f['dam'][self.dam_version]['array'].asstr()[:] #self
if(any(names(h5) == "dam")){
  str(dam)
  length(unique(dam))
  dam_version = names(h5$dam) # simulated
  h5createGroup(fid,"dam")
  h5createGroup(fid,paste0("dam/",dam_version, "/"))

  # Dams - `dam`
  max_size = max(nchar(dam))
  h5createDataset(file=fid, dataset=paste0("dam/",dam_version,"/array"),
                  dims=length(dam), storage.mode='character', size=max_size)
  h5write(obj=as.character(dam), file=fid, name=paste0("dam/",dam_version,"/array"))
  # Dams_IDs - `names(dam)`
  max_size = max(nchar(names(dam)))
  h5createDataset(file=fid, dataset=paste0("dam/",dam_version,"/sample_ID"),
                  dims=length(dam), storage.mode='character', size=max_size)
  h5write(obj=as.character(names(dam)), file=fid, name=paste0("dam/",dam_version,"/sample_ID"))
}


######## 5. GRM -  to be saved in format:  f['GRM'][self.GRM_version]['matrix'][:]
str(grm)
GRM_version = names(h5$GRM)
h5createGroup(fid,"GRM")
h5createGroup(fid, paste0("GRM/",GRM_version,"/") )

h5createGroup(fid,paste0("GRM/",GRM_version,"/row_header"))
max_size = max(nchar(rownames(grm)))
h5createDataset(file=fid, dataset=paste0("GRM/",GRM_version,"/row_header/sample_ID"),
                dims=dim(grm)[1], storage.mode='character', size=max_size)
h5write(obj=rownames(grm), file=fid, name=paste0("GRM/",GRM_version,"/row_header/sample_ID"))
h5write(obj=grm,file=fid,name=paste0("GRM/",GRM_version,"/matrix"))
# NB: this comes with a warning:
#   You created a large dataset with compression and chunking.
#   The chunk size is equal to the dataset dimensions.
#   If you want to read subsets of the dataset, you should testsmaller chunk sizes to improve read times.


######## 6. SEX COV -  f['sex_cov'][self.sex_version].asstr()[:] #####
if(any(names(h5) == "sex_cov")){ # ADDED: sex_cov is optional
  str(sex_cov)
  sex_version = names(h5$sex_cov)
  h5createGroup(fid,"sex_cov")
  h5createGroup(fid,paste0("sex_cov/", sex_version))

  # Sex_cov - `sex_cov`
  max_size = max(nchar(sex_cov))
  h5createDataset(file=fid, dataset=paste0("sex_cov/", sex_version, "/array"),
                  dims=length(sex_cov), storage.mode='character', size=max_size)
  h5write(obj=as.character(sex_cov), file=fid, name= paste0("sex_cov/", sex_version, "/array"))
  # Sex_IDs - `names(sex_cov)`
  max_size = max(nchar(names(sex_cov)))
  h5createDataset(file=fid, dataset=paste0("sex_cov/", sex_version, "/sample_ID"),
                  dims=length(sex_cov), storage.mode='character', size=max_size)
  h5write(obj=as.character(names(sex_cov)), file=fid, name=paste0("sex_cov/", sex_version, "/sample_ID"))
}

######## 7. ADDED: SUBSETS - f['subsets'][self.subset].asstr()[:]
if(any(names(h5) == "subsets") && length(subsets) > 0){
  h5createGroup(fid,"subsets")
  for(s in names(subsets)){
    max_size = max(nchar(subsets[[s]]))
    h5createDataset(file=fid, dataset=paste0("subsets/", s),
                    dims=length(subsets[[s]]), storage.mode='character', size=max_size)
    h5write(obj=as.character(subsets[[s]]), file=fid, name=paste0("subsets/", s))
  }
}

######## 8. ADDED: GRMs_LOCO - f['GRMs_LOCO'][self.GRM_version]['chr<N>']['matrix'][:]
if(any(names(h5) == "GRMs_LOCO")){
  GRM_LOCO_version = names(h5$GRMs_LOCO)
  h5createGroup(fid,"GRMs_LOCO")
  h5createGroup(fid, paste0("GRMs_LOCO/",GRM_LOCO_version))
  for(chr in names(grms_loco)){
    path = paste0("GRMs_LOCO/",GRM_LOCO_version,"/",chr)
    h5createGroup(fid, path)
    h5createGroup(fid, paste0(path,"/row_header"))
    max_size = max(nchar(rownames(grms_loco[[chr]])))
    h5createDataset(file=fid, dataset=paste0(path,"/row_header/sample_ID"),
                    dims=nrow(grms_loco[[chr]]), storage.mode='character', size=max_size)
    h5write(obj=rownames(grms_loco[[chr]]), file=fid, name=paste0(path,"/row_header/sample_ID"))
    h5write(obj=grms_loco[[chr]], file=fid, name=paste0(path,"/matrix"))
  }
}

h5ls(fid)
