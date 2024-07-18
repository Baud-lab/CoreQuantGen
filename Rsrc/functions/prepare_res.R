# My results
# Fx 1 : preparing results, from original dataset, removing unused columns (nocol)
# TODO: put this to source if it works !!!!!!
prepare_res <- function(res, nocol = c('sample_size','sample_size_all','covariates_names', 'conv', 'LML')){
  # assigning taskID = name phenoytpe1_phenotype2
  #     TODO: check how it works with univariate, imagine it would be something like trait1_None, trait2_None...
  #     if want something different can do 
  #     if ("trait2" %in% nocol) {res[,"taskID"] = res[,"trait1"] # or res[,"taskID"] = paste0(res[,"trait1"], "univariate")}
  order_cols = c("taskID",colnames(res))
  res[,"taskID"] = paste(res[,"trait1"], res[,"trait2"], sep='_') # NB: like this also have the same name that is saved as _est.txt _STE.txt output files
  res[,"taskID"] = paste(res[,"trait1"], res[,"trait2"], sep='_') # NB: like this also have the same name that is saved as _est.txt _STE.txt output files
  res = res[,order_cols]

  # filtering out unwanted columns
  res = res[,! colnames(res) %in% nocol]
  # removing columns with all -999 i.e. NAs
  NAs <-  apply(res, 2, FUN = function(res) all(res == -999))
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
message("\nDictionary STE_name - param_name created")



### col_est and col_STE for 26/02
#colest = c('trait1', 'trait2', 'sample_size1', 'sample_size1_cm',
#           'sample_size2', 'sample_size2_cm', #'cm2for1', 'cm1for2', 'sample_size_1n2', #'cm2for1', 'cm1for2', are new!
#           'covariates_names', 'conv', 'LML',
#           'prop_Ad1', 'prop_Ad2','prop_As1', 'prop_As2',
#           'corr_Ad1d2', 'corr_Ad1s1', 'corr_Ad1s2', 'corr_Ad2s1','corr_Ad2s2', 'corr_As1s2',
#           'prop_Ed1', 'prop_Ed2','prop_Es1', 'prop_Es2',
#           'corr_Ed1d2', 'corr_Ed1s1','corr_Ed1s2', 'corr_Ed2s1', 'corr_Ed2s2', 'corr_Es1s2',
#           'prop_C1', 'prop_C2','corr_C1C2', 
#           'prop_Dm1', 'prop_Dm2', 'corr_Dm1Dm2', # this is new
#           'total_var1', 'total_var2') #ADD TOTAL GEN VAR if care about it

# colnames for STE as saved from exp_bivar12
#colste = c('trait1', 'trait2', 
#           'STE_Ad1', 'STE_Ad2', 'STE_As1', 'STE_As2', 
#           'STE_Ad1d2', 'STE_Ad1s1', 'STE_Ad1s2', 'STE_Ad2s1', 'STE_Ad2s2', 'STE_As1s2',
#           'STE_Ed1', 'STE_Ed2', 'STE_Es1', 'STE_Es2', 
#           'STE_Ed1d2', 'STE_Ed1s1', 'STE_Ed1s2', 'STE_Ed2s1', 'STE_Ed2s2', 'STE_Es1s2',
#           'STE_Dm1', 'STE_Dm2', 'STE_Dm1Dm2',
#           'STE_C1C2', # NB: because of the way we calculate STE we cannot get STE for C1/C2; 
#           'STE_totv1', 'STE_totv2',
#           'corParams_Ad1_As1', 'corParams_Ed1_Es1', 'corParams_Ed1_Dm1', 'corParams_Es1_Dm1', 
#           'corParams_Ad2_As2', 'corParams_Ed2_Es2', 'corParams_Ed2_Dm2', 'corParams_Es2_Dm2')
#
