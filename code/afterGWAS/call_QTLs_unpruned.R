library(rhdf5)
library(parallel)

pval_dir = "~/PRJs/P50_HSrats/gwas/output/pvalues_LOCO/univariate/P50_Rn7_direct/sgb_cecal/dGWAS/P50_Rn7_NY_DGE_cageEffect_maternalEffect/"
alpha = 0.0001
outdir = "/no_backup/abaud/data/secondary/HSrats_gwas/rn7/shotgun_NY_DGE_cageEffect_maternalEffect/"
outsnps = file.path(outdir, paste0("sgb_cecal_snps_unpruned.RData"))
outqtls = file.path(outdir, paste0("sgb_cecal_QTLs_a",alpha,"_unpruned.tsv"))
cumpos_file = "/users/abaud/data/secondary/cumpos_P50_rats_Rn7.RData" 

# Getting all files, one per phenotype
files = list.files(pval_dir, pattern = ".h5", full.names = T)
#file = files[367] # Paraprevotella associated
length(files)
names(files) = gsub(".h5","",basename(files), fixed = T)
head(files)  

# Loading cum positions - need to plot gwas
obj = load(cumpos_file); obj
str(cumpos)

#### Function to group results all in one 
get_snps = function(file, chrs=1:20, logp_thr = 3){
  openh5 = try(h5read(file, '/'))
  #pheno_name = sub('.h5','',basename(HT_pvals),fixed=T)
  #print(pheno_name)
  
  if (inherits(openh5,'try-error')) return('Error reading file') # inherits test for a class - example inherits(my_h5, "list") if file opened correctly
  
  cat("file", file, "read successfully\n")
  # initialising dataset
  all_snps = as.data.frame(matrix(nrow=0,ncol=4)); colnames(all_snps) = c("chr","pos","pvalue","cumpos")
  # getting results per chromosome
  for (chr in chrs){
    chr_res = try(h5read(file, paste0('/chr',chr)))
    if (inherits(chr_res,'try-error')) return(paste0('Error reading', chr,'in file')) # inherits test for a class - example inherits(my_h5, "list") if file opened correctly
    
    if (!paste0('pvalues_chr',chr) %in% names(chr_res)) return(paste0('Error pvalues missing for chr ',chr))
    
    list_oi = grep(paste0("chr", chr,"$"), names(chr_res), value=T) # dollar symbol is to define the ending so that chr1 is only chr1 and not chr11, chr12...
    list_chr = lapply(chr_res[list_oi], as.numeric)
    #str(list_chr)
    df_chr = as.data.frame(do.call(cbind, list_chr))
    #str(df_chr)
    colnames(df_chr) = gsub(paste0("s_chr",chr), "", colnames(df_chr)) # beta, chr, pos, pvalue
    
    # Adding info on cumulative positions
    cumpos_chr = c(cumpos[[paste('cumpos_chr',chr,sep='')]])
    if (nrow(df_chr) != length(cumpos_chr)) stop('pb with cumpos')
    # Check they are in the right order
    if (min(cumpos_chr) != cumpos_chr[1]){
      cumpos_chr = cumpos_chr[order(cumpos_chr)]
    }
    if (min(df_chr[,"pos"]) != df_chr[1,"pos"]){
      df_chr = df_chr[order(as.numeric(df_chr[,"pos"])),]
    }
    df_chr[,"cumpos"] = cumpos_chr

    # binding result to general dataframe
    all_snps = rbind(all_snps, df_chr[colnames(all_snps)])
    # This should be true for chr 1
    #all(all_snps[,"pos"] == all_snps[,"cumpos"])
  }
  # adding id and logP columns
  all_snps[,"id"] = paste(all_snps[,"chr"], all_snps[,"pos"], sep="_")
  dim(all_snps)
  #[1] 4880609       4 # for P50_Rn7_unpruned, corresponds to wc -l of chr 1:20 in /users/abaud/data/secondary/P50_HSrats/dosages/P50_Rn7_unpruned_perChr/
  
  # handle missing (-999) P values
  missing_chrs = unique(na.omit(all_snps[all_snps[,"pvalue"] == (-999),'chr']))
  if (all(all_snps[,"chr"] %in% missing_chrs)) {
    cat('No non-missing P values for file', file, "\n")
    return(NULL)
  } else if (length(missing_chrs)>0){
    cat("P values for chromosomes(s)", missing_chrs, "are missing\n")
  } else{
    cat("P values present for all chromosomes\n")
  }
  all_snps = all_snps[!all_snps[,"chr"] %in% missing_chrs,]
  
  all_snps[,"logP"] = -log10(all_snps[,'pvalue'])
  # keep only snps for which logP >= p_thr, otherwise too heavy
  all_snps = all_snps[all_snps[,'logP'] >= logp_thr, ]
  all_snps = all_snps[order(all_snps[,"pvalue"]),]
  #if (min(all_snps[,'pvalue'])>alpha) {cat("no pvalue < alpha",alpha,"\n"); return(NULL)} # the first one is the min one because of the order
  #all_snps = all_snps[all_snps$pvalue<=alpha,]
  
  return(all_snps)
}

#pheno_name = gsub(".h5","",basename(file))
#### Function to get snps loci
#all_snps = res[[3]]
get_QTLs = function(all_snps, windowsize=1500000, alpha=0.0001){ #pheno_name
  #str(all_snps)
  if(is.null(all_snps)) return(NULL)
  if(is.character(all_snps)) {if(grepl("Error", all_snps)) return(all_snps)}
  if (min(all_snps[,'pvalue'])>alpha) {cat("no pvalue < alpha",alpha,"\n"); return(NULL)} # the first one is the min one because of the order
  all_snps = all_snps[all_snps$pvalue<=alpha,]
  
  # selecting peaks from qtls
  peaks_marker =c()
  peaks_pos=c()
  peaks_chr=c()
  peaks_cumpos=c()
  peaks_pvalue=c()
  peaks_logP=c()
  peaks_ci_starts=c()
  peaks_ci_stops=c()	
  over = F
  while (over==F) {
    qtl_chr=as.numeric(all_snps[1,'chr'])
    qtl_pos=as.numeric(all_snps[1,'pos'])
    qtl_cumpos=as.numeric(all_snps[1,'cumpos'])
    qtl_marker=all_snps[1,'id']
    qtl_pvalue=all_snps[1,'pvalue']
    qtl_logP=all_snps[1,'logP']
    
    one_side_window1=as.numeric(windowsize)
    
    #print(is.numeric(qtl_pos))
    #print(windowsize)
    #print(is.numeric(windowsize))
    qtl_ci_start=qtl_pos-one_side_window1
    qtl_ci_stop=qtl_pos+one_side_window1
    
    #qtl_ci_start=max(1,qtl_pos-one_side_window1)
    #qtl_ci_stop=min(qtl_pos+one_side_window1,ends[qtl_chr])
    
    peaks_ci_starts=c(peaks_ci_starts,qtl_ci_start)
    peaks_ci_stops=c(peaks_ci_stops,qtl_ci_stop)		
    
    peaks_marker=c(peaks_marker,qtl_marker)
    peaks_pos=c(peaks_pos,qtl_pos)
    peaks_chr=c(peaks_chr,qtl_chr)
    peaks_cumpos=c(peaks_cumpos,qtl_cumpos)
    peaks_pvalue=c(peaks_pvalue,qtl_pvalue)
    peaks_logP=c(peaks_logP,qtl_logP)
    
    remove=which(all_snps[,'chr']==qtl_chr & all_snps[,'pos'] >=qtl_ci_start & all_snps[,'pos']<= qtl_ci_stop)
    #if (length(remove) == 0) stop()
    all_snps=all_snps[-remove,]
    if (dim(all_snps)[1]==0 || all_snps[1,'pvalue']>alpha) over=T
    all_snps=all_snps[order(all_snps[,'logP'],decreasing=T),]
  }
  
  # peaks' dataframe 
  all_peaks=data.frame(#measure=pheno_name,
                       marker=peaks_marker,
                       chr=peaks_chr,
                       pos=peaks_pos,
                       cumpos=peaks_cumpos,
                       pvalue = peaks_pvalue, 
                       logP=peaks_logP,
                       ci_starts=peaks_ci_starts,
                       ci_stops=peaks_ci_stops, stringsAsFactors=F)
  
  all_peaks[,"merge"] = 0 
  all_new=NULL
  
  for (chr in unique(all_peaks[,'chr'])) {
    soub=all_peaks[which(all_peaks[,'chr']==chr),]
    changed=T
    while(changed){
      changed=F
      if (dim(soub)[1]!=1) {
        for (k in (1:(dim(soub)[1]-1))) {
          marker1=soub[k,'marker']
          for (l in ((k+1):dim(soub)[1])) {
            marker2=soub[l,'marker']
            if (soub[k,'ci_stops']>=soub[l,'ci_starts'] & soub[k,'ci_starts']<=soub[l,'ci_stops']) {
              changed=T
              if (soub[k,'merge']==0 & soub[l,'merge']==0) {
                soub[k,'merge']=k
                soub[l,'merge']=k
              } else if (soub[k,'merge']!=0 | soub[l,'merge']!=0) {
                soub[k,'merge']=soub[k,'merge']
                soub[l,'merge']=soub[k,'merge']
              }
            }
          }
        }
        
        merge_values=unique(soub[,'merge'])
        if (all(merge_values!=0)) {
          now=NULL
          for (merge_value in merge_values) {
            soubsoub=soub[soub[,'merge']==merge_value,]
            w=which.max(soubsoub[,'logP'])
            add=data.frame(#measure=pheno_name,
                           marker='merged_peak',
                           chr=chr,
                           pos=soubsoub[w,'pos'],
                           cumpos=soubsoub[w,'cumpos'],
                           pvalue = soubsoub[w,'pvalue'],
                           logP=soubsoub[w,'logP'],
                           ci_starts=min(soubsoub[,'ci_starts']),
                           ci_stops=max(soubsoub[,'ci_stops']),
                           merge=0, stringsAsFactors=F)
            now=rbind(now,add)
          }
          soub=now
        } else {
          now=soub[soub[,'merge']==0,]
          for (merge_value in merge_values[-which(merge_values==0)]) {
            soubsoub=soub[soub[,'merge']==merge_value,]
            w=which.max(soubsoub[,'logP'])
            add=data.frame(#measure=pheno_name,
                           marker='merged_peak',
                           chr=chr,
                           pos=soubsoub[w,'pos'],
                           cumpos=soubsoub[w,'cumpos'],
                           pvalue = soubsoub[w,'pvalue'], 
                           logP=soubsoub[w,'logP'],
                           ci_starts=min(soubsoub[,'ci_starts']),
                           ci_stops=max(soubsoub[,'ci_stops']),
                           merge=0, stringsAsFactors=F)
            now=rbind(now,add)
          }
          soub=now
        }
      }
    }
    all_new=rbind(all_new,soub)
  }
  return(all_new)
}

#res = lapply(files[c(367)], get_snps)
# Careful: can lead to mem issue leading to errors - and null in file
# can check by `grep "P values present for all chromosomes"` in log file
res = mclapply(files, get_snps, mc.cores = 4) 
cat("phenos in res", length(res),"\n")

qtls = lapply(res, get_QTLs, alpha = alpha)

# Character and NULL elements
missing = do.call(rbind, lapply(names(qtls), function(pheno) {
                                              qtl_pheno = qtls[[pheno]]
                                              if (is.character(qtl_pheno) || is.null(qtl_pheno)) {
                                                miss = data.frame(measure = pheno, 
                                                                  message = if (is.null(qtl_pheno)) "null" else qtl_pheno)
                                                return(miss)
                                              } else {
                                                NULL
                                              }}
                                )
                  )
cat("phenos missing or no p-val <", alpha, ":", length(unique(missing$measure)),"\n")

all_qtls = do.call(rbind, lapply(names(qtls), function(pheno) {
                                              qtl_pheno = qtls[[pheno]]
                                              if (is.data.frame(qtl_pheno)) {
                                                qtl_pheno$measure = pheno
                                                qtl_pheno = qtl_pheno[, c("measure", setdiff(names(qtl_pheno), "measure"))] # order columns so that measure at the beginning
                                                return(qtl_pheno)
                                              } else {
                                                NULL
                                              }}
                                 )
                   )
cat("phenos with QTL for p-val <", alpha, ":", length(unique(all_qtls$measure)),"\n")

# Saving to file
#save(missing, all_qtls, res, file = outfile)
save(missing, res, file = outsnps)
write.table(all_qtls, outqtls, sep="\t", quote=F, col.names = T, row.names = F)
