toplot_file = "snps_QTLs_toplot.RData"
outfile = "porcupine_sgb_cecal_NY.png"
color_file = "porcupine_colors.RData"
  
# Loading res to plot
cat("loading data to plot\n")
obj = load(toplot_file); obj

### # Add full taxon name and study name to results
### cat("annotating results\n")
### # TODO: change the path here to the 'fun_annotate_VCs_pvalues.R'
### #source('/path/to/git/P50/16S/Figures/fun_annotate_VCs_pvalues.R') # annotate() function 
### source('annotate_VCs_pvalues.R') # annotate() function 
# Selecting significant res
sig_snps = all_snps[all_snps$logP > 5.8,] 
sig_snps = sig_snps[order(sig_snps$logP, decreasing = T),]
# Defining which traits are significant 
uniqs = unique(sig_snps$trait1)
# Selecting significant traits
motch = match(uniqs, sig_snps$trait1)
sig_snps = sig_snps[motch,] # dim(sig_snps) 111 8
### # Annotating significant traits with full taxon name
### sig_snps = annotate(sig_snps)
### # Adding ASV n to taxon name
### asvs = grep('ASV', sig_snps$trait1)
### sig_snps[asvs,'full_taxon'] = paste(sig_snps[asvs,'full_taxon'], sig_snps[asvs,'taxon1'], sep=';')
### motch = match(all_snps$trait1, sig_snps$trait1)
### all_snps$full_taxon = sig_snps[motch,'full_taxon']
### all_snps$study1 = sig_snps[motch,'study1']
### #length(unique(all_snps$full_taxon)); # this is 97
### #length(unique(sig_snps$full_taxon)); # this is 97 too
### #length(uniqs) # this is 111
all_snps = all_snps[order(all_snps$logP, decreasing = T),]


# saving objects for plotting
#save(all_snps, all_ld, file = "source_files/fig4.RData")

# loading objects from source for plotting
#load(file = "source_files/fig4.RData")

# Function to plot manhattan
draw_manhattan_plot=function(dat, def.cex = 0.4, sig.cex = 2, cex.x=1.25, cex.y=1.25, cex.lab=1.4) {
  #png(outfile, height = 8, width = 23.5, units = "in", res = 300); par(mar=c(5.1,5.1,2.1,0.5))
  cols = c("grey75", "grey50") 
  # Default color for chromosomes (odd numbers)
  #dat = all_snps
  dat$colour=cols[1]
  # Chromosome with even number are of the other color
  dat[which(dat[,'chr'] %in% c(2,4,6,8,10,12,14,16,18,20)),'colour']=cols[2]
  # Default dimension and type of the dot, to change with dimensions of output image
  dat$cex = def.cex
  dat$pch = 16
  
  # Selecting rows that have a different color than the default - significant dots
  w = which(dat[,'col'] != 'darkgrey')
  sig.pch = c("NY" = 15, "MI" = 16, "TN_behavior" = 17, "TN_breeder" = 18, "all" = 16)
  
  # Defining pch and colour of significant dots depending on cohorts and taxa
  if (length(w)!=0) {
    dat[w,'cex'] = sig.cex
    dat[w,'colour'] = dat[w,'col']
    dat[w,'pch'] = sig.pch[dat[w,"study1"]]
    dat[dat[,'pch'] == 18, "cex"] = sig.cex + 0.3
  }
  
  # Need both order first by decreasing logP then by colour otherwise QTL point gets overpainted so is not visible
  dat = dat[order(dat$colour != cols[1]),]
  xmar = 10*min(dat$cumpos)
  # Plotting manhattan
  plot(dat[,c('cumpos','logP')], ylab="", col=dat[,'colour'], 
       xlab="", main='', cex=dat[,'cex'],
       pch=dat[,"pch"], axes=FALSE, 
       xaxs="i", xlim=c(min(dat$cumpos) - xmar, max(dat$cumpos) + xmar),
       ylim=c(min(dat$logP), 25) 
  )
  # adding box
  box(lwd=1)
  # adding threshold lines
  abline(h=5.8, lty=2, lwd = 1.5, col="black") # genome wide sig
  abline(h=8.4, lty=3, lwd = 1.5, col="black") # bonferroni sig
  # y-axis
  axis(2, cex.axis=cex.y, las = 1, line = 0) 
  # x-axis
  axis(1, labels = F,
       at= tapply(dat$cumpos, dat$chr, function(x) median(range(x))), 
       las = 1, tck=0.01)
  axis(1,labels=c(1:20), 
       at= tapply(dat$cumpos, dat$chr, function(x) median(range(x))), 
       cex.axis=cex.x, las = 1, tck=F, lwd = 0, line=-0.5)
  title(ylab = "-logP", cex.lab=cex.lab, 
        xlab = "Chromosome")
  #dev.off()
}


draw_legend_all = function(dat, x.space=25, y.space=23, cex = 0.8, maxn = 7, prop.width = 1){
  # otherwise QTL point gets overpainted so is not visible
  dat = dat[dat$col != 'darkgrey',] # needed otherwise match below goes to first occurence, which is darkgrey
  w = which(dat$logP > 5.8) # & paste(dat$chr, dat$pos, sep=':') %in% c(all_ld$SNP_A,all_ld$SNP_B))
  # Selecting significant dots
  sigs = dat[w,c('chr','full_taxon','col',"trait1")]
  
  ids = paste(dat[w,'chr'], dat[w,'pos'],sep=':')
  ## tops = c()
  ## for (id in ids) {
  ##   w = which(all_ld$SNP_A == id | all_ld$SNP_B == id)
  ##   top = unique(all_ld[w,'top'])
  ##   if (length(top) != 1) stop('pb') 
  ##   tops = c(tops, top)
  ## }	
  ## sigs = sigs[order(tops),]
  
  sigs = sigs[!duplicated(sigs),] #unique because of same taxon across cohorts
  
  # Ad hoc dictionary for the names - depends on the significant taxa
  ## dict = c("Firmicutes_A phylum" = "p__Firmicutes",
  ##          "CAG-485 genus" = "g__CAG-485",
  ##          "Paraprevotella genus" = "g__Paraprevotella")
  ## sigs[,"legend name"] = NA
  ## for(d in seq_along(dict)){
  ##   x = grep(dict[d], sigs[,"full_taxon"])
  ##   sigs[x,"legend name"] = names(dict)[d]
  ## }
  #sigs[,"legend name"] = ave(seq_len(nrow(sigs)), sigs$full_taxon, FUN = function(i) 
  #                          paste(sigs$full_taxon[i][1], paste(unique(sigs$trait1[i]), collapse = ","), sep = ":")
  #)
  sigs[,"legend name"] = paste0(sigs[,"full_taxon"],";",sigs[,"trait1"])
  #sigs = sigs[!duplicated(sigs[,c("col", "legend name")]),]
  
  # Defining plotting param - so that similar in all
  ycord = grconvertY(1, "npc") # at the top
  y.step = (abs(grconvertY(0, "npc")) + abs(grconvertY(1, "npc"))) / y.space #23 # step of 0.01 
  
  xcord = grconvertX(0, "npc") # change here to put more to the right/left, increase = to right
  x.step = (abs(grconvertX(0, "npc")) + abs(grconvertX(1, "npc"))) / x.space #25 # step of 0.01 
  print(x.step)
  #cex=1.4
  x.inter = 0.7
  
  # Legend for shapes 
  ## From Manhattan above
  #if (all(dat$study1 == 'all')) {
  #  sig.pch = c('all' = 16)
  #  dict_coh = c(all = 'all') #dict_coh[names(sig.pch)]
  #  dict_pch = c("16"="21") #dict_pch[as.character(sig.pch)]
  #} else {
  #  sig.pch = c("NY" = 15, "MI" = 16, "TN_behavior" = 17, "TN_breeder" = 18)
  #  dict_coh = c(NY="NY", MI="MI", TN_behavior = "TN1", TN_breeder= "TN2") #dict_coh[names(sig.pch)]
  #  dict_pch = c("15"="22", "16"="21", "17"="24", "18"="23") #dict_pch[as.character(sig.pch)]
  #}
  # legend for pch - cohorts
  #legend(x=xcord,
  #       y=ycord,
  #       legend=dict_coh[names(sig.pch)],
  #       pch=as.numeric(dict_pch[as.character(sig.pch)]),
  #       col="black",
  #       bty="n",
  #       cex = cex.lab, 
  #       border=NULL
  #)
  #xcord = xcord + x.step*3.4 # change *N to distance more the first and second legend
  
  # legend for colours - taxa
  for (chr in unique(sigs[,"chr"])[order((unique(sigs[,"chr"])))]){
    lgdf =  sigs[sigs$chr == chr,]
    if(nrow(lgdf) > maxn){
      steps = ceiling(nrow(lgdf) / maxn)
      step = 1; r = 1
      xcord.p = xcord
      while(step <= steps){
        lgdf_split = lgdf[r:min((maxn*step), nrow(lgdf)),]
        for(x in 1:nrow(lgdf_split)){
          cat(lgdf_split[x, "legend name"]," ", lgdf_split[x,"col"], "\n")
          legend(x=xcord.p, 
                 y=ycord,
                 legend = lgdf_split[x, "legend name"], 
                 x.intersp = x.inter, 
                 lwd = 9, lty= 2, seg.len=0.5,
                 col = lgdf_split[x,"col"], 
                 border = F,
                 cex = cex, 
                 horiz=T, 
                 bty="n", xpd = T)
          this.step = strwidth(lgdf_split[x, "legend name"]) / prop.width  + x.step
          print(this.step)
          if(x ==1 & step == 1){x.step1 = this.step}
          xcord.p = xcord.p + this.step 
          print(xcord.p)
        }
        if(step < steps) { ycord = (abs(ycord))-y.step/1.5 }
        else{ ycord = (abs(ycord))-y.step}
        r = r+maxn
        step = step+1
        xcord.p = xcord + x.step1/4
      }
    }else{
    #dupl = duplicated(lgdf[, "chr"], fromLast = T)
    # legend for taxa that are duplicated, many colours close to one name
    #if(any(dupl)){
    #if(TRUE){
      #lgdf[dupl,"legend name"] = "" # removing names to duplicated lines
      xcord.p = xcord
      for(x in 1:nrow(lgdf)){
        cat(lgdf[x, "legend name"]," ", lgdf[x,"col"], "\n")
        legend(x=xcord.p, 
               y=ycord,
               legend = lgdf[x, "legend name"], 
               x.intersp = x.inter, 
               lwd = 9, lty= 2, seg.len=0.5,
               col = lgdf[x,"col"], 
               border = F,
               cex = cex, 
               horiz=T, 
               bty="n", xpd = T)
        this.step = strwidth(lgdf[x, "legend name"]) / prop.width  + x.step
        print(this.step)
        xcord.p = xcord.p + this.step 
      }
      ycord = (abs(ycord))-y.step
    }
  }
}

###### Setting colors
cat("Setting colours\n")
colres = unique(all_snps$col)

# as is, _v4 - order of colours is important to have the most different colours in the highest peaks
## coolors = c("#FF6E00FF", "#1A476FFF", "#8F7EE5FF", "#980043FF", "#59A14FFF", "#FABFD2FF",
##             "#8CD17DFF", "#A0CBE8FF", "#51A3CCFF", "#FFAA0EFF", "#835B82FF", "#DF65B0FF",
##             "#B07AA1FF", "#B26F2CFF", "#CC5500FF", "#FFD200FF", "#85B22CFF", "#F28E2BFF", 
##             "#D7B5A6FF", "#FFD8B2FF", "#993D00FF", "#FFBE7DFF", "#FFE474FF", "#B6992DFF",
##             "#E5FFB2FF", "#9D7660FF", "#BFB2FFFF", "#B2E5FFFF", "#8491B4FF", "#B2AD8FFF",
##             "#6E8E84FF", "#91D1C2FF", "#DC0000FF", "#0F6B99FF", "#CE1256FF", "#260F99FF", 
##             "#D4A6C8FF", "#8A60B0FF", "#C994C7FF", "#499894FF", "#6551CCFF", "#E5B17EFF",
##             "#800080FF", "#C3E57EFF", "#E7298AFF", "#D37295FF", "#7E6148FF", "#E57E7EFF",   
##             "#662700FF", "#FFB2B2FF", "#CC5151FF")  
coolors = c("#FF6E00FF", "#8F7EE5FF", "#980043FF", "#1A476FFF", "#59A14FFF", "#FABFD2FF",
            "#8CD17DFF", "#A0CBE8FF", "#51A3CCFF", "#FFAA0EFF", "#835B82FF", "#DF65B0FF",
            "#B07AA1FF", "#B26F2CFF", "#CC5500FF", "#FFD200FF", "#85B22CFF", "#F28E2BFF", 
            "#D7B5A6FF", "#FFD8B2FF", "#993D00FF", "#FFBE7DFF", "#FFE474FF", "#B6992DFF",
            "#E5FFB2FF", "#9D7660FF", "#BFB2FFFF", "#B2E5FFFF", "#8491B4FF", "#B2AD8FFF",
            "#6E8E84FF", "#91D1C2FF", "#DC0000FF", "#0F6B99FF", "#CE1256FF", "#260F99FF", 
            "#D4A6C8FF", "#8A60B0FF", "#C994C7FF", "#499894FF", "#6551CCFF", "#E5B17EFF",
            "#800080FF", "#C3E57EFF", "#E7298AFF", "#D37295FF", "#7E6148FF", "#E57E7EFF",   
            "#662700FF", "#FFB2B2FF", "#CC5151FF")  
names(coolors) = colres[-which(colres == "darkgrey")]
coolors = c(coolors, "darkgrey" = "darkgrey")

#to plot for the whole sample AFTER plotting for the individual cohorts, comment out the following lines
all_snps$col = unname(coolors[all_snps$col])
tosave = all_snps[,c("trait1","full_taxon","col")]
save(tosave, file = color_file)

##to plot for the whole sample AFTER plotting for the individual cohorts, uncomment the following lines
#load("porcupine_colors.RData")
#w = which(all_snps$col != 'darkgrey')
#motch = match(all_snps[w,'full_taxon'], tosave$full_taxon)
#all_snps[w,'col'] = tosave[motch,'col']
#all_snps[is.na(all_snps$col),'col'] = 'darkgrey'


#### Subset for tests
## dot = do.call(rbind, lapply(as.numeric(unique(all_snps$chr)), function(x) {
##   set.seed(34)
##   non_sig = all_snps[sample(which(all_snps[,"chr"] == x & all_snps[,"col"] == "darkgrey"), size = 90), ]
##   sr = which(all_snps[,"chr"] == x & all_snps[,"col"] != "darkgrey")
##   if(length(sr) != 0){
##     if(length(sr) < 10) sig = all_snps[sample(sr, size = length(sr)), ]
##     else sig = all_snps[sample(sr, size = 10), ]
##   } else{ sig = matrix(nrow = 0, ncol = ncol(non_sig))}
##   return(rbind(sig, non_sig))
## }
## ))
##  # Plotting porcupine
##  pdf("porcupine_uncollapsed_genus2_test.pdf", h=7, w=23.5)
##  par(mar=c(5.1,5.1,2.1,0.5))
##  draw_manhattan_plot(dot, def.cex=0.6, cex.lab=2, cex.x = 1.4, cex.y = 1.4)
##  draw_legend(dot, x.space = 70) 
##  dev.off()



# Open pdf to save plot
#pdf("porcupine_uncollapsed_genus2.pdf", h=8, w=23.5)
#par(mar=c(5.1,5.1,2.1,0.5))
png(outfile, height = 18, width = 26, units = "in", res = 300); 
par(mar=c(5.1,5.1,2.1,0.5))

cat("Plotting\n")
layout(matrix(c(1,2), ncol = 1, nrow = 2))
draw_manhattan_plot(all_snps, def.cex=0.6, cex.lab=2, cex.x = 1.3, cex.y = 1.3)
plot.new()
par(mar=c(0.1,0.1,0.1,0.1), xpd = T)
draw_legend_all(all_snps, x.space = 22, y.space = 20, cex = 1, prop.width = 1.3, maxn = 7)
dev.off()


## # Annotating snps in ld
## pvalues_dir=''
## target_loci = c('1:196217481','4:70834123','10:101974959') # GWAS per cohort # comment to plot _ALL
## #target_loci = c('1:196498032','4:70473002','10:101972884') # updated based on GWAS with whole sample # uncomment to plot _ALL
## all_ld = NULL
## for (target_locus in target_loci) {
##   splot = strsplit(target_locus,':')[[1]]
##   target_chr = splot[1]
##   target_pos = splot[2]
##   load(paste0(pvalues_dir,'LD_',target_chr, '_', target_pos,'.RData'))
##   ld$top = target_locus
##   all_ld = rbind(all_ld, ld)
## }
## all_ld = all_ld[all_ld$R2 >= 0.80,]
## # Function to plot legend # this function needs all_ld 
## draw_legend = function(dat, all_ld, x.space=25, y.space=23){
##   # otherwise QTL point gets overpainted so is not visible
##   dat = dat[dat$col != 'darkgrey',] # needed otherwise match below goes to first occurence, which is darkgrey
##   w = which(dat$logP > 5.8 & paste(dat$chr, dat$pos, sep=':') %in% c(all_ld$SNP_A,all_ld$SNP_B))
##   # Selecting significant dots
##   sigs = dat[w,c('chr','full_taxon','col')]
##   
##   ids = paste(dat[w,'chr'], dat[w,'pos'],sep=':')
##   tops = c()
##   for (id in ids) {
##     w = which(all_ld$SNP_A == id | all_ld$SNP_B == id)
##     top = unique(all_ld[w,'top'])
##     if (length(top) != 1) stop('pb') 
##     tops = c(tops, top)
##   }	
##   sigs = sigs[order(tops),]
##   
##   sigs = sigs[!duplicated(sigs),] #unique because of same taxon across cohorts
##   
##   # Ad hoc dictionary for the names - depends on the significant taxa
##   dict = c("Firmicutes_A phylum" = "p__Firmicutes",
##            "CAG-485 genus" = "g__CAG-485",
##            "Paraprevotella genus" = "g__Paraprevotella")
##   sigs[,"legend name"] = NA
##   for(d in seq_along(dict)){
##     x = grep(dict[d], sigs[,"full_taxon"])
##     sigs[x,"legend name"] = names(dict)[d]
##   }
##   sigs = sigs[!duplicated(sigs[,c("col", "legend name")]),]
##   
##   # Defining plotting param - so that similar in all
##   ycord = grconvertY(1, "npc") # at the top
##   y.step = (abs(grconvertY(0, "npc")) + abs(grconvertY(1, "npc"))) / y.space #23 # step of 0.01 
##   
##   xcord = grconvertX(0, "npc") # change here to put more to the right/left, increase = to right
##   x.step = (abs(grconvertX(0, "npc")) + abs(grconvertX(1, "npc"))) / x.space #25 # step of 0.01 
##   cex.lab=1.4
##   x.inter = 0.7
##   
##   # Legend for shapes 
##   # From Manhattan above
##   if (all(dat$study1 == 'all')) {
##     sig.pch = c('all' = 16)
##     dict_coh = c(all = 'all') #dict_coh[names(sig.pch)]
##     dict_pch = c("16"="21") #dict_pch[as.character(sig.pch)]
##   } else {
##     sig.pch = c("NY" = 15, "MI" = 16, "TN_behavior" = 17, "TN_breeder" = 18)
##     dict_coh = c(NY="NY", MI="MI", TN_behavior = "TN1", TN_breeder= "TN2") #dict_coh[names(sig.pch)]
##     dict_pch = c("15"="22", "16"="21", "17"="24", "18"="23") #dict_pch[as.character(sig.pch)]
##   }
##   # legend for pch - cohorts
##   legend(x=xcord,
##          y=ycord,
##          legend=dict_coh[names(sig.pch)],
##          pch=as.numeric(dict_pch[as.character(sig.pch)]),
##          col="black",
##          bty="n",
##          cex = cex.lab, 
##          border=NULL
##   )
##   xcord = xcord + x.step*3.4 # change *N to distance more the first and second legend
##   
##   # legend for colours - taxa
##   for (chr in unique(sigs[,"chr"])[order((unique(sigs[,"chr"])))]){
##     lgdf =  sigs[sigs$chr == chr,]
##     
##     dupl = duplicated(lgdf[, "legend name"], fromLast = T)
##     # legend for taxa that are duplicated, many colours close to one name
##     if(any(dupl)){
##       lgdf[dupl,"legend name"] = "" # removing names to duplicated lines
##       xcord.p = xcord
##       for(x in 1:nrow(lgdf)){
##         cat(lgdf[x, "legend name"]," ", lgdf[x,"col"], "\n")
##         legend(x=xcord.p, 
##                y=ycord,
##                legend = lgdf[x, "legend name"], 
##                x.intersp = x.inter, 
##                lwd = 9, lty= 2, seg.len=0.5,
##                col = lgdf[x,"col"], 
##                border = F,
##                cex = cex.lab, 
##                horiz=T, 
##                bty="n", xpd = T)
##         xcord.p = xcord.p + x.step 
##       }
##     }else{
##       # legend for taxa non duplicated, one colour one name
##       legend(x=xcord,
##              y=ycord,
##              legend = lgdf[, "legend name"], 
##              x.intersp = x.inter, 
##              lwd = 9, lty= 2, seg.len=0.5,
##              col = lgdf[,"col"], 
##              border = F,
##              cex = cex.lab, 
##              horiz=T, 
##              bty="n", xpd = T)
##     }
##     ycord = (abs(ycord))-y.step
##   }
## }
## 