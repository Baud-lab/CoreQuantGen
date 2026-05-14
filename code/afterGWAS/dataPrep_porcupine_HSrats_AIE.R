library(rhdf5)
library(tibble) # column_to_rownames()
library(dplyr) # mutate()
library(tidyr) # separate()

# All
tax_file = "..."
dict_file = "..."

# Cohort
study = "..."
qtl_file = "..."
outplot = "snps_QTLs_toplot.RData"
outqtls = "top_QTLs.tsv"

# Loading 'unpruned_bug_QTLs'
# for porcupine plot for individual cohorts (Fig 4):
objs = load(qtl_file); objs # loading all_qtls, missing, res (which is snps)
top_qtls = all_qtls[all_qtls$logP > 5.8, ]
pheno_names = unique(top_qtls$measure)

# Read taxonomic dictionary 
dict = read.table(dict_file,check.names = F, sep="\t"); colnames(dict) = c("SGB_number", "clade_name")
head(dict)

tax_dict = dict |>
  separate(
    col  = clade_name,
    into = c("Kingdom","Phylum","Class","Order","Family","Genus","Species"),
    sep  = "\\;"
  ) |>
  mutate(across(Kingdom:Species, ~ sub(".*__", "", .x))) |>
  column_to_rownames("SGB_number")
head(tax_dict)
tax_sig = tax_dict[pheno_names,]
dim(tax_sig)
tax_sig[tax_sig[,"Genus"] == "","Genus"] = paste0("f__",tax_sig[tax_sig[,"Genus"] == "","Family"])

# Setting mock colours to set them later - col1:coln
n = length(unique(tax_sig[pheno_names,"Genus"]))
colours = paste0("col", 1:n)
names(colours) = unique(tax_sig[pheno_names,"Genus"])

# Setting colours to plot
# Removing phenos for which error was detected
err = missing[grepl("Error", missing$message), "measure"]
res = res[!names(res) %in% err]

measures = names(res)
names(measures) = measures
#m = measures[4]
all_snps = lapply(measures, function(m) {snps = res[[m]]
                                        snps$col = 'darkgrey'
                                        snps$trait1 = m
                                        snps$full_taxon = paste(tax_dict[m,c("Phylum","Genus")], collapse = ";")
                                        snps$study1 = study
                                        if(m %in% rownames(tax_sig)){
                                          taxon = tax_sig[m,"Genus"]
                                          w = which(top_qtls[,'measure'] == m)
                                          for (i in w) {
                                            quels = which(snps[,'chr'] == top_qtls[i,'chr'] & snps[,'pos'] == top_qtls[i,'pos'])
                                            #		if (length(quels) != 1) print(k)
                                            snps[quels,'col'] = colours[taxon] 
                                          }
                                        }
                                        return(snps)
                                        })

# merging all together
#sum(sapply(all_snps, nrow))
# [1] 2839113 - cecal sgb NY
apply(sapply(all_snps, colnames), 1, unique)

all_snps = do.call(rbind, all_snps)
unique(all_snps$col)

cat("saving file to", outplot, "\n")
save(all_snps, file = outplot)


# Doing the same with top qtls to have them in a file, adding full_taxon
top_qtls$full_taxon = apply(tax_sig[top_qtls$measure,c("Phylum","Class","Order","Family","Genus")], 1, function(x) paste(x, collapse = ";"))
front_col = c("full_taxon", "measure")
top_qtls = top_qtls[c(front_col, setdiff(colnames(top_qtls), front_col))]
top_qtls = top_qtls[order(top_qtls$logP, decreasing = T),]
top_qtls

cat("writing topqtls to", outqtls, "\n")
write.table(top_qtls, file = outqtls, sep="\t", col.names = T, row.names = F, quote = F)
