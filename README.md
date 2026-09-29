
# Core Quantitative Genetic
code for single- and multi-trait genetic analysis (variance decomposition and GWAS) with DGE and IGE<br>

## in code/<br>
1. `afterGWAS/` scripts to parse output of VD - and save results as R object<br>
2. `afterVD/` scripts to parse output of VD - and save results as R object<br>
3. `preVD/` scripts to prepare simulations to analyse with VD <br>
4. `VD/` scripts to run variance decomposition analysis or null covariance matrix for LOCO GWAS<br>

## in nf-realdata/<br>
Nextflow pipeline to run variance decomposition analysis

## in nf-simulations/<br>
Nextflow pipeline to run simulations and then variance decomposition analysis<br>

## in nf-gwas/<br>
Nextflow pipeline to run gwas - under development <br>

## in nf-genotype-prep/<br>
Nextflow pipeline to prepare genotypes for gwas analysis - under development <br>

**NB:** Figures related to analysis of CFW and HS mice published in Tonnelé et al. 2026 
can be found [here](https://github.com/Baud-lab/CFW_HS_mice/tree/Master/bivIGE_paper)
