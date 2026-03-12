
# Core Quantitative Genetic
code for single- and multi-trait genetic analysis (variance decomposition and GWAS) with DGE and IGE<br>

## in code/<br>
1. `VD/` scripts to run variance decomposition analysis or null covariance matrix for LOCO GWAS<br>
2. `afterVD/` scripts to parse output of VD - and save results as R object<br>
3. `preVD/` scripts to prepare simulations to analyse with VD <br>

## in nf-realdata/<br>
Nextflow pipeline to run variance decomposition analysis

## in nf-simulations/<br>
Nextflow pipeline to run simlulations and then variance decomposition analysis<br>

**NB:** Figures related to analysis of CFW and HS mice published in [Tonnelé et al. 2026](https://www.biorxiv.org/content/10.64898/2026.03.10.710784v1) can be found [here](https://github.com/Baud-lab/CFW_HS_mice)
