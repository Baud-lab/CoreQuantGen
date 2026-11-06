# code

Analysis scripts. Most are called by the Nextflow pipelines: each pipeline's `bin/` folder contains symlinks to these files
(e.g. `nf-realdata/bin/covRunner.py -> ../../code/VD/covRunner.py`), so the same script is shared by several pipelines.

| Folder | What it does | Status |
|---|---|---|
| [`VD/`](VD) | Python code for VD and GWAS mapping, built on [limix-core](https://pypi.org/project/limix-core/) and [limix-lmm](https://pypi.org/project/limix-lmm/); see [its README](VD/README.md) | used by the pipelines |
| [`preVD/`](preVD) | prepares input for VD (simulation parameters, simulated phenotypes, phenotype pairs; `subset_h5.R` makes a smaller `.h5` from an existing one) | used by `nf-simulations` |
| [`afterVD/`](afterVD) | parses VD output (tables to `.Rdata`, grouping of `.pkl` files) | used by `nf-realdata`, `nf-simulations` |
| [`afterGWAS/`](afterGWAS) | parses GWAS output (QTL calling, plots) | under development |
