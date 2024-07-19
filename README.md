
# Core Quantitative Genetic
code for single- and multi-trait genetic analysis (variance decomposition and GWAS) with DGE and IGE<br>

## in `pysrc`<br>
`exp_bivar12_wMat.py` to run variance decomposition analysis or null covariance matrix for LOCO GWAS<br>
needs `classes/`
	
  + `SigmaRhoCov.py`
  + `dirIndirCov_v2.py`
  + `dirIndirVD_noMT_wMat.py`
  + `social_data_wMat.py`

`map_LOCO_noMT.py` to run LOCO GWAS (in development)<br>

## in `Rsrc`<br>
`estimate_to_rdata.R` to parse etimates and STE files after variance decomposition with exp_bivar<br>
needs `functions/prepare_res.R`

