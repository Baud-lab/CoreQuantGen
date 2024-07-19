
# Core Quantitative Genetic
code for single- and multi-trait genetic analysis (variance decomposition and GWAS) with DGE and IGE<br>

## in pysrc/<br>
1. `exp_bivar12_wMat.py` to run variance decomposition analysis or null covariance matrix for LOCO GWAS<br>
needs `classes/`
	
  + SigmaRhoCov.py
  + dirIndirCov\_v2.py
  + dirIndirVD\_noMT\_wMat.py
  + social\_data\_wMat.py

2. `map_LOCO_noMT.py` to run LOCO GWAS (in development)<br>

## in Rsrc/<br>
1. `estimate_to_rdata.R` to parse etimates and STE files after variance decomposition with exp_bivar<br>
needs `functions/prepare_res.R`

