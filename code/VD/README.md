# How to write .h5 input to run covRunner.py

Will have different H5I_GROUPS (which names need to match the following ones): <br/>

### 1. **`phenotypes/`**

1.1. subgroup: *`<pheno_subgroup>/`* (e.g. "data_bcNcovariates") <br/>
  
  - \<pheno_subgroup\> corresponds to 'phenos_version'<br/> (refers to the type of phenotypes to analyse, e.g. phenotypes that went through different type of normalization)
  - passed with `--phenos_v` option (**required**); in code, is stored in `self.phenos_version` <br/>

1.2. datasets: <br/>

  - `['matrix']`;
  - `['col_header']['phenotype_ID']`;
  - `['col_header']['covariatesUsed']`; per each phenotype, the comma-sep string of covariates used - names corresponding to `covariate_ID` in `covariates/` (e.g. "sex,age,Batch", "sex,Batch"...)
  - `['row_header']['sample_ID']`
<br/> 

### 2. **`covariates/`**
2.1. subgroup: *`<cov_subgroup>/`* (e.g. "data_bcNcovariates") <br/>
  
  - \<cov_subgroup\> corresponds to 'covs_version' <br/> (similar to pheno_subgroup)
  - passed with `--covs_v` option (**required**, set 'None' if not used); in code, is stored in `self.covs_version` <br/>

2.2. datasets: <br/>

  - `['matrix']`;
  - `['col_header']['covariate_ID']`; e.g. "sex", "age", "Batch"
  - `['row_header']['sample_ID']`
<br/> 

### 3. **`cages/`**
3.1. subgroup: *`<cage_subgroup>/`* (e.g. "all") <br/>
  
  - \<cage_subgroup\> corresponds to 'cage_version' <br/> (for different batches or analysis)
  - passed with `--cage_v` option (**required**); in code, is stored in `self.cage_version` <br/>

3.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`
<br/> 

### 4. **`dam/`**
4.1. subgroup: *`<dam_subgroup>/`* (e.g. "mother_name") <br/>
  
  - \<dam_subgroup\> corresponds to 'dam_version' <br/> (for different batches or analysis)
  - passed with `--dam_v` option (**required**, set 'None' if not used); in code, is stored in `self.dam_version` <br/>

4.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`
<br/> 

### 5. **`GRM/`** or **`GRM_LOCO/`**
5.1. subgroup: *`<GRM_subgroup>/`* (e.g. "Andres_kinship") <br/>
  
  - \<GRM_subgroup\> corresponds to 'GRM_version' <br/>
  - passed with `--grm_v` option (**required**); in code, is stored in `self.GRM_version` <br/>

5.2. datasets: <br/>

  - `['matrix']`; 
  - `['row_header']['sample_ID']`
<br/> 

### 6. **`sex_cov/`**
6.1. subgroup: *`<sex_subgroup>/`* (e.g. "all") <br/>
  
  - \<sex_subgroup\> corresponds to 'sex_version'
  - passed with `--sex_v` option, optional (default is 'None'); in code, is stored in `self.sex_version` <br/>

6.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`
<br/> 

### 7. **`subsets/`** <br/>
7.1. *`<subset_name>`* (e.g. "include") <br/>

  - \<subset_name\> is the name of a specific subset, to handle optional subset of individuals, e.g. excluding singletons
  - passed with `-s` option, optional (default is 'None')
  
7.2 datasets: <br/>  

  - `['<subset_name>']`: contains all sample_id that are part of the subset

<br/><br/>

## Examples

### A. input.h5 
In R can look at it using, `h5ls("input.h5")`. NB: in this example there is no group "subsets"
```
                                       group               name       otype
0                                          /                GRM   H5I_GROUP
1                                       /GRM     Andres_kinship   H5I_GROUP
2                        /GRM/Andres_kinship             matrix H5I_DATASET
3                        /GRM/Andres_kinship         row_header   H5I_GROUP
4             /GRM/Andres_kinship/row_header          sample_ID H5I_DATASET
5                                          /              cages   H5I_GROUP
6                                     /cages                all   H5I_GROUP
7                                 /cages/all              array H5I_DATASET
8                                 /cages/all          sample_ID H5I_DATASET
9                                          /         covariates   H5I_GROUP
10                               /covariates data_bcNcovariates   H5I_GROUP
11            /covariates/data_bcNcovariates         col_header   H5I_GROUP
12 /covariates/data_bcNcovariates/col_header       covariate_ID H5I_DATASET
13            /covariates/data_bcNcovariates             matrix H5I_DATASET
14            /covariates/data_bcNcovariates         row_header   H5I_GROUP
15 /covariates/data_bcNcovariates/row_header          sample_ID H5I_DATASET
16                                         /                dam   H5I_GROUP
17                                      /dam        mother_name   H5I_GROUP
18                          /dam/mother_name              array H5I_DATASET
19                          /dam/mother_name          sample_ID H5I_DATASET
20                                         /         phenotypes   H5I_GROUP
21                               /phenotypes data_bcNcovariates   H5I_GROUP
22            /phenotypes/data_bcNcovariates         col_header   H5I_GROUP
23 /phenotypes/data_bcNcovariates/col_header     covariatesUsed H5I_DATASET
24 /phenotypes/data_bcNcovariates/col_header    macrophenotypes H5I_DATASET
25 /phenotypes/data_bcNcovariates/col_header       phenotype_ID H5I_DATASET
26            /phenotypes/data_bcNcovariates             matrix H5I_DATASET
27            /phenotypes/data_bcNcovariates         row_header   H5I_GROUP
28 /phenotypes/data_bcNcovariates/row_header          sample_ID H5I_DATASET
29                                         /            sex_cov   H5I_GROUP
30                                  /sex_cov                all   H5I_GROUP
31                              /sex_cov/all              array H5I_DATASET
32                              /sex_cov/all          sample_ID H5I_DATASET
33                                         /            subsets   H5I_GROUP
34                                  /subsets            females H5I_DATASET
35                                  /subsets            include H5I_DATASET
36                                  /subsets              males H5I_DATASET
```


### B. Example of running the code

Modelling univariate DGE+IEE+cageEffect (i.e. no IGE but include IEE)
```
python covRunner.py \
      --input HSmice.h5 \
      --phenos_v data_bcNcovariates \
      --covs_v data_bcNcovariates \
      --cage_v all \
      --grm_v Andres_kinship \
      --analysis_type VD \
      --out ./ \
      -p 1 \
      -m uni \
      -e DGE,IEE,cageEffect
```
<br/>     

Modelling bivariate DGE+IGE+cageEffect+maternalEffect (only for samples in "include" subset)
```
python covRunner.py \
      --input HSmice.h5 \
      --phenos_v data_bcNcovariates \
      --covs_v data_bcNcovariates \
      --cage_v all \
      --dam_v mother_name \
      --grm_v Andres_kinship \
      --analysis_type VD \
      --out ./ \
      -p 1 \
      -c combins.csv \
      -s include \
      -m bi \
      -e DGE,IGE,cageEffect,maternalEffect \
```
<br/>       

Modelling sexvariate DGE+IGE+cageEffect, with corr_As1s2 constrained to 1
```
python covRunner.py \
      --input HSmice.h5 \
      --phenos_v data_bcNcovariates \
      --covs_v data_bcNcovariates \
      --cage_v all \
      --grm_v Andres_kinship \
      --sex_v all \
      --analysis_type VD \
      --out ./ \
      -p 1 \
      -m sex \
      -e DGE+IGE+cageEffect  \
      -z corr_As1s2,1
```
