## HOW TO WRITE the .h5

Will have different H5I_GROUPS (which names need to match the following ones): <br/>

##### 1. **`GRM/`** or **`GRM_LOCO/`**
1.1. subgroup: *`<GRM_subgroup>/`* (e.g. "Andres_kinship") <br/>
  
  - \<GRM_subgroup\> corresponds to 'GRM_version' and is passed **as argument** <br/>
  - in code, is stored in `self.GRM_version` <br/>

1.2. datasets: <br/>

  - `['matrix']`; 
  - `['row_header']['sample_ID']`

<br/>

##### 2. **`cages/`**
2.1. subgroup: *`<cage_subgroup>/`* (e.g. "real") <br/>
  
  - \<cage_subgroup\> corresponds to 'cage_version' and is passed **as argument** <br/> (for different batches or analysis)
  - in code, is stored in `self.cage_version` <br/>

2.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`

<br/>

##### 3. **`covariates/`**
3.1. subgroup: *`<cov_subgroup>/`* (e.g. "from_file") <br/>
  
  - \<cov_subgroup\> corresponds to 'covs_version' and is passed **as argument** <br/> (similar to pheno_subgroup)
  - in code, is stored in `self.covs_version` <br/>

3.2. datasets: <br/>

  - `['matrix']`;
  - `['col_header']['covariate_ID']`; e.g. "sex", "age", "Batch"
  - `['row_header']['sample_ID']`

<br/>

##### 4. **`dam/`**
4.1. subgroup: *`<dam_subgroup>/`* (e.g. "simulated") <br/>
  
  - \<dam_subgroup\> corresponds to 'dam_version' and is passed **as argument** <br/> (for different batches or analysis)
  - in code, is stored in `self.dam_version` <br/>

4.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`

<br/>

##### 5. **`phenotypes/`**

5.1. subgroup: *`<pheno_subgroup>/`* (e.g. "simulations") <br/>
  
  - \<pheno_subgroup\> corresponds to 'phenos_version' and is passed **as argument** <br/> (is the type of phenotypes to analyse, e.g. phenotypes that went through different type of normalization.)
  - in code, is stored in `self.phenos_version` <br/>

5.2. datasets: <br/>

  - `['matrix']`;
  - `['col_header']['phenotype_ID']`;
  - `['col_header']['covariatesUsed']`; per each phenotype, the comma-sep string of covariates used - names corresponding to `covariate_ID` in `covariates/` (e.g. "sex,age,Batch", "sex,Batch"...)
  - `['row_header']['sample_ID']`
  
<br/>

##### 6. **`sex_cov/`** <br/>
6.1. subgroup: *`<sex_cov>/`* (e.g. "real") <br/>
  
  - \<sex_cov\> corresponds to 'sex_version' and is passed **as argument** <br/>
  - in code, is stored in `self.sex_version` <br/>

6.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`

<br/>

##### 7. **`subsets/`** <br/>
7.1 dataset: `['subset']` - **can be** passed as argument with option `-s`, e.g. excluding singletons <br/>


<br/><br/>
