## HOW TO WRITE the .h5

Will have different H5I_GROUPS (which names need to match the following ones): <br/>

### 1. **`GRM/`** or **`GRM_LOCO/`**
1.1. subgroup: *`<GRM_subgroup>/`* (e.g. "Andres_kinship") <br/>
  
  - \<GRM_subgroup\> corresponds to 'GRM_version' and is passed **as argument** <br/>
  - in code, is stored in `self.GRM_version` <br/>

1.2. datasets: <br/>

  - `['matrix']`; 
  - `['row_header']['sample_ID']`


### 2. **`cages/`**
2.1. subgroup: *`<cage_subgroup>/`* (e.g. "real") <br/>
  
  - \<cage_subgroup\> corresponds to 'cage_version' and is passed **as argument** <br/> (for different batches or analysis)
  - in code, is stored in `self.cage_version` <br/>

2.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`

<br/>

### 3. **`covariates/`**
3.1. subgroup: *`<cov_subgroup>/`* (e.g. "from_file") <br/>
  
  - \<cov_subgroup\> corresponds to 'covs_version' and is passed **as argument** <br/> (similar to pheno_subgroup)
  - in code, is stored in `self.covs_version` <br/>

3.2. datasets: <br/>

  - `['matrix']`;
  - `['col_header']['covariate_ID']`; e.g. "sex", "age", "Batch"
  - `['row_header']['sample_ID']`

<br/>

### 4. **`dam/`**
4.1. subgroup: *`<dam_subgroup>/`* (e.g. "simulated") <br/>
  
  - \<dam_subgroup\> corresponds to 'dam_version' and is passed **as argument** <br/> (for different batches or analysis)
  - in code, is stored in `self.dam_version` <br/>

4.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`

<br/>

### 5. **`phenotypes/`**
5.1. subgroup: *`<pheno_subgroup>/`* (e.g. "simulations") <br/>
  
  - \<pheno_subgroup\> corresponds to 'phenos_version' and is passed **as argument** <br/> (is the type of phenotypes to analyse, e.g. phenotypes that went through different type of normalization.)
  - in code, is stored in `self.phenos_version` <br/>

5.2. datasets: <br/>

  - `['matrix']`;
  - `['col_header']['phenotype_ID']`;
  - `['col_header']['covariatesUsed']`; per each phenotype, the comma-sep string of covariates used - names corresponding to `covariate_ID` in `covariates/` (e.g. "sex,age,Batch", "sex,Batch"...)
  - `['row_header']['sample_ID']`
  
<br/>

### 6. **`sex_cov/`** <br/>
6.1. subgroup: *`<sex_cov>/`* (e.g. "real") <br/>
  
  - \<sex_cov\> corresponds to 'sex_version' and is passed **as argument** <br/>
  - in code, is stored in `self.sex_version` <br/>

6.2. datasets: <br/>

  - `['array']`; 
  - `['sample_ID']`

<br/>

### 7. **`subsets/`** <br/>
7.1 dataset: `['subset']` - **can be** passed as argument with option `-s`, e.g. excluding singletons <br/>

<br/>

### Example:
NB: not all groups are needed if not used (e.g. dam is absent here); can have additional groups
```
                            group           name       otype
0                               /            GRM   H5I_GROUP
1                            /GRM pruned_dosages   H5I_GROUP
2             /GRM/pruned_dosages         matrix H5I_DATASET
3             /GRM/pruned_dosages     row_header   H5I_GROUP
4  /GRM/pruned_dosages/row_header      sample_ID H5I_DATASET
5                               /          cages   H5I_GROUP
6                          /cages         all623   H5I_GROUP
7                   /cages/all623          array H5I_DATASET
8                   /cages/all623      sample_ID H5I_DATASET
9                               /     covariates   H5I_GROUP
10                    /covariates        noBatch   H5I_GROUP
11            /covariates/noBatch     col_header   H5I_GROUP
12 /covariates/noBatch/col_header   covariate_ID H5I_DATASET
13            /covariates/noBatch         matrix H5I_DATASET
14            /covariates/noBatch     row_header   H5I_GROUP
15 /covariates/noBatch/row_header      sample_ID H5I_DATASET
16                              /     phenotypes   H5I_GROUP
17                    /phenotypes        noBatch   H5I_GROUP
18            /phenotypes/noBatch     col_header   H5I_GROUP
19 /phenotypes/noBatch/col_header covariatesUsed H5I_DATASET
20 /phenotypes/noBatch/col_header   phenotype_ID H5I_DATASET
21            /phenotypes/noBatch         matrix H5I_DATASET
22            /phenotypes/noBatch     row_header   H5I_GROUP
23 /phenotypes/noBatch/row_header      sample_ID H5I_DATASET
24                              /        sex_cov   H5I_GROUP
25                       /sex_cov            all   H5I_GROUP
26                   /sex_cov/all          array H5I_DATASET
27                   /sex_cov/all      sample_ID H5I_DATASET
28                              /        subsets   H5I_GROUP
29                       /subsets        females H5I_DATASET
30                       /subsets        include H5I_DATASET
31                       /subsets include_not435 H5I_DATASET
32                       /subsets          males H5I_DATASET
```