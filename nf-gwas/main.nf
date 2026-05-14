#!/usr/bin/env nextflow

include { GET_PHENOS }  from './modules/get_phenos/main.nf'
include { NULL_COVAR }  from './modules/null_covar/main.nf'
include { MAP_LOCO   }  from './modules/map_loco/main.nf'

/*
 * Required params
 */

// params.h5       = "/users/abaud/htonnele/PRJs/P50_HSrats/microbiome_ht/output/microb_prep/h5files/HSrats_AIE_NY.h5"
// params.outdir   = "/users/abaud/htonnele/git/me/GWAS/test_HSrats"
// 
// params.phenov = "sgb_cecal"
// params.covsv  = "None"
// params.cagev  = "all657"
// params.grmv   = "round10_2"
// params.damv   = "mother_name"
// params.subset = "NY"
// params.effects    = "DGE,cageEffect,maternalEffect"
// 
// params.dsnpv = "round10_2_direct"
// params.gwas  = "dGWAS"

// /*
//  * Can be phenotype names or phenotype column numbers.
//  *
//  * Examples:
//  *   --phenotypes 216
//  *   --phenotypes 216,217,218
//  *   --phenotypes trait_a,trait_b
//  */
// params.phenotypes = "216"
// 
// /*
//  * Chromosomes needed for null covariates and then passed as a list to map_loco.
//  *
//  * Example:
//  *   --chromosomes 1,2,3,4,5,6,7,8,9,10
//  */
// params.chromosomes = "10"
// 
//params.pheno_list = null // if null, execute module to get one - easy R code, reading h5, taking col names and output pheno_list 


workflow {
    // Define channel for input h5 file
    ch_h5 = Channel.fromPath(params.h5)
    
    // Define channel of phenotypes - will have one result per phenotype
    if (params.pheno_list) {
        ch_pheno_list = Channel.fromPath(params.pheno_list)
                           .flatMap { file -> 
                                      // Read the file and enumerate the lines
                                      def lines = file.readLines()
                                      def lineNumbers = (1..lines.size()).toList()
                                      // Emit  
                                       return lineNumbers.collect { lineNumber -> [id: lines[lineNumber - 1], num: lineNumber]}
                                    }

    } else {
      // execute module to generate pheno_list if noe existing
      ch_get_phenos_input = ch_h5.map { h5 ->
          tuple(h5, h5.parent, params.phenov)
      }
      GET_PHENOS(ch_get_phenos_input)
      
      ch_pheno_list = GET_PHENOS.out.flatMap { file -> 
                                      // Read the file and enumerate the lines
                                      def lines = file.readLines()
                                      def lineNumbers = (1..lines.size()).toList()
                                      // Emit  
                                      return lineNumbers.collect { lineNumber -> [id: lines[lineNumber - 1], num: lineNumber]}
                                    }
      
    }
    //ch_pheno_list.view()

    /* Phenotype inputs.
     * These are the values passed to:
     *   -p <phenotype>
     * They may be numbers or names.
     */
    if(params.phenotypes == "all"){
      ch_traits = ch_pheno_list
    }else{
      phenos = params.phenotypes.toString().split(',').collect { it.trim() } // trim removes blank spaces
      //phenos.view()
      /* Resolve phenotype input to trait name.
       *    Output:
       *    tuple(pheno_input, trait)
       */
      ch_traits = ch_pheno_list
            .filter { meta ->
                phenos.contains(meta.id.toString()) ||
                phenos.contains(meta.num.toString())
            }
    }
    //ch_traits.view()
    
    // Chromosomes as individual values for NULL_COVAR.
    chrs = Channel.fromList(params.chromosomes.toString().split(',')*.trim())
    //chrs.view()
    
    // Chromosomes as one comma-separated list for MAP_LOCO.
    chrs_list = Channel.from(params.chromosomes.toString())
   
    /* Create one meta object per trait x chromosome.
     *    meta.null_dir is where covRunner.py writes: 
     *    params.outdir/null_covars_LOCO/univariate/<phenov>/<grmv>_<subset>_<eff>/<trait>/<trait>_chr*.h5
     */
    ch_meta = ch_traits
        .combine(chrs)
        .map { meta, chr ->
            def eff_dir = "$params.effects".replaceAll(',', '_')
            def null_dir = ["null_covars_LOCO", "univariate", "${params.phenov}", "${params.grmv}_${params.subset}_${eff_dir}", meta.id]
                               .join('/')
            // output the meta as
            [   id       : meta.id,
                num      : meta.num,
                chr      : chr,
                null_dir : null_dir
            ]
        }
    //ch_meta.view()

//if(0==1){
    // Input channel for NULL_COVAR
    ch_null_covar_input = ch_meta
        .combine(ch_h5)
        .map { meta, h5 ->
            tuple(
                meta, // id: trait, num: num, chr: chr, null_dir: null_dir
                h5,
                params.phenov,
                params.covsv,
                params.cagev,
                params.grmv,
                params.damv,
                params.effects,
                params.subset
                )
        }
    ch_null_covar_input.view()


    // 1. Run one NULL_COVAR task per trait x chromosome.
    NULL_COVAR(ch_null_covar_input)
    
    NULL_COVAR.out.view()
    

    /* Group all chromosomes by trait.
     *    NULL_COVAR emits:
     *      tuple(meta, null_covar_h5)
     *
     * Group by meta.id so all chromosome H5 files for the same trait
     */
    ch_null_covars_by_trait = NULL_COVAR.out
        .map { meta, null_h5 ->
            tuple([id: meta.id, num: meta.num, null_dir: meta.null_dir], null_h5)
        }
        .groupTuple()
    //ch_null_covars_by_trait.view()

    // // Gather all null-covar H5 files for one trait into one directory
    // GROUP_NULL_COVARS(ch_null_covars_by_trait)
    // GROUP_NULL_COVARS.out.dir.view()

    // Add full chromosome list for MAP_LOCO
    //    pval_dir = params.outdir/pvalues_LOCO/univariate/<dsnpv>/<phenov>/<gwas>/<grmv>_<subset>_<eff>
    snpv = [params.dsnpv, params.ssnpv].findAll{ it != null && it != '' && it != 'None' }.join('.')
    ch_map_loco_input = ch_null_covars_by_trait
       .combine(chrs_list)
       .combine(ch_h5)
       .map { meta, covars, chrs_list, h5 ->
           def eff_dir = "$params.effects".replaceAll(',', '_')
           meta.pval_dir = ["pvalues_LOCO", "univariate", "${snpv}", "${params.phenov}", "${params.gwas}","${params.grmv}_${params.subset}_${eff_dir}"]
                            .join('/')
           tuple(
               meta, //id:, num:, null_dir:, pval_dir:
               covars,
               chrs_list,
               h5,
               params.phenov,
               params.covsv,
               params.cagev,
               params.grmv,
               params.damv,
               params.dsnpv,
               params.ssnpv,
               params.effects,
               params.gwas,
               params.subset
           )
       }
    ch_map_loco_input.view()
    
    /*
    * Run MAP_LOCO once per trait.
    */
    MAP_LOCO(ch_map_loco_input)
    MAP_LOCO.out.view()
//}
  
}