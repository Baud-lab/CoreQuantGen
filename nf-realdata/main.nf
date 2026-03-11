#!/usr/bin/env nextflow

include { run_VD as run_VD_uni       } from "./modules/run_VD/main" // should add sexvariate option
include { run_VD as run_VD_bi        } from "./modules/run_VD/main" 
include { concat_files as concat_est } from "./modules/concat_files/main"
include { concat_files as concat_STE } from "./modules/concat_files/main"
include { txtToRdata                 } from "./modules/txtToRdata/main"
include { group_pkl                  } from "./modules/group_pkl/main"

// DEFINING OUTIR 
effs_name = "${params.EFFECTS}".replace(",","_")
outputVD  = "${params.outdir}/VD/${params.vdMODEL}variate/${params.PHENOV}/${params.GRMV}_${params.SUBSET}_${effs_name}"

// GENERAL WORKFLOW:
workflow{
    vdMode = params.vdMODEL
    
    // ------ 1. run variance decomposition analysis 
    // creating list of number of rows - 1 row, 1 VD run
    // corresponds to actual task in uni; to pheno1 paired with all other pheno2 in bi
    if (vdMode == "uni" || vdMode == "sex"){
      trait1 = "${params.GRMV}_${params.SUBSET}_${effs_name}"
      trait1_rows = Channel.from( 1..params.phenos.toInteger() )
                           .map { i -> [trait1, null, i] }
    }
    if (vdMode == "bi"){
      trait1_rows = Channel.fromPath("${params.combinDir}/combins_${params.phenos}*.csv")
                           .flatMap { file -> 
                                      def sampleName = file.baseName.toString().replace("combins_", "")
                                      // Read the file and enumerate the lines
                                      def lines = file.readLines()
                                      def lineNumbers = (1..lines.size()).toList()

                                      // Emit the sample name, file name, and enumerated line numbers
                                       return lineNumbers.collect { lineNumber -> [sampleName, file, lineNumber]}
                                    }
      }
    
    // creating tuple data to input to run expbivar 
    // tuple val(trait1), val(combins_path), val(row_num), path(h5), val(pheno), val(covs), val(cage), val(dam), val(grm), val(sexv), val(subset), val(model), val(effects), val(corr0)
    data = trait1_rows
              .map{ it -> [ it[0], 
                            it[1], 
                            it[2], 
                            params.input, 
                            params.PHENOV, 
                            params.COVSV, 
                            params.CAGEV, 
                            params.DAMV, 
                            params.GRMV, 
                            params.SEXV,
                            params.SUBSET, 
                            vdMode, 
                            params.EFFECTS, 
                            params.CORR_NULL ] } 
    data.view()
    // TODO: have to check work of CORR_NULL here

    // run process 
    if (vdMode == "uni" || vdMode == "sex"){ // TODO: need to check this
      run_VD_uni(data, "./*/*/*/*/", "${outputVD}") 
      out_est = run_VD_uni.out.est.groupTuple()      
      out_STE = run_VD_uni.out.STE.groupTuple()
      out_pkl = run_VD_uni.out.pkl.groupTuple() 
    } else if (vdMode =="bi"){
      //trait1 = data.map{it[0]}
      //trait1.view()
      run_VD_bi(data, "./*/*/*/*/*/", "${outputVD}")
      out_est = run_VD_bi.out.est.groupTuple()
      out_STE = run_VD_bi.out.STE.groupTuple()
      out_pkl = run_VD_bi.out.pkl.groupTuple()
    }
    
    
    // ------ 2. gathering results for one phenotype all together ------//
    // TODO: have to check work of CORR_NULL here
    corr_null_name = "${params.CORR_NULL}".replaceAll(/,1$/, '_one').replaceAll(/,0$/, '_zero')

    concat_est("all_est.txt", out_est, corr_null_name, "${outputVD}") 
    //concat_est.out.view()
    concat_STE("all_STE.txt", out_STE, corr_null_name, "${outputVD}") 
    //concat_STE.out.view()
    
    estNste_txt = concat_est.out.combine(concat_STE.out, by:[0,1])
                                .map{ it[ 0, 1, 2, 3] }

    // C. preparing files:
    /*    - adding col names
    /*    - removing col with all NAs
    /*    - merging est and ste
    /*    - saving as Rdata - 1 object called "VCs"
    */
    txtToRdata("estNste.Rdata", estNste_txt, params.SWAPtraits, "${outputVD}") 
    txtToRdata.out.view()

    // D. grouping pkl files in a single one 
    group_pkl("VC.pkl.gz", out_pkl, "${outputVD}") 

}
