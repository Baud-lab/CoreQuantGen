#!/usr/bin/env nextflow

include { run_VD as run_VD_uni       } from "./modules/run_VD/main" // used for uni and sex
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
    h5     = file(params.input, checkIfExists: true)

    // VD settings, the same for all tasks
    vd = [ pheno:   params.PHENOV,
           covs:    params.COVSV,
           cage:    params.CAGEV,
           dam:     params.DAMV,
           grm:     params.GRMV,
           sex:     params.SEXV,
           subset:  params.SUBSET,
           model:   vdMode,
           effects: params.EFFECTS,
           corr0:   params.CORR_NULL ]

    // ------ 1. run variance decomposition analysis
    // one task per row: meta = [id: trait1, row: n]
    // uni/sex: one row per phenotype; bi: one row per line (pair of phenotypes) of each combins file
    if (vdMode == "uni" || vdMode == "sex"){
      trait1 = "${params.GRMV}_${params.SUBSET}_${effs_name}"
      data = Channel.of( 1..params.phenos.toInteger() )
                    .map { i -> [ [id: trait1, row: i], h5, [] ] } // [] = no combins file
    }
    if (vdMode == "bi"){
      data = Channel.fromPath("${params.combinDir}/combins_${params.phenos}*.csv")
                    .flatMap { combins ->
                               def name   = combins.baseName.toString().replace("combins_", "")
                               def nrows  = combins.readLines().size()
                               (1..nrows).collect { i -> [ [id: name, row: i], h5, combins ] }
                             }
    }
    data.view { meta, _h5file, _combins -> "run_VD input: ${meta}" }

    // run process; outputs are grouped per trait1 (meta without row)
    if (vdMode == "uni" || vdMode == "sex"){
      run_VD_uni(data, vd, "./*/*/*/*/", "${outputVD}")
      vd_out = run_VD_uni.out
    } else if (vdMode =="bi"){
      run_VD_bi(data, vd, "./*/*/*/*/*/", "${outputVD}")
      vd_out = run_VD_bi.out
    }
    out_est = vd_out.est.map { meta, f -> [ [id: meta.id], f ] }.groupTuple()
    out_STE = vd_out.STE.map { meta, f -> [ [id: meta.id], f ] }.groupTuple()
    out_pkl = vd_out.pkl.map { meta, f -> [ [id: meta.id], f ] }.groupTuple()


    // ------ 2. gathering results for one phenotype all together ------//
    corr_null_name = "${params.CORR_NULL}".replaceAll(/,1$/, '_one').replaceAll(/,0$/, '_zero')

    concat_est("all_est.txt", out_est, corr_null_name, "${outputVD}")
    concat_STE("all_STE.txt", out_STE, corr_null_name, "${outputVD}")

    estNste_txt = concat_est.out.join(concat_STE.out) // [meta, est, STE]

    // C. preparing files:
    /*    - adding col names
    /*    - removing col with all NAs
    /*    - merging est and ste
    /*    - saving as Rdata - 1 object called "VCs"
    */
    txtToRdata("estNste.Rdata", estNste_txt, corr_null_name, params.SWAPtraits, "${outputVD}")
    txtToRdata.out.view{ _meta, rdata -> "RData output: ${rdata}" }

    // D. grouping pkl files in a single one
    group_pkl("VC.pkl.gz", out_pkl, "${outputVD}")

}
