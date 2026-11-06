#!/usr/bin/env nextflow

// MODULES:
include{ starting_pars              } from "./modules/starting_pars/main"
include{ run_simulations            } from "./modules/run_simulations/main"
include{ simP_fromH5                } from "./modules/simP_fromH5/main"
include{ create_combin              } from "./modules/create_combin/main"
include{ run_VD as run_VD_uni       } from "./modules/run_VD/main"
include{ run_VD as run_VD_bi        } from "./modules/run_VD/main"
include{ concat_files as concat_est } from "./modules/concat_files/main"
include{ concat_files as concat_STE } from "./modules/concat_files/main"
include{ txtToRdata                 } from "./modules/txtToRdata/main"
include{ group_pkl                  } from "./modules/group_pkl/main"

// DEFINING DIRS:
subdir    = "cage_${params.NCAGES}/set$params.SIT/${params.P}/"
outputSim = "${params.output_folder}/mockphenos/${subdir}" // "${params.output_folder}/simulations/${subdir}"
outputVD  = "${params.output_folder}/VD/${subdir}"
if (params.PERM == "yes"){
  suboutVD  = "${params.vdMODEL}variate/${params.PHENOV}/permutations" // this is to replicate dir structure from expbivar // ${params.GRMV}"+"_"+"${params.EFFECTS}".replace(",","_")
}else{
  suboutVD  = "${params.vdMODEL}variate/${params.PHENOV}/"
}
// GENERAL WORKFLOW:
workflow{
    values  = Channel.fromList(params.V)
    seeds   = Channel.fromList(params.SEED)
    simMode = params.simMODEL
    vdMode  = params.vdMODEL
    nphenos = channel.of(params.NPHENO)

    // simulation settings, the same for all tasks
    sim = [ grm:      params.GRMV,
            cage:     params.CAGEV,
            dam:      params.DAMV,
            sex:      params.SEXV,
            pop:      params.POP,
            npheno:   params.NPHENO,
            model:    simMode,
            ncages:   params.NCAGES,
            subv:     params.SUBV,
            nas:      params.NAs,
            orphenov: params.ORPHENOV ]

    // VD settings, the same for all tasks
    vd = [ pheno:   params.PHENOV,
           covs:    params.COVSV,
           cage:    params.CAGEV,
           dam:     params.DAMV,
           grm:     params.GRMV,
           sex:     params.SEXV,
           subset:  params.SUBV,
           model:   vdMode,
           effects: params.EFFECTS,
           corr0:   params.CORR_NULL,
           perm:    params.PERM ]

    if(!params.skip_simulations){

        // ------ 1. creating file with parameters to simulate, setting a specific value for a specific parameter
        starting_pars( "${params.original_params}", params.SIT, params.P, values )

        // combine seeds to each value; one simulation per value x seed: meta = [id, value, seed]
        sim_pars = starting_pars.out
                        .combine( seeds )
                        .map { value, simCsv, seed -> [ [id: "V${value}_S${seed}", value: value, seed: seed], simCsv ] }

        // ------ 2. run simulations
        run_simulations("${params.original_H5}", sim_pars, sim, "${outputSim}")
        simFiles = run_simulations.out // [meta, sim_h5]

        // ------ 3. getting simulated parameters (this will have the proportionals as well) from h5
        //           another way to keep track of what has been simulated and how
        simP_fromH5(simFiles, "${outputSim}")


        // ------ 4. creating file with combinations of pairs, this is one time only - can put in "set" directory
        // creating list of number of rows - 1 row, 1 VD run: [row, combins] ([] = no combins file)
        if (vdMode=="uni" || vdMode=="sex"){
          // based on the number of phenos for univariate: 1 row 1 pheno
          rows = nphenos.toInteger()
                        .flatMap { value -> (1..value).collect { i -> [ i, [] ] } }
        }else if (vdMode=="bi"){
          // based on combins_file (i.e. create_combin.out) in bivariate: 1 row 1 line (i.e. 1 pair)
          create_combin(nphenos)
          rows = create_combin.out.flatMap { combins ->
                                          def nrows = combins.readLines().size()
                                          (1..nrows).collect { i -> [ i, combins ] }
                                        }
        }
    } else {
      // NB: will have to do the files one by one for now
      simFiles = Channel.of( [ [id: "V${params.V}_S${params.SEED}", value: params.V, seed: params.SEED], file(params.sim_h5) ] )

      rows = Channel.fromPath("${params.combin}")
                    .flatMap{ combins ->
                               def nrows = combins.readLines().size()
                               (1..nrows).collect { i -> [ i, combins ] }
                    }
    }


    // ------ 5. run variance decomposition analysis
    // one VD task per row and simulated h5: meta = [id, value, seed, row]
    data = rows.combine(simFiles)
               .map { row, combins, meta, sim_h5 -> [ meta + [row: row], sim_h5, combins ] }
    //data.view()
    if (vdMode == "uni" || vdMode=="sex"){
      run_VD_uni(data, vd, "./*/*/*/*/")
      vd_out = run_VD_uni.out
    } else if (vdMode =="bi"){
      run_VD_bi(data, vd, "./*/*/*/*/*/")
      vd_out = run_VD_bi.out
    }
    // grouping outputs per simulated h5 (meta without row)
    out_est = vd_out.est.map { meta, f -> [ meta.subMap(['id', 'value', 'seed']), f ] }.groupTuple()
    out_STE = vd_out.STE.map { meta, f -> [ meta.subMap(['id', 'value', 'seed']), f ] }.groupTuple()
    out_pkl = vd_out.pkl.map { meta, f -> [ meta.subMap(['id', 'value', 'seed']), f ] }.groupTuple()

    // ------ 6. collecting output from variance decompostion analysis
    // A. _est and _ste files grouped and mapped to h5 of input (in 'simFiles'): [meta, sim_h5, files]
    single_est = simFiles.join(out_est)
    single_STE = simFiles.join(out_STE)
    single_pkl = simFiles.join(out_pkl)

    // B. concat all est and all STE in a single file - will have 1 file of _est and 1 of _STE per .h5
    effs_name = "${params.EFFECTS}".replace(",","_")
    corr_null_name = "${params.CORR_NULL}".replaceAll(/,1$/, '_one').replaceAll(/,0$/, '_zero')

    out_prefx = "${params.GRMV}_${effs_name}_${corr_null_name}_all"

    concat_est("${out_prefx}_est.txt", single_est, "${outputVD}", "${suboutVD}")
    concat_STE("${out_prefx}_STE.txt", single_STE, "${outputVD}", "${suboutVD}")

    estNste_txt = concat_est.out.join(concat_STE.out) // [meta, est, STE]

    // C. preparing files:
    /*    - adding col names
    /*    - removing col with all NAs
    /*    - merging est and ste
    /*    - saving as Rdata - 1 object called "VCs"
    */
    txtToRdata("${out_prefx}_estNste.Rdata", estNste_txt, params.CORR_NULL, "${outputVD}", "${suboutVD}")
    estNste = simFiles.join(txtToRdata.out) // [meta, sim_h5, Rdata]
    estNste.view{ _meta, _sim_h5, rdata -> "RData output: ${rdata}" }

    // D. grouping pkl files in a single one
    group_pkl("${out_prefx}_VC.pkl.gz", single_pkl, "${outputVD}", "${suboutVD}")
}
