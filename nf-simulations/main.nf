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
suboutVD  = "${params.vdMODEL}variate/${params.PHENOV}/" // this is to replicate dir structure from expbivar // ${params.GRMV}"+"_"+"${params.EFFECTS}".replace(",","_")

// GENERAL WORKFLOW:
workflow{
    values  = Channel.fromList(params.V)
    seeds   = Channel.fromList(params.SEED)
    simMode = params.simMODEL
    vdMode  = params.vdMODEL
    nphenos = channel.of(params.NPHENO)
    
    //def val_seed = Channel.of(zip(values, seeds))
    
    // ------ 1. creating file with parameters to simulate, setting a specific value for a specific parameter
    starting_pars( "${params.original_params}", params.SIT, params.P, values )
    
    // combine seeds to each value 
    val_seed = starting_pars.out.combine( seeds )
    // store simulations characteristics - input of sun_simulations
    // tuple val(value_oi), path(simCsv), val(seed_oi), val(grmv), val(cagev), val(damv), val(sexv), val(pop), val(npheno), val(model), val(ncages), val(subv), val(nas), val(orphenov) 
    sim_char = val_seed.map{ [it[0], 
                              it[1], 
                              it[2], 
                              params.GRMV, 
                              params.CAGEV, 
                              params.DAMV,
                              params.SEXV,
                              params.POP, 
                              params.NPHENO, 
                              simMode, 
                              params.NCAGES, 
                              params.SUBV, 
                              params.NAs, 
                              params.ORPHENOV ]
                           } 
    
    // ------ 2. run simulations 
    run_simulations("${params.original_H5}", sim_char, "${outputSim}")
    simFiles = run_simulations.out // value, seed, sim_h5
    //simFiles.view()

    // ------ 3. getting simulated parameters (this will have the proportionals as well) from h5 
    //           another way to keep track of what has been simulated and how
    simP_fromH5(simFiles, "${outputSim}")
    simPfiles = simP_fromH5.out // value, seed, param.txt -> this will be used to get simulated proproportional params
    //simPfiles.view()
    

    // ------ 4. creating file with combinations of pairs, this is one time only - can put in "set" directory
    // creating list of number of rows - 1 row, 1 VD run
    if (vdMode=="uni" || vdMode=="sex"){
      // based on the number of phenos for univariate: 1 row 1 pheno
      rows = nphenos.toInteger()
                    .flatMap { value -> 
                              (1..value).collect { i -> [ i, null ] }
              }
              // //value -> (1..value) }
    }else if (vdMode=="bi"){
      // based on combins_file (i.e. create_combin.out) in bivariate: 1 row 1 line (i.e. 1 pair)
      create_combin(nphenos) 
      rows = create_combin.out.flatMap { file -> 
                                      //def sampleName = file.baseName.toString().replace("combins_", "")
                                      // Read the file and enumerate the lines
                                      def lines = file.readLines()
                                      def lineNumbers = (1..lines.size()).toList()

                                      // Emit the enumerated line numbers and file name
                                      return lineNumbers.collect { lineNumber -> [lineNumber, file]}
                                    }
    }

    // ------ 5. run variance decomposition analysis 
    // creating tuple data to input to run_expbivar - check in modules/run_VD/main.nf the order and the values
    // rows has: linenumber/pheno to analyse + combin_file 
    // simFiles has: val(value_oi), val(seed_oi), path("${pop}*${npheno}*${model}*${ncages}*${seed_oi}*.h5") 
    data = rows.combine(simFiles)
               .map{ [ it[0], 
                       it[1],
                       it[2], 
                       it[3], 
                       it[4], 
                       params.PHENOV, 
                       params.COVSV, 
                       params.CAGEV, 
                       params.DAMV, 
                       params.GRMV,
                       params.SEXV,
                       params.SUBV, 
                       vdMode, 
                       params.EFFECTS, 
                       params.CORR_NULL ] }
    //data.view()
    // run process 
    // tuple val(row_num), val(combins_path), val(value_oi), val(seed_oi), path(sim_h5), val(pheno), val(covs), val(cage), val(dam), val(grm), val(sexv), val(subset), val(model), val(effects), val(corr0)
    if (vdMode == "uni" || vdMode=="sex"){
      run_VD_uni(data, "./*/*/*/*/") 
      out_est = run_VD_uni.out.est.groupTuple(by:[0,1])
      out_STE = run_VD_uni.out.STE.groupTuple(by:[0,1])
      out_pkl = run_VD_uni.out.pkl.groupTuple(by:[0,1]) 
    } else if (vdMode =="bi"){
      run_VD_bi(data, "./*/*/*/*/*/") 
      out_est = run_VD_bi.out.est.groupTuple(by:[0,1])
      out_STE = run_VD_bi.out.STE.groupTuple(by:[0,1])
      out_pkl = run_VD_bi.out.pkl.groupTuple(by:[0,1])
    } 
    
    // ------ 6. collecting output from variance decompostion analysis
    // TODO: for real data, CHECK if/else for uni/bi has to be implemented because of the structure of directories and file names (or maybe not? Have to think about this)
    //if (mode == "uni"){
    //}else if (mode =="bi"){}
    
    // A. _est and _ste files grouped and mapped to h5 of input (in 'simFiles')
    single_est = simFiles.combine(out_est, by:[0,1])
    //out_est.map{[it[0], it[1], it[3]]}.view()
    //single_est.view()
    
    single_STE = simFiles.combine(out_STE, by:[0,1])
    //out_STE.view()
    
    single_pkl = simFiles.combine(out_pkl, by:[0,1])
    
    // B. concat all est and all STE in a single file - will have 1 file of _est and 1 of _STE per .h5
    effs_name = "${params.EFFECTS}".replace(",","_")
    corr_null_name = "${params.CORR_NULL}".replaceAll(/,1$/, '_one').replaceAll(/,0$/, '_zero')

    out_prefx = "${params.GRMV}_${effs_name}_${corr_null_name}_all"
  
    concat_est("${out_prefx}_est.txt", single_est, "${outputVD}", "${suboutVD}")
    concat_STE("${out_prefx}_STE.txt", single_STE, "${outputVD}", "${suboutVD}")
    
    estNste_txt = concat_est.out.combine(concat_STE.out, by:[0,1])
    estNste_txt = simFiles.combine(estNste_txt, by:[0,1])
    //estNste_txt.view()
    
    // C. preparing files:
    /*    - adding col names
    /*    - removing col with all NAs
    /*    - merging est and ste
    /*    - saving as Rdata - 1 object called "VCs"
    */
    txtToRdata("${out_prefx}_estNste.Rdata", estNste_txt.map{it[0,1,3,4]}, params.CORR_NULL, "${outputVD}", "${suboutVD}")
    //alltxt_toRdata.out.view()
    estNste = simFiles.combine(txtToRdata.out, by:[0,1])
    estNste.view()
    //}else{
    //  simFiles.view()
    //}
    
    // D. grouping pkl files in a single one 
    group_pkl("${out_prefx}_VC.pkl.gz", single_pkl, "${outputVD}", "${suboutVD}")
}

