// STEP 2: run simulations
process run_simulations{
  fair true // simulate in order of val1: seed1, seed2 ... ; val2: seed1, seed2 ... ; ...
  publishDir("${outputSim}/${meta.value}/${meta.seed}/", mode: 'copy')
  container "tonnelene/r-simulations:1.0"
  tag{ meta.id }
  
  label 'mem_5'
  
  input:
  path(orH5) // this is always the same
  tuple val(meta), path(simCsv) // meta: [id, value, seed]
  val(sim)                      // simulation settings, the same for all tasks: grm, cage, dam, sex, pop, npheno, model, ncages, subv, nas, orphenov
  val(outputSim)
  
  output:
  tuple val(meta), path("${sim.pop}*${sim.npheno}*${sim.model}*${sim.ncages}*${meta.seed}*.h5") 
  
  script:
  """
  ls ${orH5}
  ls ${simCsv}
  echo VALUE: ${meta.value} - SEED: ${meta.seed}
  
  run_simulations.R \
          --in_file ${orH5} \
          --GRM_version ${sim.grm} \
          --cage_version ${sim.cage} \
          --dam_version ${sim.dam} \
          --sex_version ${sim.sex} \
          --out . \
          --prefix ${sim.pop} \
          --model ${sim.model} \
          --vars_file ${simCsv} \
          --phenos ${sim.npheno} \
          --seed ${meta.seed} \
          --subset ${sim.ncages} \
          --sub_version ${sim.subv} \
          --missing ${sim.nas} \
          --pheno_version ${sim.orphenov}
  """
}
