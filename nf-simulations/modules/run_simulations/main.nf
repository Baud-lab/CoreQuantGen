// STEP 2: run simulations
process run_simulations{
  fair true // simulate in order of val1: seed1, seed2 ... ; val2: seed1, seed2 ... ; ...
  publishDir("${outputSim}/${value_oi}/${seed_oi}/", mode: 'copy')
  memory='5.G' 
  
  input:
  path(orH5) // this is always the same
  tuple val(value_oi), path(simCsv), val(seed_oi), val(grmv), val(cagev), val(damv), val(sexv), val(pop), val(npheno), val(model), val(ncages), val(subv), val(nas), val(orphenov) 
  val(outputSim)
  
  output:
  tuple val(value_oi), val(seed_oi), path("${pop}*${npheno}*${model}*${ncages}*${seed_oi}*.h5") 
  
  script:
  """
  ls ${orH5}
  ls ${simCsv}
  echo VALUE: ${value_oi} - SEED: ${seed_oi}
  
  run_simulations.R \
          --in_file ${orH5} \
          --GRM_version ${grmv} \
          --cage_version ${cagev} \
          --dam_version ${damv} \
          --sex_version ${sexv} \
          --out . \
          --prefix ${pop} \
          --model ${model} \
          --vars_file ${simCsv} \
          --phenos ${npheno} \
          --seed ${seed_oi} \
          --subset ${ncages} \
          --sub_version ${subv} \
          --missing ${nas} \
          --pheno_version ${orphenov}
  """
}
