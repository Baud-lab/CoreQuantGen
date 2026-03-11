process simP_fromH5{
  fair true // simulate in order of val1: seed1, seed2 ... ; val2: seed1, seed2 ... ; ...
  publishDir("${outputSim}/${value_oi}/${seed_oi}/", mode: 'copy')
  
  input:
  tuple val(value_oi), val(seed_oi), path(sim_h5)
  val(outputSim)

  output:
  tuple val(value_oi), val(seed_oi), path("params_${params.simMODEL}_V${value_oi}_S${seed_oi}.txt")
  
  """
  echo -e "getting params"
  get_simParams.R ${sim_h5} params_${params.simMODEL}_V${value_oi}_S${seed_oi}.txt
  """
}
