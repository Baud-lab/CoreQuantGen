process simP_fromH5{
  fair true // simulate in order of val1: seed1, seed2 ... ; val2: seed1, seed2 ... ; ...
  publishDir("${outputSim}/${meta.value}/${meta.seed}/", mode: 'copy')
  container "tonnelene/r-simulations:1.0"
  tag{ meta.id }
  
  input:
  tuple val(meta), path(sim_h5)
  val(outputSim)

  output:
  tuple val(meta), path("params_${params.simMODEL}_V${meta.value}_S${meta.seed}.txt")
  
  script:
  """
  echo -e "getting params"
  get_simParams.R ${sim_h5} params_${params.simMODEL}_V${meta.value}_S${meta.seed}.txt
  """
}
