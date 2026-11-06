process txtToRdata{
  container "tonnelene/r-simulations:1.0"
  
  cache false // so that it runs again when running failed tasks
  publishDir("$outputVD/", mode: 'copy')
  tag{ meta.id }
  
  input:
  val(suffx) 
  tuple val(meta), path(est_file), path(ste_file)
  val(corr0)
  val(swaptraits)
  val(outputVD)

  output:
  tuple val(meta), path("${meta.id}_${corr0}_${suffx}") 
  
  script:
  """
  VDest_to_rdata.R --est $est_file --ste $ste_file --out ${meta.id}_${corr0}_${suffx} --corr0 ${corr0} --swap ${swaptraits}
  """
}
