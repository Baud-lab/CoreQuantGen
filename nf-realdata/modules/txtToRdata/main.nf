process txtToRdata{
  cache false // so that it runs again when running failed tasks
  publishDir("$outputVD/", mode: 'copy')
  tag{ trait1 }
  
  input:
  val(suffx) 
  tuple val(trait1), val(corr0), path(est_file), path(ste_file)
  val(swaptraits)
  val(outputVD)

  output:
  tuple val(trait1), val(corr0),
  file("${trait1}_${corr0}_${suffx}") 
  
  script:
  """
  VDest_to_rdata.R --est $est_file --ste $ste_file --out ${trait1}_${corr0}_${suffx} --corr0 ${corr0} --swap ${swaptraits}
  """
}
