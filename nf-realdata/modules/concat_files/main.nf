process concat_files{
  container "tonnelene/py_corequantgen:1.1" // only needs bash and coreutils; same image as run_VD
  
  cache false // so that it runs again when running failed tasks
  publishDir("$outputVD/", mode: 'copy') 
  tag{ meta.id }

  input:
  val(suffx)
  tuple val(meta), path(input_files)
  val(corr0)
  val(outputVD)

  output:
  tuple val(meta), path("${meta.id}_${corr0}_${suffx}")
  
  script:
  """
	cat $input_files >> ${meta.id}_${corr0}_${suffx}
  """
}
