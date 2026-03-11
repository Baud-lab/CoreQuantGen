process concat_files{
  
  cache false // so that it runs again when running failed tasks
  publishDir("$outputVD/", mode: 'copy') 
  tag{ trait1 }

  input:
  val(suffx)
  tuple val(trait1), path(input_files)
  val(corr0)
  val(outputVD)

  output:
  tuple val(trait1), val(corr0), file("${trait1}_${corr0}_${suffx}")
  
  script:
  """
	cat $input_files >> ${trait1}_${corr0}_${suffx}
  """
}
