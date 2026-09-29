process group_pkl{
    container "community.wave.seqera.io/library/python_pip_pandas:0a27509c7c11c41b"
    
    cache false // so that it runs again when running failed tasks
    publishDir("${outputVD}/", mode: 'copy')
    tag{ trait1 }
    
    input:
	  val(suffx)
	  tuple val(trait1), path(input_files)
	  val(outputVD)

    output:
    tuple val(trait1),
    file("${trait1}_${suffx}")
		
    script:
    """
    ls ${input_files} > pickles.list
    group_pickles.py -i pickles.list -o ${trait1}_${suffx}
    """
}