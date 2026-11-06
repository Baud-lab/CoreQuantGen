process group_pkl{
    container "community.wave.seqera.io/library/python_pip_pandas:0a27509c7c11c41b"
    
    cache false // so that it runs again when running failed tasks
    publishDir("${outputVD}/", mode: 'copy')
    tag{ meta.id }
    
    input:
	  val(suffx)
	  tuple val(meta), path(input_files)
	  val(outputVD)

    output:
    tuple val(meta), path("${meta.id}_${suffx}")
		
    script:
    """
    ls ${input_files} > pickles.list
    group_pickles.py -i pickles.list -o ${meta.id}_${suffx}
    """
}
