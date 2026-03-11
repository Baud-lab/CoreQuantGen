process group_pkl{
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