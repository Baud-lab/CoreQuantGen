process group_pkl{
    publishDir("${outputVD}/${meta.value}/${meta.seed}/${suboutVD}", mode: 'copy')
    cache false // cache false so that repeat every time - this is to make sure that it is done everytime that run the analysis
    container "community.wave.seqera.io/library/python_pip_pandas:0a27509c7c11c41b"
    tag{ meta.id }
    
    input:
	  val(outfile)
	  tuple val(meta), path(sim_h5), path(input_files)
	  val(outputVD)
	  val(suboutVD)

    output:
	  tuple val(meta), path(outfile)
		
    script:
    """
    ls ${input_files} > pickles.list
    group_pickles.py -i pickles.list -o ${outfile}
    """
}
