process concat_files{
    container "tonnelene/py_corequantgen:1.1" // only needs bash and coreutils; same image as run_VD
    publishDir("${outputVD}/${meta.value}/${meta.seed}/${suboutVD}", mode: 'copy')
    cache false // cache false so that repeat every time - this is to make sure that it is done everytime that run the analysis
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
    ls $sim_h5
	  cat $input_files >> ${outfile}
    """

}
