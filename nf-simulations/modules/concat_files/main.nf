process concat_files{
    publishDir("${outputVD}/${value_oi}/${seed_oi}/${suboutVD}", mode: 'copy')
    cache false // cache false so that repeat every time - this is to make sure that it is done everytime that run the analysis
    
    input:
	  val(outfile)
	  tuple val(value_oi), val(seed_oi), path(sim_h5), path(input_files)
	  val(outputVD)
	  val(suboutVD)

    output:
	  tuple val(value_oi), val(seed_oi), path(outfile)
		
    script:
    """
    ls $sim_h5
	  cat $input_files >> ${outfile}
    """

}
