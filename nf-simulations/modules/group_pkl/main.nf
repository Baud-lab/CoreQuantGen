process group_pkl{
    publishDir("${outputVD}/${value_oi}/${seed_oi}/${suboutVD}", mode: 'copy')
    cache false // cache false so that repeat every time - this is to make sure that it is done everytime that run the analysis
    
    input:
	  val(outfile)
	  tuple val(value_oi), val(seed_oi), 
	        path(sim_h5), path(input_files)
	  val(outputVD)
	  val(suboutVD)

    output:
	  tuple val(value_oi), val(seed_oi), 
	        path(outfile)
		
    script:
    """
    ls ${input_files} > pickles.list
    group_pickles.py -i pickles.list -o ${outfile}
    """
}