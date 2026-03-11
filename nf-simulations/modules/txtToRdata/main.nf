process txtToRdata{
    publishDir("${outputVD}/${value_oi}/${seed_oi}/${suboutVD}", mode: 'copy')
    cache false // cache false so that repeat every time - this is to make sure that it is done everytime that run the analysis
    
    input:
    val(outRdata)
    tuple val(value_oi), val(seed_oi), path(est_file), path(ste_file)
	  val(corr_null)
	  val(outputVD)
	  val(suboutVD)

    output:
    tuple val(value_oi), val(seed_oi), path(outRdata)
          
    script:
    def corr = "${corr_null}".replaceAll(/,1$/, '').replaceAll(/,0$/, '')
    """
    VDest_to_rdata.R --est $est_file --ste $ste_file --out $outRdata --corr0 ${corr} 
    """
}
