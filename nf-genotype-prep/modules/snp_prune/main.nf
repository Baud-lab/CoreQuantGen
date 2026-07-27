process SNP_PRUNE {

    tag "${meta.id}"
    publishDir("${params.outdir}/host/genotypes", mode: "copy", pattern: "*.{prune.in,log}")
    //publishDir("${params.outdir}/host/genotypes/log", mode: "copy", pattern: "*.log")
    container "https://depot.galaxyproject.org/singularity/plink:1.90b6.21--h7b50bb2_6" // plink 1.9

    label 'time_30m'

    input:
    tuple val(meta), path(bed), path(bim), path(fam)
    val(chrs)
    tuple val(prune_window), val(prune_step), val(prune_r2)    
    //${params.chromosomes.join(' ')}
    
    output:
    tuple val(meta), path(bed), path(bim), path(fam), path("${prefix}.prune.in"), emit: genotypes
    path("${prefix}.log"), emit: log

    script:
    bfile = "${bed.simpleName}"
    prefix = "${meta.id}_pruned_${prune_window}_${prune_step}_${prune_r2}"
    """
    plink \
      --bfile ${bfile} \
      --keep-allele-order \
      --chr ${chrs.join(' ')} \
      --indep ${prune_window} ${prune_step} ${prune_r2} \
      --out ${prefix}
    """
}