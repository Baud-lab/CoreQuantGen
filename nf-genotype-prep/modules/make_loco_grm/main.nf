process MAKE_LOCO_GRM {

    tag "${meta.id}_loco_${meta.chr}"
    publishDir("${params.outdir}/host/genotypes/GRM", mode: "copy", pattern: "*.{rel,log}*")
    //publishDir("${params.outdir}/host/genotypes/log", mode: "copy", pattern: "*.log")
    container "https://depot.galaxyproject.org/singularity/plink:1.90b6.21--h7b50bb2_6" // plink 1.9
    
    label 'cpu_2'
    
    input:
    tuple val(meta), path(bed), path(bim), path(fam), path(prune_in)
    tuple val(maf), val(geno)

    output:
    tuple val(meta), path("${prefix}.rel"), path("${prefix}.rel.id"), emit: grm
    path("${prefix}.log"), emit: log

    script:
    bfile = "${bed.simpleName}"
    prefix = "${meta.id}_LOCO_${meta.chr}"

    """
    plink \
      --threads ${task.cpus} \
      --bfile ${bfile} \
      --make-rel square \
      --extract ${prune_in} \
      --not-chr ${meta.chr} \
      --maf ${maf} \
      --geno ${geno} \
      --out ${prefix}
    """
}