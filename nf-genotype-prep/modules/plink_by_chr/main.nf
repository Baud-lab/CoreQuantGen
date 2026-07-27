process PLINK_BY_CHR {
    cache false // for if need to rerun with resume 
    
    tag "${meta.id} ${meta.chr}"
    publishDir("${params.outdir}/bychr/unpruned/", mode: "copy", pattern: "*.{bed,bim,fam,log}")
    publishDir("${params.outdir}/positions/", mode: "copy", pattern: "*.{tsv}")
    //publishDir("${params.outdir}/host/genotypes/log", mode: "copy", pattern: "*.log")
    container "https://depot.galaxyproject.org/singularity/plink:1.90b6.21--h7b50bb2_6" // plink 1.9
    
    label 'cpu_2'
    
    input:
    tuple val(meta), path(bed), path(bim), path(fam)
    //val(geno)
    // prune_in will not be used - leave unpruned for GWAS
    // remove snps with missing data
    
    output:
    tuple val(meta), path("${prefix}.bed"), path("${prefix}.bim"), path("${prefix}.fam"), emit: plink
    tuple val(meta), path("*_pos_alleles.tsv"), emit: positions
    path("${prefix}.log"), emit: log
    
    script:
    bfile = "${bed.simpleName}"
    prefix = "${meta.id}_${meta.chr}" //_geno${geno}"
    """
    cut -f 1,4,5,6 ${bfile}.bim > ${meta.id}_chrALL_pos_alleles.tsv
    
    plink \
      --threads ${task.cpus} \
      --bfile ${bfile} \
      --chr ${meta.chr} \
      --keep-allele-order \
      --make-bed \
      --out ${prefix}
    
    cut -f 1,4,5,6 ${prefix}.bim > ${meta.chr}_pos_alleles.tsv
    """
    //--geno ${geno} \
}