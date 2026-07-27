process VCF_TO_PLINK {

    tag "${meta.id}"
    publishDir("${params.outdir}/", mode: "copy", pattern: "*.{bim,bed,fam,log,tsv}")
    //publishDir("${params.outdir}/host/genotypes/log", mode: "copy", pattern: "*.log")
    container "https://depot.galaxyproject.org/singularity/plink:1.90b6.21--h7b50bb2_6" // plink 1.9

    input:
    tuple val(meta), path(vcf), path(ind_keep)

    output:
    tuple val(meta), path("${prefix}.bed"), path("${prefix}.bim"), path("${prefix}.fam"), emit: genotypes
    // tuple val(meta), path("${meta.id}_chrALL_pos_alleles.tsv"), emit: positions
    path("${prefix}.log"), emit: log
    
    script:
    prefix = "${meta.id}_sub"
    """
    plink \
      --vcf ${vcf} \
      --make-bed \
      --set-missing-var-ids @:# \
      --real-ref-alleles \
      --keep-allele-order \
      --keep ${ind_keep}
      --out ${prefix}
    # zcat ${vcf} | grep -v '#' | cut -f 1,2,4,5 | tr ' ' '\t' > ${meta.id}_chrALL_pos_alleles.tsv
    """
}