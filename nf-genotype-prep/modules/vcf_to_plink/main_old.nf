// Module for vcf to plink - per chr - for a subset of individuals in list
process VCF_TO_PLINK {

    tag "${meta.id}_chr${meta.chr}"
    container "oras://community.wave.seqera.io/library/bcftools_plink:c77c05a9fa4a6c98" // bcftools plink 1.9
    //container "https://depot.galaxyproject.org/singularity/plink2:2.00a5.12--h9948957_1" // plink 2 only

    label 'cpu_4' 
    label 'mem_24'
    label 'time_12h'

    input:
    tuple val(meta), path(vcf)
    path samples_file

    output:
    tuple val(meta), path("${meta.id}_chr${meta.chr}.bed"), path("${meta.id}_chr${meta.chr}.bim"), path("${meta.id}_chr${meta.chr}.fam")

    script:
    // bcftools + plink 1.9
    """
    bcftools view \
      --threads ${task.cpus} \
      --samples-file ${samples_file} \
      --regions ${meta.chr} \
      -Oz \
      -o subset.vcf.gz \
      ${vcf}

    plink \
      --threads ${task.cpus} \
      --vcf subset.vcf.gz \
      --make-bed \
      --out ${meta.id}_chr${meta.chr}
    """

    /* plink2
    """
    plink2 \
      --vcf ${vcf} \
      --keep ${samples_file} \
      --chr ${meta.chr} \
      --make-bed \
      --threads ${task.cpus} \
      --memory ${task.memory.toMega()} \
      --out ${meta.id}_chr${meta.chr}
    """
    */
    
}