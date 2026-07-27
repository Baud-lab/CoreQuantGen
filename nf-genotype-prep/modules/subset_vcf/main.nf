process SUBSET_VCF {

    tag "${meta.id}"
    publishDir("${params.outdir}/", mode: "copy")
    container "https://depot.galaxyproject.org/singularity/bcftools:1.21--h3a4d415_1" // bcftools 
    
    label 'time_6h'
    label 'cpu_2' 

    input:
    tuple val(meta), path(vcf), path(samples_file)

    output:
    tuple val(meta), path("${prefix}.vcf.gz"), path("${prefix}.vcf.gz.tbi")

    script:
    // bcftools
    prefix = "${meta.id}_sub" 
    """
    bcftools view \
      --threads ${task.cpus} \
      --samples-file ${samples_file} \
      -Oz \
      -o ${prefix}.vcf.gz \
      ${vcf}

    tabix -p vcf ${prefix}.vcf.gz
    """

}
