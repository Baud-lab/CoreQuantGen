process VCF_MISSING {

    tag "${meta.id}"
    publishDir("${params.outdir}/", mode: "copy", pattern: "*.txt")
    container "https://depot.galaxyproject.org/singularity/bcftools:1.21--h3a4d415_1" // bcftools 
    
    label "time_30m"

    input:
    tuple val(meta), path(vcf)
    path samples_file

    output:
    tuple val(meta), path(vcf), path("${meta.id}_2keep.txt"), emit: keep
    path("${meta.id}_missing.txt"), emit: missing
    
    shell:
    // bcftools + plink 1.9
    """
    bcftools query -l ${vcf} > vcf_samples.txt

    # -2 suppress column 2 (lines unique to FILE2); -3 suppress column 3 (lines that appear in both files)
    comm -23 <(sort ${samples_file}) <(sort vcf_samples.txt) > ${meta.id}_missing.txt 
    comm -12 <(sort ${samples_file}) <(sort vcf_samples.txt) > ${meta.id}_2keep.txt 
    
    ## comm -12 <(sort ${samples_file}) <(sort vcf_samples.txt) | awk 'BEGIN{OFS="\t"}{print \$1, \$1}' > ${meta.id}_2keep.txt 
    """

}

