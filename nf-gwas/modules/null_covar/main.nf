process NULL_COVAR {
    
    fair true
    tag "trait=${meta.id}, chr=${meta.chr}"
    container "oras://community.wave.seqera.io/library/python:3.10.20--cdca65284f2e380d" // python version 3.10
    publishDir{"${params.outdir}/${meta.null_dir}"} //, mode: 'copy' // they are quite heavy and easy to generate again, I wouldn't keep them, it's around 250M per folder
    
    input:
    tuple val(meta), // id, num, chr, null_dir
          path(h5),
          val(phenov),
          val(covsv),
          val(cagev),
          val(grmv),
          val(damv),
          val(eff),
          val(subset)

    output:
    tuple val(meta), path("${meta.id}_chr${meta.chr}.h5")

    script:
    """
    covRunner.py \
        --input ${h5} \
        --phenos_v ${phenov} \
        --covs_v ${covsv} \
        --cage_v ${cagev} \
        --grm_v ${grmv} \
        --dam_v ${damv} \
        --analysis_type null_covars_LOCO \
        --out ./ \
        -p ${meta.id} \
        -m uni \
        -e ${eff} \
        -s ${subset} \
        -C ${meta.chr}

    cp "${meta.null_dir}/${meta.id}_chr${meta.chr}.h5" ./
    """
}