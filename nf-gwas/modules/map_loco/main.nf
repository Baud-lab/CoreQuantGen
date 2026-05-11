process MAP_LOCO {

    tag "trait=${meta.id}"
    container "oras://community.wave.seqera.io/library/python:3.10.20--cdca65284f2e380d" // python version 3.10
    publishDir{"${params.outdir}/${meta.pval_dir}"}, mode: 'copy'

    label 'mem_40'
    label 'time_1h'
    
    input:
    tuple val(meta), path(null_covs), val(chrs_list), path(h5), val(phenov), val(covsv), val(cagev), val(grmv), val(damv), val(dsnpv), val(ssnpv), val(eff), val(gwas), val(subset)

    output:
    tuple val(meta), path("${meta.id}.h5")
      
    script:
    """
    mkdir -p ${meta.null_dir}
    mv -iv -t ${meta.null_dir} ${null_covs} 
    
    map_LOCO_noMT.py \
        --input ${h5} \
        --phenos_v ${phenov} \
        --covs_v ${covsv} \
        --cage_v ${cagev} \
        --grm_v ${grmv} \
        --dam_v ${damv} \
        --directSNP_v ${dsnpv} \
        --socialSNP_v ${ssnpv} \
        --covarDir null_covars_LOCO/ \
        --out ./ \
        -p ${meta.id} \
        -m uni \
        -e ${eff} \
        -g ${gwas} \
        -s ${subset} \
        -C ${chrs_list}
        
    cp ${meta.pval_dir}/${meta.id}.h5 ./
    """
}