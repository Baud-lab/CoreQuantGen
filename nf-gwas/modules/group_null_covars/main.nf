process GROUP_NULL_COVARS {
    container "tonnelene/py_corequantgen:1.1" // only needs bash and coreutils
    
    cache false // repeat every time
    tag "trait=${meta.id}"

    input:
    tuple val(meta), path(null_covars)

    output:
    tuple val(meta), path("${meta.id}/"), emit: dir
    path("${meta.id}/*chr*.h5"), emit: files // this is if the copy didn't work, it raises error

    script:
    """
    set -euo pipefail

    mkdir -p ${meta.id}

    cp ${null_covars} ${meta.id}/

    echo "Files gathered:"
    ls -lh ${meta.id}/
    """
}