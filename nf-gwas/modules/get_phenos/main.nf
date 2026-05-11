process GET_PHENOS {

    tag "${phenov}"
    publishDir { h5_parent }, mode: 'copy'

    input:
    tuple path(h5), val(h5_parent), val(phenov)

    output:
    path("*_list.txt")

    script:
    """
    get_phenos_fromh5.R ${h5} ${phenov}
    """
}