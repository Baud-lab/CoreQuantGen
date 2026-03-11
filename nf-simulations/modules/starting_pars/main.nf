// STEP 1: set the var/covar that want to change
//         modifying param of interest (params.P) with value(s) (params.V) - look at params.yaml 
process starting_pars{

    input:
    path(or_pars)
    val(set_oi)
    val(par_oi)
    val(value_oi)

    output:
    tuple val(value_oi), path("set${set_oi}_simulated_${par_oi}_${value_oi}.csv")

    script:
    """
    set_params.py -i ${or_pars} -p ${par_oi} -v ${value_oi} -o "set${set_oi}_simulated_${par_oi}_${value_oi}.csv"
    """
    
}
