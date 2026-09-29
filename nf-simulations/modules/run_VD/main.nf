process run_VD{
  cache params.CACHE_VD
  fair true // goes in order
  tag{ "pheno_pair:$row_num;value:$value_oi;seed:$seed_oi" }
  container "docker://tonnelene/py_corequantgen:1.1"
  
  memory {
    (params.vdMODEL == 'bi' ? 4.GB : 3.GB)
  }

  time {
    def base = [uni: 1.h, bi: 10.h, sex: 2.h][params.vdMODEL] ?: 1.h
    (base * task.attempt)
  }
  
  maxRetries 1
  errorStrategy {
    task.attempt < 2 ? 'retry' : 'ignore'
  }

  input:
  tuple val(row_num), val(combins_path), val(value_oi), val(seed_oi), path(sim_h5), val(pheno), val(covs), val(cage), val(dam), val(grm), val(sexv), val(subset), val(model), val(effects), val(corr0), val(perm)
  val(dirstr)

  output:
  tuple val(value_oi), val(seed_oi), path("*_est.txt"), emit: est
  tuple val(value_oi), val(seed_oi), path("*_STE.txt"), emit: STE
  tuple val(value_oi), val(seed_oi), path("*_VC.pkl.gz"), emit: pkl

  script:
  def combins = combins_path ? "${combins_path}" : "None"

  """
  covRunner.py \
        --input ${sim_h5} \
        --phenos_v ${pheno} \
        --covs_v ${covs} \
        --cage_v ${cage} \
        --dam_v ${dam} \
        --grm_v ${grm} \
        --sex_v ${sexv} \
        --analysis_type VD \
        --out ./ \
        -p ${row_num} \
        -c ${combins} \
        -s ${subset} \
        -m ${model} \
        -e ${effects} \
        -z ${corr0} \
        -P ${perm}
        
  cp ${dirstr}*_est.txt .
  cp ${dirstr}*_STE.txt .
  cp ${dirstr}*_VC.pkl.gz .
  """
}
