process run_VD{
  cache params.CACHE_VD
  fair true // goes in order
  tag{ "pheno_pair:${meta.row};value:${meta.value};seed:${meta.seed}" }
  container "tonnelene/py_corequantgen:1.1"
  
  memory {
    (params.vdMODEL == 'bi' ? 4.GB : 3.GB)
  }

  time {
    def base = [uni: 1.h, bi: params.bitime, sex: 2.h][params.vdMODEL] ?: 1.h
    (base * task.attempt)
  }
  
  maxRetries 1
  errorStrategy {
    task.attempt < 2 ? 'retry' : 'ignore'
  }

  input:
  tuple val(meta), path(sim_h5), path(combins) // meta: [id, value, seed, row]; row: phenotype column (uni/sex) or line of combins file (bi); combins: [] in uni/sex
  val(vd)                                      // VD settings, the same for all tasks: pheno, covs, cage, dam, grm, sex, subset, model, effects, corr0, perm
  val(dirstr)

  output:
  tuple val(meta), path("*_est.txt"), emit: est
  tuple val(meta), path("*_STE.txt"), emit: STE
  tuple val(meta), path("*_VC.pkl.gz"), emit: pkl

  script:
  def combins_arg = combins ? "${combins}" : "None"

  """
  covRunner.py \
        --input ${sim_h5} \
        --phenos_v ${vd.pheno} \
        --covs_v ${vd.covs} \
        --cage_v ${vd.cage} \
        --dam_v ${vd.dam} \
        --grm_v ${vd.grm} \
        --sex_v ${vd.sex} \
        --analysis_type VD \
        --out ./ \
        -p ${meta.row} \
        -c ${combins_arg} \
        -s ${vd.subset} \
        -m ${vd.model} \
        -e ${vd.effects} \
        -z ${vd.corr0} \
        -P ${vd.perm}
        
  cp ${dirstr}*_est.txt .
  cp ${dirstr}*_STE.txt .
  cp ${dirstr}*_VC.pkl.gz .
  """
}
