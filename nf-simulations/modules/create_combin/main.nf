process create_combin{
  container "tonnelene/py_corequantgen:1.1" // only needs bash; same image as run_VD
  //publishDir("$outputSim/${value_oi}/", mode: 'copy')
  
  input:
  val(phenos)
  
  output:
  path("combins_${phenos}phenos.txt")
  
  script:
  """
  create_combin_file.sh 1 $phenos > combins_${phenos}phenos.txt
  """
}
