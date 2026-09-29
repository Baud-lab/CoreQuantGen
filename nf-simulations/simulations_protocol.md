## Protocol to run simulations
1. go to github Baud-lab/CoreQuantGen/nf-simulations - you're now in the folder to execute the pipeline

STEP YOU NEED TO DO
  - prepare the .h5 
    - **... or look at notebook**
    
  - set the initial parameters for simulations -> in `input/vc`
    - copy the draft, set the paramater for initial simulations (NB: the ones you put in there might be slightly different because of the sample variance **...**)
    - **?if starting from real params, have code to go from VC from code to start_params_file format?**
    - name of the set: this name is going to be used in the folder naming and in the pipeline, consider a unique and short name
    
  - edit the params file for input -> in `params/...`
    - NB: the cages need to put the tot number of cages
    - **...description of how output dir works?**
    
  - edit the submit_sim.sh -> to launch from CRG cluster -> look for # EDIT HERE
    - path to output and error directory - use scratch, make sure the directory exists otherwise will give error
  
  - last before submitting:
    - check input files and output directory - to limit crashes (avoid )

HOW DOES IT WORK (BRIEFLY)
  - 

DEBUG

NAVIGATE THE OUTPUT

RUN WITH PERMUTATIONS:
  - set cache_vd to false


NB: 
  - log files: they don't overwrite - PRO: keep track of each run; CON: can create a lot of them, they are not heavy but consider checking and keeping only the good ones
  - nextflow version 
  - you can launch the pipeline twice with different params and so, just make sure of these couple of things (consider that with this the .nextflow.log is the same and they are going to be overwritten)
  otherwise you can create a dir per run and launch it from there --> putting the name doesn't work when running pipelines again - will have to find another strategy
  - nextflow version- especially if debugging 
  - containers and bla
  - when containers don't work - try to clean the cache (both singularity and aptainer), pull outside of nextflow --> I had to do this when running nf interactively, I'm not sure why
  
  
```
nextflow run main.nf -params-file params_A.yaml -work-dir ./work_A -name runA
nextflow run main.nf -params-file params_B.yaml -work-dir ./work_B -name runB

# resume explicitly by name:
nextflow run main.nf -params-file params_A.yaml -work-dir ./work_A -resume runA
nextflow run main.nf -params-file params_B.yaml -work-dir ./work_B -resume runB
```


TODO FOR HELENE: 
DONE ~Remove the Rfun and put them in the code - doesn't work with github, check all in realdata and simulations~
DONE ~Need to add container for simulations (and maybe realdata as well)~
test whether sbatching the pipeline when container are not pulled it works - I had problems of memory when doing that on interactive session 