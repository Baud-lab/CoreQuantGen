include { VCF_MISSING   } from "./modules/vcf_missing/main.nf"
include { SUBSET_VCF    } from "./modules/subset_vcf/main.nf"
include { VCF_TO_PLINK  } from "./modules/vcf_to_plink/main.nf"
include { SNP_PRUNE     } from "./modules/snp_prune/main.nf"
include { PLINK_BY_CHR  } from "./modules/plink_by_chr/main.nf"
include { MAKE_GRM      } from "./modules/make_grm/main.nf"
include { MAKE_LOCO_GRM } from "./modules/make_loco_grm/main.nf"

// we have (--real-ref-alleles from vcf->plink) and --keep-allele-order(vcf/plink->plink) in all modules 
// This will preserve allele order in vcf, with A1 = alt, A2 = ref
def parseChrParam(chr_param) {

    chr_param = chr_param.toString().trim()

    // Case 1: numeric range, e.g. 1..20 -> returns: ['1', '2', ..., '20']
    if (chr_param ==~ /^\d+\.\.\d+$/) {
        def (start, end) = chr_param.split(/\.\./).collect { it as int }
        return (start..end).collect { it.toString() }
    }

    // Case 2: chr range, e.g. chr1..chr20 -> returns: ['chr1', 'chr2', ..., 'chr20']
    if (chr_param ==~ /^chr\d+\.\.chr\d+$/) {
        def nums = chr_param.replaceAll(/chr/, '').split(/\.\./).collect { it as int }
        return (nums[0]..nums[1]).collect { "chr${it}" }
    }

    // Case 3: comma-separated values preserves what you pass: 1,12 -> ['1', '12']; chr1,chr12 -> ['chr1', 'chr12']
    return chr_param
        .split(',')
        .collect { it.trim() }
        .findAll { it }
}    

workflow {
    
    if( ! params.skip_vcf ){
        // A. Input VCF
        vcf = Channel
            .fromPath("${params.input_vcf}")
            .map { file ->
                tuple([ id: file.baseName.replaceAll(/\.vcf/, '') ], file) 
            }
        vcf.view()
        // B. Samples file
        individ_file = file("${params.individ}")
    
        // 1. Retain list of samples in vcf
        VCF_MISSING(
          vcf, individ_file
        )
    
        // 2. Run subset vcf 
        // this is only if you want to keep a subset of vcf, 
        // not needed if you work with plink in general and might be comput expensive to keep big vcf + plink
        SUBSET_VCF(
            VCF_MISSING.out.keep
        )
        // view output
        SUBSET_VCF.out.view()
        // 3. Convert subsetted VCF to PLINK bed/bim/fam
        VCF_TO_PLINK(
           SUBSET_VCF.out // this has the vcf and list of individuals to keep
        )

        // // 2. Convert VCF to PLINK bed/bim/fam for selected individuals
        // VCF_TO_PLINK(
        //    VCF_MISSING.out.keep // this has the vcf and list of individuals to keep
        // )
        plink_files = VCF_TO_PLINK.out.genotypes // tuple val(meta), path("${prefix}.bed"), path("${prefix}.bim"), path("${prefix}.fam"), emit: genotypes
    }else{
      plink_files = Channel.of(
          tuple(
              [id: file(params.plink_file).getName()],
              file("${params.plink_file}.bed"),
              file("${params.plink_file}.bim"),
              file("${params.plink_file}.fam")
          )
      )
    }
    
    // C. Chromosomes
    //all_chr = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20'] 
    all_chr = parseChrParam(params.chr)
    chr = Channel.fromList(all_chr)

    // 4. Split plink by chr - will need it for other purposes - e.g. genotype check
    // Channel bed/bim/fam by chromosome
    if( ! params.skip_bychr ){
        // if ! skip.bychr
        ch_plink_chr = plink_files
            .combine(chr)
            .map { meta, bed, bim, fam, chr ->
                tuple(meta + [chr: chr], bed, bim, fam)
            }
        ch_plink_chr.view()
        
        PLINK_BY_CHR(
             ch_plink_chr //, params.geno_chr
        )
        PLINK_BY_CHR.out.plink.view()
    }

    if( !params.skip_pruning ){
        //all_chr = chr.join(' ') // using chr as defined above
        prune_opt = tuple(params.prune_window, params.prune_step, params.prune_r2)
        // 5. SNP pruning
        SNP_PRUNE(
            plink_files,
            all_chr,
            prune_opt
        )
        plink_pruned = SNP_PRUNE.out.genotypes // tuple val(meta), path(bed), path(bim), path(fam), path("${prefix}.prune.in")
        plink_pruned.view()
        
        if( ! params.skip_grm ) { // do GRM only if pruning - might want to change this
            // 6. Global GRM using all selected chromosomes
            grm_opt = tuple(params.maf, params.geno)
            
            MAKE_GRM(
                plink_pruned, 
                all_chr,
                grm_opt
            )
            MAKE_GRM.out.grm.view()
            
            ch_loco_grm = plink_pruned
                .combine(chr)
                .map { meta, bed, bim, fam, prune_in, chr ->
                    tuple(meta + [chr: chr], bed, bim, fam, prune_in)
                }
            ch_loco_grm.view()
            
            // 7. LOCO GRMs
            MAKE_LOCO_GRM(
                ch_loco_grm, 
                grm_opt
            )
            MAKE_LOCO_GRM.out.grm.view()
        }
    }
    
    //emit: 
    //  plink_files = plink_files
    //  plink_pruned = plink_pruned
}
