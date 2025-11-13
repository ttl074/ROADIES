import glob
from collections import OrderedDict
import random,os
from pathlib import Path
import subprocess

num_species = len(os.listdir(config["GENOMES"]))
num_genomes = len(SAMPLES)

g = config["REF_DIR"]+"/samples/out.fa"
		
rule lastz:
	input:
		genes = g,
		genome = config["GENOMES"] + "/{sample}." + ("fa.gz" if EXTENSION[0]=="gz" else "fa")
	output:
		config["OUT_DIR"]+"/alignments/{sample}.maf"
	benchmark:
		config["OUT_DIR"]+"/benchmarks/{sample}.lastz.txt"
	threads: lambda wildcards: int(config['num_threads'])
	params:
		species = "{sample}",
		identity = config['IDENTITY'],
		identity_deep = config['IDENTITY_DEEP'],
		coverage = config['COVERAGE'],
		continuity = config['CONTINUITY'],
		align_dir = config['OUT_DIR']+ "/alignments",
		max_dup = 2*int(config['MAX_DUP']),
		steps = config["STEPS"],
		deep_mode = str(deep_mode),
		scores = config['SCORES']
	shell:
		'''
		if [[ "{input.genome}" == *.gz ]]; then
			if [[ "{params.deep_mode}" == "True" ]]; then
				lastz_40 <(gunzip -dc {input.genome})[multiple] {input.genes} --coverage={params.coverage} --continuity={params.continuity} --filter=identity:{params.identity_deep} --format=maf --output={output} --ambiguous=iupac --step={params.steps} --queryhspbest={params.max_dup} --scores={params.scores}
			else
				lastz_40 <(gunzip -dc {input.genome})[multiple] {input.genes} --coverage={params.coverage} --continuity={params.continuity} --filter=identity:{params.identity} --format=maf --output={output} --ambiguous=iupac --step={params.steps} --queryhspbest={params.max_dup}
			fi
		else
			if [[ "{params.deep_mode}" == "True" ]]; then
				lastz_40 {input.genome}[multiple] {input.genes}  --coverage={params.coverage} --continuity={params.continuity} --filter=identity:{params.identity_deep} --format=maf --output={output} --ambiguous=iupac --step={params.steps} --queryhspbest={params.max_dup} --scores={params.scores}
			else
				lastz_40 {input.genome}[multiple] {input.genes}  --coverage={params.coverage} --continuity={params.continuity} --filter=identity:{params.identity} --format=maf --output={output} --ambiguous=iupac --step={params.steps} --queryhspbest={params.max_dup} 
			fi
		fi
		'''

rule lastz2fasta:
	input:
		expand(config["OUT_DIR"]+"/alignments/{sample}.maf",sample=SAMPLES)   
	output:
		expand(config["OUT_DIR"]+"/genes/gene_{id}.fa",id=IDS),
		report(config["OUT_DIR"]+"/plots/num_genes.png",caption="../report/num_genes_p.rst",category="Genes Report"),
		report(config["OUT_DIR"]+"/statistics/homologs.csv",caption="../report/homologs.rst",category="Genes Report"),
		report(config["OUT_DIR"]+"/statistics/num_genes.csv",caption="../report/num_genes_t.rst",category="Genes Report"),
		report(config["OUT_DIR"]+"/statistics/num_gt.txt",caption="../report/num_gt.rst",category="Genes Report"),
		report(config["OUT_DIR"]+"/plots/gene_dup.png",caption="../report/gene_dup.rst",category="Genes Report"),
		report(config["OUT_DIR"]+"/plots/homologs.png",caption="../report/homologs_p.rst",category="Genes Report")
	params:
		k = num,
		out = config["OUT_DIR"]+"/genes",
		p = config["OUT_DIR"]+"/alignments",
		m = MIN_ALIGN,
		plotdir = config["OUT_DIR"]+"/plots",
		statdir = config["OUT_DIR"]+"/statistics",
		d = config["MAX_DUP"],
		mode = mode
	shell:
		"python workflow/scripts/lastz2fasta.py -k {params.k} --path {params.p} --outdir {params.out} -m {params.m} --plotdir {params.plotdir} --statdir {params.statdir} -d {params.d} --tool {params.mode}" 

rule pasta:
	input:
		input_sequence = config["OUT_DIR"]+"/genes/gene_{id}.fa",
	output:
		gene_tree = config["OUT_DIR"]+"/genes/gene_{id}.fa.aln.raxml.bestTree"
	params:
		m=MIN_ALIGN,
		n=config["OUT_DIR"],
		max_len=int(1.5*config["LENGTH"]),
		prefix = "gene_{id}",
		suffix = "fa.aln",
		outdir = config["OUT_DIR"]+"/genes",
		workdir = config["OUT_DIR"]+"/genes/gene_{id}",
		msa = config["OUT_DIR"]+"/genes/gene_{id}.fa.aln",
		ref_msa = config["REF_DIR"]+"/genes/gene_{id}_filtered.fa.aln",
        ref_gene_tree = config["REF_DIR"]+"/genes/gene_{id}_filtered.fa.aln.raxml.bestTree",
        ref_model = config["REF_DIR"]+"/genes/gene_{id}_filtered.fa.aln.raxml.bestModel",
		ref_sequences = config["REF_DIR"]+"/genes/gene_{id}.fa"
	benchmark:
		config["OUT_DIR"]+"/benchmarks/{id}.pasta.txt"
	threads: lambda wildcards: int(8)
	shell:
		'''
		if [[ `grep -n '>' {input.input_sequence} | wc -l` -gt 0 ]]
		then
			if [[ -s {params.ref_gene_tree} ]]
			then
				./workflow/scripts/placement.sh {input.input_sequence} {threads} {params.workdir} {params.ref_msa} {params.ref_gene_tree} {params.ref_model} {params.msa} {output.gene_tree} {params.ref_sequences}
				
			else
				cp {params.ref_gene_tree} {output.gene_tree}
			fi
		else
			cp {params.ref_gene_tree} {output.gene_tree}
		fi
		'''

rule mergeTrees:
	input:
		expand(config["OUT_DIR"]+"/genes/gene_{id}.fa.aln.raxml.bestTree",id=IDS)
	output:
		original_list=config["OUT_DIR"]+"/genetrees/original_list.txt",
		merged_list=config["OUT_DIR"]+"/genetrees/gene_tree_merged.nwk"
	params:
		msa_dir = config["OUT_DIR"]+"/genes",
		plotdir = config["OUT_DIR"]+"/plots",
		statdir = config["OUT_DIR"]+"/statistics"
	shell:
		'''
		for file in {params.msa_dir}/*.fa.aln.raxml.bestTree; do
            id=$(echo $file | sed 's/.*gene_//;s/.fa.aln.raxml.bestTree//')
            cat $file >> {output.merged_list}
            echo "$id, $(cat $file)" >> {output.original_list}
        done
		'''

