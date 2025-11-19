num_species = len(os.listdir(config["GENOMES"]))

g = config["OUT_DIR"]+"/samples/out.fa"
		
rule lastz:
    input:
        genes = g,
        genome = config["GENOMES"] + "/{sample}." + ("fa.gz" if EXTENSION[0]=="gz" else "fa")
    output:
        maf = config["OUT_DIR"] + "/alignments/{sample}.maf"
    benchmark:
        config["OUT_DIR"] + "/benchmarks/{sample}.lastz.txt"
    threads: lambda wildcards: int(48)
    params:
        align_dir = config["OUT_DIR"] + "/alignments",
        tool_dir = "/home/ang037@AD.UCSD.EDU/.conda/envs/roadies_env/ROADIES/KegAlign/scripts",
        num_gpu = config.get("NUM_GPU", 1)
    conda:
        "../envs/kegalign.yaml"
    shell:
        """
		exec > >(tee {wildcards.sample}_timing.log) 2>&1
        sample_workdir={params.align_dir}/{wildcards.sample}
        mkdir -p $sample_workdir
        cd $sample_workdir
        mkdir -p work
        cd work

        /usr/bin/time faToTwoBit <(gzip -cdfq {input.genome}) ref.2bit
        /usr/bin/time faToTwoBit <(gzip -cdfq {input.genes}) query.2bit

        cd ..

        /usr/bin/time -v python {params.tool_dir}/runner.py \
            --diagonal-partition \
            --format maf- \
            --num-cpu {threads} \
            --num-gpu {params.num_gpu} \
            --output-file data_package.tgz \
            --output-type tarball \
            --tool_directory {params.tool_dir} \
            {input.genome} {input.genes}

        /usr/bin/time -v python {params.tool_dir}/package_output.py \
            --format_selector maf \
            --tool_directory {params.tool_dir}

        /usr/bin/time -v python {params.tool_dir}/run_lastz_tarball.py \
            --input=data_package.tgz \
            --output={wildcards.sample}.maf \
            --parallel={threads}

        /usr/bin/time -v kegalign {input.genome} {input.genes} work/ \
            --num_gpu {params.num_gpu} \
            --num_threads {threads} > {wildcards.sample}_lastz-commands.txt

		/usr/bin/time -v awk '{{print $0, "--coverage=85 --continuity=85 --filter=identity:65 --ambiguous=iupac --step=1 --queryhspbest=10"}}' {wildcards.sample}_lastz-commands.txt | parallel --max-procs {threads}

        (echo "##maf version=1"; cat *.maf-) > {output.maf}

		rm -rf "$sample_workdir"
        """

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
		d = config["MAX_DUP"]
	shell:
		"python workflow/scripts/lastz2fasta.py -k {params.k} --path {params.p} --outdir {params.out} -m {params.m} --plotdir {params.plotdir} --statdir {params.statdir} -d {params.d}" 


rule pasta:
	input:
		config["OUT_DIR"]+"/genes/gene_{id}.fa"
	output:
		config["OUT_DIR"]+"/genes/gene_{id}.fa.aln"
	params:
		m=MIN_ALIGN,
		n=config["OUT_DIR"],
		max_len=int(1.5*config["LENGTH"]),
		prefix = "gene_{id}",
		suffix = "fa.aln"
	benchmark:
		config["OUT_DIR"]+"/benchmarks/{id}.pasta.txt"
	threads: lambda wildcards: int(config['num_threads'])
	shell:
		'''
		if [[ `grep -n '>' {input} | wc -l` -gt {params.m} ]] || [[ `awk 'BEGIN{{l=0;n=0;st=0}}{{if (substr($0,1,1) == ">") {{st=1}} else {{st=2}}; if(st==1) {{n+=1}} else if(st==2) {{l+=length($0)}}}} END{{if (n>0) {{print int((l+n-1)/n)}} else {{print 0}} }}' {input}` -gt {params.max_len} ]]
		then
			input_file={input}
			output_file={output}
			reference=""
			all_matched=true

			while IFS= read -r line; do
				line=$(echo "$line" | tr '[:lower:]' '[:upper:]')
  				if [[ "$line" != ">"* ]]; then
    				if [ -z "$reference" ]; then
      					reference="$line"
    				elif [ "$line" != "$reference" ]; then
       					all_matched=false
      					break
    				fi
 	 			fi
			done < "$input_file"

			if [ "$all_matched" = true ]; then
				cp "$input_file" "$output_file"
			else
				python pasta/run_pasta.py -i {input} -j {params.prefix} --alignment-suffix={params.suffix} --num-cpus {threads}
			fi
		fi
		touch {output}

		'''
		
rule filtermsa:
	input:
		config["OUT_DIR"]+"/genes/gene_{id}.fa.aln"
	output:
		config["OUT_DIR"]+"/genes/gene_{id}_filtered.fa.aln"
	params:
		m = config["FILTERFRAGMENTS"],
		n = config["MASKSITES"],
	shell:
		'''
		python pasta/run_seqtools.py -masksitesp {params.n} -filterfragmentsp {params.m} -infile {input} -outfile {output}
			
		'''