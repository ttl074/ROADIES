num_species = len(os.listdir(config["GENOMES"]))

if (mode == "placement"):
	g = config["REF_DIR"]+"/samples/out.fa"
else:
	g = config["OUT_DIR"]+"/samples/out.fa"

rule kegalign:
    input:
        genes = g,
        genome = config["GENOMES"] + "/{sample}." + ("fa.gz" if EXTENSION[0]=="gz" else "fa")
    output:
        maf = config["OUT_DIR"] + "/alignments/{sample}.maf"
    benchmark:
        config["OUT_DIR"] + "/benchmarks/{sample}.kegalign.txt"
    threads: lambda wildcards: int(config["num_threads"])
    params:
        species = "{sample}",
        identity = config["IDENTITY"],
        identity_deep = config["IDENTITY_DEEP"],
        coverage = config["COVERAGE"],
        continuity = config["CONTINUITY"],
        steps = config["STEPS"],
        max_dup = 2 * int(config["MAX_DUP"]),
        deep_mode = str(deep_mode),
        scores = config["SCORES"],
        align_dir = config["OUT_DIR"] + "/alignments",
        tool_dir = "/home/ang037@AD.UCSD.EDU/.conda/envs/roadies_env/ROADIES/KegAlign/scripts",
        num_gpu = num_gpus

    conda:
        "../envs/kegalign.yaml"

    shell:
        r"""
        exec > >(tee {wildcards.sample}_timing.log) 2>&1

        sample_workdir={params.align_dir}/{wildcards.sample}
        mkdir -p $sample_workdir/work
        cd $sample_workdir/work

        # Convert fasta to 2bit
        /usr/bin/time faToTwoBit <(gzip -cdfq {input.genome}) ref.2bit
        /usr/bin/time faToTwoBit <(gzip -cdfq {input.genes}) query.2bit

        cd ..

        # Package inputs
        /usr/bin/time -v python {params.tool_dir}/runner.py \
            --diagonal-partition \
            --format maf- \
            --num-cpu {threads} \
            --num-gpu {params.num_gpu} \
            --output-file data_package.tgz \
            --output-type tarball \
            --tool_directory {params.tool_dir} \
            {input.genome} {input.genes}

        # Extract results
        /usr/bin/time -v python {params.tool_dir}/package_output.py \
            --format_selector maf \
            --tool_directory {params.tool_dir}

        /usr/bin/time -v python {params.tool_dir}/run_lastz_tarball.py \
            --input=data_package.tgz \
            --output={wildcards.sample}.maf \
            --parallel={threads}

        # Generate lastz command list (GPU accelerated)
        /usr/bin/time -v kegalign {input.genome} {input.genes} work/ \
            --num_gpu {params.num_gpu} \
            --num_threads {threads} > {wildcards.sample}_lastz-commands.txt

        # Inject biological thresholds (mirror LASTZ behavior)
        awk '{{
            sub(/ 2> /,
                " --coverage={params.coverage} --continuity={params.continuity} "
                "--filter=identity:{params.identity} "
                "--ambiguous=iupac --step={params.steps} "
                "--queryhspbest={params.max_dup} "
                "--scores={params.scores} 2> ");
            print
        }}' {wildcards.sample}_lastz-commands.txt > {wildcards.sample}_lastz-commands.final.sh

        chmod +x {wildcards.sample}_lastz-commands.final.sh

        /usr/bin/time -v parallel --max-procs {threads} < {wildcards.sample}_lastz-commands.final.sh

        (echo "##maf version=1"; cat *.maf-) > {output.maf}
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
		d = config["MAX_DUP"],
        mode = mode,
        num_gpus = num_gpus
	shell:
		"python workflow/scripts/lastz2fasta.py -k {params.k} --path {params.p} --outdir {params.out} -m {params.m} --plotdir {params.plotdir} --statdir {params.statdir} -d {params.d} --tool {params.mode} --gpu {params.num_gpus}" 
