#!/usr/bin/env python3
# converge.py
# Iterative ROADIES convergence controller with PLACEMENT mode support

# REQUIREMENTS: Activated conda environment with snakemake and ete3

import os, sys, glob
import argparse
import random
import subprocess
import signal
from ete3 import Tree
from reroot import rerootTree
import yaml
from pathlib import Path
import time
import math
import csv
import shutil


############################################
# Tree utilities
############################################

def comp_tree(t1, t2):
    d = t1.compare(t2)
    return d["norm_rf"]


############################################
# Config helpers
############################################

def update_config(config_path, base_gene_count):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["GENE_COUNT"] = base_gene_count * 2

    with open(config_path, "w") as file:
        yaml.dump(config, file)


def read_initial_gene_count(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["GENE_COUNT"]


############################################
# Snakemake runner
############################################

def run_snakemake(cores, mode, out_dir, run, roadies_dir, config_path,
                  fixed_parallel_instances, deep_mode, MIN_ALIGN, num_gpus,
                  extra_cfg=None):

    num_threads = cores // fixed_parallel_instances

    cmd = [
        "snakemake",
        "--cores", str(cores),
        "--config",
        "mode=" + str(mode),
        "config_path=" + str(config_path),
        "num_threads=" + str(num_threads),
        "deep_mode=" + str(deep_mode),
        "MIN_ALIGN=" + str(MIN_ALIGN),
        "num_gpus=" + str(num_gpus),
        "--use-conda",
        "--rerun-incomplete",
    ]

    if extra_cfg:
        for k, v in extra_cfg.items():
            cmd.append(f"{k}={v}")

    print("[CONVERGE] CMD:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # archive run output into OUT_DIR/run
    os.system(f"./workflow/scripts/get_run.sh {out_dir} {run} {roadies_dir}")

############################################
# Combine trees
############################################

def combine_iter_accurate(out_dir, run, cores, roadies_dir, ref_path):

    os.system(
        f"astral-pro3 -t {cores} -i {out_dir}/{run}/gene_tree_merged.nwk -o {out_dir}/{run}.nwk -a {out_dir}/{run}/genes/mapping.txt"
    )


def combine_iter(out_dir, run, cores, roadies_dir, ref_path):

    os.system(f"cat {out_dir}/{run}/gene_tree_merged.nwk >> {out_dir}/master_gt.nwk")
    os.system(f"cp {out_dir}/master_gt.nwk {out_dir}/{run}.gt.nwk")
    os.system(f"cat {out_dir}/{run}/genes/mapping.txt {ref_path}/genes/mapping.txt >> {out_dir}/{run}/genes/mapping_combined.txt")
    os.system(f"cat {out_dir}/{run}/genes/mapping_combined.txt >> {out_dir}/master_map.txt")
    os.system(f"cp {out_dir}/master_map.txt {out_dir}/{run}.map.txt")

    os.system(
        f"astral-pro3 -t {cores} -i {out_dir}/master_gt.nwk -o {out_dir}/{run}.nwk -a {out_dir}/master_map.txt"
    )
    os.system(
        f"astral-pro3 -t {cores} -u 3 -i {out_dir}/master_gt.nwk -o {out_dir}/{run}_stats.nwk -a {out_dir}/master_map.txt"
    )

    os.system(f"cp {out_dir}/{run}.nwk {roadies_dir}/roadies.nwk")
    os.system(f"cp {out_dir}/{run}_stats.nwk {roadies_dir}/roadies_stats.nwk")

    with open(out_dir + "/master_gt.nwk") as f:
        gene_trees = f.readlines()
    return gene_trees


############################################
# Archive ROADIES_DIR into OUT_DIR with tag
############################################

def archive_iteration(roadies_dir, out_dir, tag):
    dst = os.path.join(out_dir, tag)
    print(f"[CONVERGE] ARCHIVE {roadies_dir} -> {dst}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(roadies_dir, dst)
    return dst


############################################
# One ROADIES iteration (normal accurate/balanced/fast)
############################################

def converge_run(iteration, cores, mode, out_dir, ref_exist, ref, roadies_dir,
                 support_thr, config_path, fixed_parallel_instances, deep_mode,
                 MIN_ALIGN, ref_path, num_gpus, grow, extra_cfg=None):

    os.system(f"rm -rf {roadies_dir}")
    os.makedirs(roadies_dir, exist_ok=True)
    os.system("rm -f sampling_output.txt")

    run = "iteration_{iteration:02d}"

    if iteration >= 2:
        base_gene_count = read_initial_gene_count(config_path)
        update_config(config_path, base_gene_count)

    run_snakemake(
        cores, "accurate", out_dir, "iteration_{iteration:02d}_accurate", roadies_dir, config_path,
        fixed_parallel_instances, deep_mode, MIN_ALIGN, num_gpus, extra_cfg
    )

    combine_iter_accurate(out_dir, "iteration_{iteration:02d}_accurate", cores, roadies_dir, mode, ref_path)

    run_snakemake(
        cores, "placement", out_dir, "iteration_{iteration:02d}_placement", roadies_dir, config_path,
        fixed_parallel_instances, deep_mode, MIN_ALIGN, num_gpus, extra_cfg
    )

    ref_path = out_dir + "iteration_{iteration:02d}_accurate"

    gene_trees = combine_iter(out_dir, "iteration_{iteration:02d}_placement", cores, roadies_dir, mode, ref_path)
    t = Tree(out_dir + f"/{run}.nwk")

    if ref_exist:
        reroottree = t
        rerootTree(ref, reroottree)
        reroottree.write(outfile=out_dir + f"/{run}.rerooted.nwk")

    # parse support
    local_pp_values = []
    count = 0
    with open("freqQuad.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter="\t")
        rows = list(csv_reader)
        total_rows = len(rows)
        for i, row in enumerate(rows):
            if (i + 1) % 3 == 1:
                value = float(row[3])
                if value >= support_thr:
                    count += 1
                local_pp_values.append(value)

    percent_high_support = (count / (total_rows / 3)) * 100
    return percent_high_support, len(gene_trees), t, run


############################################
# MAIN
############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=32)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--mode", default="accurate")
    parser.add_argument("--deep", default="False")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--grow", action="store_true")

    args = vars(parser.parse_args())

    config_path = args["config"]
    CORES = args["cores"]
    MODE = args["mode"]
    deep_mode = args["deep"]
    num_gpus = args["gpu"]
    grow = args["grow"]

    config = yaml.safe_load(Path(config_path).read_text())

    ref_exist = False
    ref = None
    if config["REFERENCE"] is not None:
        ref_exist = True
        ref = Tree(config["REFERENCE"])

    genomes = config["GENOMES"]
    out_dir = config["ALL_OUT_DIR"]   # global archive
    roadies_dir = config["OUT_DIR"]   # working dir

    NUM_GENOMES = len(os.listdir(genomes))
    MIN_ALIGN = max(4, math.ceil(0.1 * NUM_GENOMES))
    support_thr = config["SUPPORT_THRESHOLD"]
    fixed_parallel_instances = config["NUM_INSTANCES"]
    ref_path = config["REF_DIR"]

    master_gt = out_dir + "/master_gt.nwk"
    master_map = out_dir + "/master_map.txt"

    os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    open(master_gt, "w").close()
    open(master_map, "w").close()

    os.system("snakemake --unlock")

    time_stamps = []
    high_support_list = []
    if ref_exist:
        ref_dists = []

    iteration = 0
    start_time = time.time()
    with open(out_dir + "/time_stamps.csv", "w") as f:
        f.write("Start time," + time.asctime() + "\n")

    ########################################
    # ITERATION LOOP
    ########################################
    while True:

        print(f"\n=========== ITER {iteration:02d} MODE={MODE} ===========")

        # Stage 1: Accurate
        percent_high_support, num_gt, outputtree, run = converge_run(
            iteration, CORES, "accurate", out_dir, ref_exist, ref, roadies_dir,
            support_thr, config_path, fixed_parallel_instances, deep_mode,
            MIN_ALIGN, ref_path, num_gpus, grow
        )
        
        ########################################
        # Logging & convergence logic (unchanged)
        ########################################
        curr_time = time.time()
        high_support_list.append(percent_high_support)
        elapsed_time = curr_time - start_time

        with open(out_dir + "/time_stamps.csv", "a") as f:
            f.write(f"{iteration},{num_gt},{percent_high_support},{time.asctime()},{elapsed_time}\n")

        if ref_exist:
            ref_dist = comp_tree(ref, outputtree)
            with open(out_dir + "/ref_dist.csv", "a") as f:
                f.write(f"{iteration},{num_gt},{ref_dist}\n")

        iteration += 1

        # original stopping rule
        if ((iteration == 1) and (percent_high_support == 100)) or (
            (iteration >= 2) and (
                (abs(percent_high_support - high_support_list[iteration - 2]) < 1) or
                (percent_high_support == 100) or
                (iteration == 9)
            )
        ):
            break

    print("\n[CONVERGE] FINISHED")
