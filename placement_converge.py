import os
import csv
import subprocess
import time
import yaml
from pathlib import Path
import shutil

# ---------------- CONFIG ----------------
roadies_script = "run_roadies.py"
config_file = "config/config.yaml"
out_base_dir = "roadies_iterations"
support_thr = 0.95
max_iterations = 9
cores = 64  # cores for ASTRAL
roadies_dir = "roadies_final"

backbone_species = "/data/ang037/roadies-datasets/Output_folder_1000_taxa/reference_genomes"
query_species = "/data/ang037/roadies-datasets/Output_folder_1000_taxa/query_genomes"

Path(out_base_dir, roadies_dir).mkdir(parents=True, exist_ok=True)

iteration = 1
time_stamps = [time.time()]
high_support_list = []

def run_roadies(mode):
    cmd = ["python3", roadies_script, "--mode", mode, "--config", config_file, "--cores", str(cores), "--noconverge"]
    print(f"Running ROADIES: {cmd}")
    subprocess.run(cmd, check=True)

def update_gene_count(iteration, base_gene_count=100):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if iteration <= 2:
        gene_count = base_gene_count
    else:
        gene_count = base_gene_count * (2 ** (iteration - 2))

    config["GENE_COUNT"] = gene_count

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    print(f"[ITER {iteration}] GENE_COUNT set to {gene_count}")

def compute_percent_high_support(freq_file, threshold):
    count = 0
    with open(freq_file, "r") as file:
        csv_reader = csv.reader(file, delimiter="\t")
        rows = list(csv_reader)
        total_rows = len(rows)
        for i, row in enumerate(rows):
            if (i + 1) % 3 == 1:
                value = float(row[3])
                if value >= threshold:
                    count += 1
    percent_high_support = (count / (total_rows / 3)) * 100
    return percent_high_support

def update_config_yaml(out_dir=None, species=None, ref_dir=None):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if out_dir:
        config["OUT_DIR"] = out_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    if species:
        config["GENOMES"] = species
    if ref_dir:
        config["REF_DIR"] = ref_dir

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

def combine_iter(out_dir, iteration, run, cores, roadies_dir):
    master_gt = Path(out_dir) / "master_gt.nwk"
    master_map = Path(out_dir) / "master_map.txt"
    run_dir = Path(out_dir) / run

    # Append gene trees
    with open(run_dir / "genetrees/gene_tree_merged.nwk") as infile, open(master_gt, "a") as outfile:
        for line in infile:
            outfile.write(line)

    # Append mapping
    with open(run_dir / "genes/mapping_combined.txt") as infile, open(master_map, "a") as outfile:
        for line in infile:
            outfile.write(line)

    # Run ASTRAL
    astral_cmd_main = [
        "astral-pro3",
        "-t", str(cores),
        "-i", str(master_gt),
        "-o", str(run_dir / f"{run}.nwk"),
        "-a", str(master_map)
    ]
    astral_cmd_stats = [
        "astral-pro3",
        "-t", str(cores),
        "-u", "3",
        "-i", str(master_gt),
        "-o", str(run_dir / f"{run}_stats.nwk"),
        "-a", str(master_map)
    ]
    subprocess.run(astral_cmd_main, check=True)
    subprocess.run(astral_cmd_stats, check=True)

    # Copy to roadies_dir
    shutil.copy(run_dir / f"{run}.nwk", Path(out_base_dir) / roadies_dir / "roadies.nwk")
    shutil.copy(run_dir / f"{run}_stats.nwk", Path(out_base_dir) / roadies_dir / "roadies_stats.nwk")


# ---------------- ITERATIVE PIPELINE ----------------
while iteration <= max_iterations:
    print(f"\n=== ITERATION {iteration} ===")

    update_gene_count(iteration, base_gene_count=100)
    
    # 1. Backbone run (non-converge)
    backbone_out_dir = f"{out_base_dir}/iter_{iteration}_backbone"
    update_config_yaml(out_dir=backbone_out_dir, species=backbone_species, ref_dir=None)
    run_roadies(mode="accurate")
    
    # 2. Placement run (query species)
    placement_out_dir = f"{out_base_dir}/iter_{iteration}_placement"
    update_config_yaml(out_dir=placement_out_dir, species=query_species, ref_dir=backbone_out_dir)
    run_roadies(mode="placement")

    # 3. Combine gene trees + run ASTRAL from 2nd iteration onwards
    combine_iter(out_base_dir, iteration, f"iter_{iteration}_placement", cores, roadies_dir)

    # 4. Compute percent high support
    freq_file = "freqQuad.csv"
    percent_high_support = compute_percent_high_support(freq_file, support_thr)
    print(f"Iteration {iteration}: Percent high support = {percent_high_support:.2f}%")
    
    # 5. Save time stamps and high support
    curr_time = time.time()
    curr_time_l = time.asctime(time.localtime(curr_time))
    elapsed_time = curr_time - time_stamps[0]
    time_stamps.append(curr_time)
    high_support_list.append(percent_high_support)
    
    with open(Path(out_base_dir) / "time_stamps.csv", "a") as t_out:
        t_out.write(f"{iteration},{percent_high_support:.2f},{curr_time_l},{elapsed_time:.2f}\n")
    
    # 6. Check convergence
    if ((iteration == 1 and percent_high_support == 100) or
        (iteration >= 2 and
         (abs(percent_high_support - high_support_list[iteration - 2]) < 1
          or percent_high_support == 100
          or iteration == max_iterations))):
        print("Convergence criteria met. Stopping iterations.")
        break
    
    iteration += 1
