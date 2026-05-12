#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <cuda_runtime.h>

#include "io/input_validation.hpp"
#include "io/jplace.hpp"
#include "util/precision.hpp"
#include "spr/local_spr.hpp"
#include "model_utils.hpp"
#include "io/tree_newick.hpp"
#include "likelihood/root_likelihood.cuh"
#include "tree/tree.hpp"
#include "placement/placement.cuh"
#include "io/parse_file.hpp"
#include "msa_preprocess.hpp"
#include "util/mlipper_util.h"

namespace {

// ----- Batch commit helpers -----

struct BatchCommitTimingStats {
    double free_prev_ms = 0.0;
    double build_ms = 0.0;
    double initial_update_ms = 0.0;
    double evaluate_ms = 0.0;
    double append_ms = 0.0;
    double newick_ms = 0.0;
    int batches = 0;
    int queries = 0;
    CommitTimingStats commit{};
};

using BatchCommitClock = std::chrono::steady_clock;

double batch_commit_elapsed_ms(const BatchCommitClock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        BatchCommitClock::now() - start).count();
}

void accumulate_commit_timing(CommitTimingStats& dst, const CommitTimingStats& src) {
    dst.initial_upward_host_ms += src.initial_upward_host_ms;
    dst.initial_downward_host_ms += src.initial_downward_host_ms;
    dst.initial_upward_stage_ms += src.initial_upward_stage_ms;
    dst.initial_downward_stage_ms += src.initial_downward_stage_ms;
    dst.query_reset_stage_ms += src.query_reset_stage_ms;
    dst.query_build_clv_stage_ms += src.query_build_clv_stage_ms;
    dst.query_kernel_total_ms += src.query_kernel_total_ms;
    dst.insertion_pre_clv_ms += src.insertion_pre_clv_ms;
    dst.insertion_upward_host_ms += src.insertion_upward_host_ms;
    dst.insertion_downward_host_ms += src.insertion_downward_host_ms;
    dst.insertion_upward_stage_ms += src.insertion_upward_stage_ms;
    dst.insertion_downward_stage_ms += src.insertion_downward_stage_ms;
    dst.initial_upward_ops += src.initial_upward_ops;
    dst.initial_downward_ops += src.initial_downward_ops;
    dst.insertion_upward_ops += src.insertion_upward_ops;
    dst.insertion_downward_ops += src.insertion_downward_ops;
    dst.initial_updates += src.initial_updates;
    dst.query_evals += src.query_evals;
    dst.insertion_updates += src.insertion_updates;
}

} // namespace

namespace mlenv = mlipper::env;
namespace mlinput = mlipper::input;
namespace mljplace = mlipper::jplaceio;
namespace mlmodel = mlipper::model;
namespace mltreeio = mlipper::treeio;

int main(int argc, char** argv) {
    auto start_gpu = std::chrono::steady_clock::time_point{};
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;
    float gpu_ms_kernel = 0.0f;
    BuildToGpuResult res{};
    PlacementOpBuffer placement_ops{};

    CLI::App app{"MLIPPER"};
    app.get_formatter()->column_width(40);

    // Single config object filled directly by CLI11 options.
    parse::RunConfig config;

    // ---- Input (files/tree) ----
    std::string tree_newick;
    std::string jplace_out;
    std::string commit_tree_out;
    // Internal-only output/control knobs, not exposed via CLI.
    double commit_collapse_internal_epsilon = 1e-6;
    bool commit_to_tree = false;

    auto* opt_tree_alignment = app.add_option("--tree-alignment", config.files.tree_alignment, "Reference alignment (tree MSA)")
                                  ->group("Input")
                                  ->check(CLI::ExistingFile);
    auto* opt_query_alignment = app.add_option("--query-alignment", config.files.query_alignment,
                                               "Query alignment for placement (optional; defaults to --tree-alignment)")
                                   ->group("Input")
                                   ->check(CLI::ExistingFile);
    auto* opt_tree_file = app.add_option("--tree", config.files.tree, "Reference tree topology (Newick file)")
                              ->group("Input")
                              ->check(CLI::ExistingFile);
    auto* opt_tree_newick = app.add_option("--tree-newick", tree_newick, "Reference tree topology (Newick string)")
                                ->group("Input");
    auto* opt_jplace_out =
        app.add_option("--jplace-out", jplace_out, "Optional output path for a top-k placement jplace file")
            ->group("Output");
    app.add_option("--commit-to-tree", commit_tree_out,
                   "Commit query placements to reference tree and write the final tree in Newick (.nwk) format to this path")
        ->group("Output");
    opt_tree_file->excludes(opt_tree_newick);
    opt_tree_newick->excludes(opt_tree_file);

    // ---- Model ----
    // Defaults match the previous `config.yaml` defaults (pre-CLI refactor).
    config.model.states = 4;
    config.model.subst_model = "GTR";
    config.model.ncat = 4;
    config.model.alpha = 0.3;
    config.model.pinv = 0.0;
    config.model.rates = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    config.model.per_rate_scaling = true;
    std::string best_model_file;
    bool no_per_rate_scaling = false;
    bool empirical_freqs = false;
    int full_opt_passes = -1;
    int refine_global_passes = -1;
    int refine_extra_passes = -1;
    int refine_detect_topk = -1;
    int refine_topk = -1;
    double refine_gap_top2 = -1.0;
    double refine_gap_top5 = -1.0;
    double refine_converged_loglk_eps = -1.0;
    double refine_converged_length_eps = -1.0;
    bool placement_fast = false;
    bool local_spr = false;
    bool local_spr_fast = false;
    bool fast_mode = false;
    // Internal-only local SPR tuning knobs not exposed via CLI.
    int batch_insert_size = 0;
    int local_spr_radius = 2;
    int local_spr_cluster_threshold = 3;
    int local_spr_topk_per_unit = 8;
    bool local_spr_dynamic_validation_conflicts = false;
    int local_spr_rounds = 1;

    app.add_option("--states", config.model.states, "Number of states")->group("Model");
    app.add_option("--subst-model", config.model.subst_model, "Substitution model")->group("Model");
    app.add_option("--ncat", config.model.ncat, "Number of rate categories")->group("Model");
    app.add_option("--alpha", config.model.alpha, "Gamma shape alpha")->group("Model");
    app.add_option("--pinv", config.model.pinv, "Proportion of invariant sites")->group("Model");
    app.add_option(
           "--best-model",
           best_model_file,
           "Read a bestModel file and overwrite the corresponding model flags")
        ->group("Model")
        ->check(CLI::ExistingFile);
    auto* opt_freqs = app.add_option("--freqs", config.model.freqs, "Equilibrium freqs (list)")
                          ->group("Model")
                          ->delimiter(',');
    auto* opt_empirical_freqs = app.add_flag(
        "--empirical-freqs",
        empirical_freqs,
        "Estimate equilibrium freqs from --tree-alignment (distributes ambiguous DNA symbols across represented states)")
                                    ->group("Model");
    opt_freqs->excludes(opt_empirical_freqs);
    opt_empirical_freqs->excludes(opt_freqs);
    app.add_option("--rates", config.model.rates, "GTR rates rAC,rAG,rAT,rCG,rCT,rGT (list)")->group("Model")->delimiter(',');
    app.add_option("--rate-weights", config.model.rate_weights, "Rate category weights (list)")->group("Model")->delimiter(',');
    app.add_flag("--no-per-rate-scaling", no_per_rate_scaling, "Disable per-rate scaling")->group("Model");

    app.add_flag("--placement-fast", placement_fast,
                 "Use fast placement scoring (1 full optimization pass instead of baseline 4)")
        ->group("Placement");
    auto* opt_local_spr =
        app.add_flag("--local-spr", local_spr,
                     "Enable local subtree SPR after each batch insert")
            ->group("Placement");
    auto* opt_local_spr_fast =
        app.add_flag("--local-spr-fast", local_spr_fast,
                     "Use fast local SPR scoring (1 full optimization pass instead of baseline 4)")
            ->group("Placement");
    auto* opt_fast =
        app.add_flag("--fast", fast_mode,
                     "Enable both --placement-fast and --local-spr-fast")
            ->group("Placement");
    auto* opt_batch_insert_size =
        app.add_option("--batch-insert-size", batch_insert_size,
                       "Insert+commit query batches of size N (0 = all at once)")
            ->group("Placement");
    auto* opt_local_spr_radius =
        app.add_option("--local-spr-radius", local_spr_radius,
                       "Local SPR radius (filters candidate edges and defines subtree neighborhood)")
            ->group("Placement");
    auto* opt_local_spr_cluster_threshold =
        app.add_option("--local-spr-cluster-threshold", local_spr_cluster_threshold,
                       "Anchor-distance threshold for grouping inserted queries into local SPR repair units")
            ->group("Placement");
    auto* opt_local_spr_rounds =
        app.add_option("--local-spr-rounds", local_spr_rounds,
                       "Run up to N rounds of local SPR, rebuilding between accepted rounds")
            ->group("Placement");
    auto* opt_local_spr_dynamic_validation_conflicts =
        app.add_flag(
               "--local-spr-dynamic-validation-conflicts",
               local_spr_dynamic_validation_conflicts,
               "Validate local SPR candidates with dynamic conflict resolution instead of one-per-unit filtering")
            ->group("Placement");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (batch_insert_size < 0) {
        return app.exit(CLI::ValidationError("--batch-insert-size", "must be >= 0"));
    }
    if (local_spr_radius < 0) {
        return app.exit(CLI::ValidationError("--local-spr-radius", "must be >= 0"));
    }
    if (local_spr_cluster_threshold < 0) {
        return app.exit(CLI::ValidationError("--local-spr-cluster-threshold", "must be >= 0"));
    }
    if (local_spr_rounds <= 0) {
        return app.exit(CLI::ValidationError("--local-spr-rounds", "must be >= 1"));
    }
    if (fast_mode) {
        placement_fast = true;
        local_spr_fast = true;
    }
    if (local_spr) {
        if (batch_insert_size <= 0) {
            batch_insert_size = 5;
        }
    }
    commit_to_tree = !commit_tree_out.empty();

    const bool local_spr_tuning_requested =
        opt_local_spr_radius->count() > 0 ||
        opt_local_spr_cluster_threshold->count() > 0 ||
        opt_local_spr_rounds->count() > 0 ||
        opt_local_spr_fast->count() > 0;
    if (local_spr_tuning_requested && !local_spr) {
        return app.exit(CLI::ValidationError(
            "--local-spr",
            "local SPR tuning flags require --local-spr"));
    }
    if (local_spr && !commit_to_tree) {
        return app.exit(CLI::ValidationError(
            "--local-spr",
            "--local-spr requires --commit-to-tree"));
    }
    if (opt_batch_insert_size->count() > 0 && batch_insert_size > 0 && !commit_to_tree) {
        return app.exit(CLI::ValidationError(
            "--batch-insert-size",
            "batch insert mode requires --commit-to-tree"));
    }
    if (batch_insert_size > 0 && commit_to_tree && opt_jplace_out->count() > 0) {
        return app.exit(CLI::ValidationError(
            "--jplace-out",
            "batch insert mode does not support --jplace-out"));
    }
    (void)opt_local_spr;
    (void)opt_fast;

    const std::filesystem::path config_base = std::filesystem::current_path();

    parse::RunInputs inputs;
    try {
        if (!best_model_file.empty()) {
            try {
                const auto best_model = mlmodel::parse_best_model_file(
                    mlinput::resolve_path(config_base, best_model_file));
                config.model.states = best_model.model.states;
                config.model.subst_model = best_model.model.subst_model;
                config.model.ncat = best_model.model.ncat;
                config.model.alpha = best_model.model.alpha;
                // bestModel -> pinv override is intentionally disabled for now.
                config.model.freqs = best_model.model.freqs;
                config.model.rates = best_model.model.rates;
                empirical_freqs = best_model.empirical_freqs;
            } catch (const std::exception& e) {
                throw CLI::ValidationError("--best-model", e.what());
            }
        }
        if (no_per_rate_scaling) config.model.per_rate_scaling = false;
        if (config.files.tree_alignment.empty()) throw CLI::RequiredError("--tree-alignment");
        if (config.files.query_alignment.empty()) {
            config.files.query_alignment = config.files.tree_alignment;
        }
        if (tree_newick.empty() && config.files.tree.empty()) throw CLI::RequiredError("one of [--tree, --tree-newick]");

        if (!commit_tree_out.empty()) {
            mlinput::validate_output_path(config_base, "--commit-to-tree", commit_tree_out);
        }
        if (!jplace_out.empty()) {
            mlinput::validate_output_path(config_base, "--jplace-out", jplace_out);
        }
        if (!commit_tree_out.empty() && !jplace_out.empty()) {
            const std::filesystem::path commit_path =
                mlinput::normalize_cli_path(config_base, commit_tree_out);
            const std::filesystem::path jplace_path =
                mlinput::normalize_cli_path(config_base, jplace_out);
            if (commit_path == jplace_path) {
                throw CLI::ValidationError(
                    "--jplace-out",
                    "must not be the same path as --commit-to-tree");
            }
        }

        mlinput::validate_model_inputs(config.model);

        parse::Alignment tree_alignment;
        try {
            tree_alignment = parse::read_alignment_file(
                mlinput::resolve_path(config_base, config.files.tree_alignment));
        } catch (const std::exception& e) {
            throw CLI::ValidationError("--tree-alignment", e.what());
        }

        parse::Alignment query_alignment;
        try {
            query_alignment = parse::read_alignment_file(
                mlinput::resolve_path(config_base, config.files.query_alignment));
        } catch (const std::exception& e) {
            throw CLI::ValidationError("--query-alignment", e.what());
        }

        std::string tree_text;
        if (tree_newick.empty()) {
            try {
                tree_text = mlinput::read_file_to_string(
                    mlinput::resolve_path(config_base, config.files.tree));
            } catch (const std::exception& e) {
                throw CLI::ValidationError("--tree", e.what());
            }
        } else {
            tree_text = tree_newick;
        }
        tree_text = parse::normalize_newick(tree_text);
        mlinput::validate_newick_with_pll(
            tree_text,
            tree_newick.empty() ? "--tree" : "--tree-newick");

        inputs = parse::RunInputs{
            std::move(config),
            std::move(tree_alignment),
            std::move(query_alignment),
            std::move(tree_text)};

        if (inputs.tree_alignment.names.empty())
            throw CLI::ValidationError("--tree-alignment", "contains no sequences");
        if (inputs.tree_alignment.sites == 0)
            throw CLI::ValidationError("--tree-alignment", "contains zero sites");

        if (inputs.query_alignment.names.empty())
            throw CLI::ValidationError("--query-alignment", "contains no sequences");
        if (inputs.query_alignment.sites == 0)
            throw CLI::ValidationError("--query-alignment", "contains zero sites");
        if (inputs.query_alignment.sites != inputs.tree_alignment.sites) {
            throw CLI::ValidationError("--query-alignment",
                                       "sites mismatch with --tree-alignment (" +
                                           std::to_string(inputs.query_alignment.sites) + " vs " +
                                           std::to_string(inputs.tree_alignment.sites) + ")");
        }

        mlinput::validate_alignment_names(inputs.tree_alignment, "--tree-alignment");
        mlinput::validate_alignment_names(inputs.query_alignment, "--query-alignment");
        mlinput::validate_alignment_symbols(
            inputs.tree_alignment,
            inputs.config.model.states,
            "--tree-alignment");
        mlinput::validate_alignment_symbols(
            inputs.query_alignment,
            inputs.config.model.states,
            "--query-alignment");
        if (commit_to_tree) {
            mlinput::validate_query_reference_name_overlap(
                inputs.tree_alignment,
                inputs.query_alignment,
                "--query-alignment");
        }
    } catch (const CLI::Error& e) {
        return app.exit(e);
    }

    try {
    start_gpu = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreate(&stream));

    const auto& alignment = inputs.tree_alignment;

    const auto& msa_names = alignment.names;
    std::vector<std::string> rows = alignment.sequences;
    size_t sites = alignment.sites;
    const std::string& newick = inputs.tree;
    const auto& model = inputs.config.model;
    int states = model.states;
    int rate_cats = model.ncat;
    bool per_rate_scaling = model.per_rate_scaling;
    std::vector<unsigned> pattern_weights(sites, 1u);

    std::vector<double> pi =
        empirical_freqs ? mlmodel::estimate_empirical_pi(alignment, states)
                        : mlmodel::ensure_normalized_pi(model.freqs, states);
    std::vector<double> rate_weights = build_mixture_weights(model, rate_cats);
    std::vector<double> rate_multipliers = build_gamma_rate_categories(model.alpha, rate_cats);
    std::vector<double> Q = build_gtr_q_matrix(states, model, pi);

    std::cout << "Equilibrium frequencies ("
              << (empirical_freqs ? "empirical" : (model.freqs.empty() ? "uniform" : "manual"))
              << ") =";
    for (double value : pi) {
        std::cout << ' ' << std::fixed << std::setprecision(8) << value;
    }
    std::cout << "\n";

    std::vector<NewPlacementQuery> placement_queries =
        build_placement_query(inputs.query_alignment.names, inputs.query_alignment.sequences);
    if (repetitive_column_compression_enabled()) {
        remove_repetitive_columns(rows, placement_queries, pattern_weights, sites);
        if (sites == 0) {
            throw std::runtime_error("All columns were removed after repetitive-column compression.");
        }
    }
    const bool disable_pattern_weights =
        mlenv::env_flag_enabled("MLIPPER_DISABLE_PATTERN_WEIGHTS");
    const std::vector<unsigned> no_pattern_weights;
    const std::vector<unsigned>& pattern_weights_arg =
        disable_pattern_weights ? no_pattern_weights : pattern_weights;

    mlenv::set_int_env_if_specified("MLIPPER_FULL_OPT_PASSES", full_opt_passes);
    mlenv::set_int_env_if_specified("MLIPPER_REFINE_GLOBAL_PASSES", refine_global_passes);
    mlenv::set_int_env_if_specified("MLIPPER_REFINE_EXTRA_PASSES", refine_extra_passes);
    mlenv::set_int_env_if_specified("MLIPPER_REFINE_DETECT_TOPK", refine_detect_topk);
    mlenv::set_int_env_if_specified("MLIPPER_REFINE_TOPK", refine_topk);
    mlenv::set_double_env_if_specified("MLIPPER_REFINE_GAP_TOP2", refine_gap_top2);
    mlenv::set_double_env_if_specified("MLIPPER_REFINE_GAP_TOP5", refine_gap_top5);
    mlenv::set_double_env_if_specified(
        "MLIPPER_REFINE_CONVERGED_LOGLK_EPS",
        refine_converged_loglk_eps);
    mlenv::set_double_env_if_specified(
        "MLIPPER_REFINE_CONVERGED_LENGTH_EPS",
        refine_converged_length_eps);
    if (placement_fast) {
        if (full_opt_passes < 0) {
            setenv("MLIPPER_FULL_OPT_PASSES", "1", 1);
        }
        if (refine_global_passes < 0) {
            setenv("MLIPPER_REFINE_GLOBAL_PASSES", "0", 1);
        }
        if (refine_extra_passes < 0) {
            setenv("MLIPPER_REFINE_EXTRA_PASSES", "0", 1);
        }
        if (refine_detect_topk < 0) {
            setenv("MLIPPER_REFINE_DETECT_TOPK", "0", 1);
        }
        if (refine_topk < 0) {
            setenv("MLIPPER_REFINE_TOPK", "0", 1);
        }
    }

    printf("Precision mode: %s\n", FP_MODE_NAME);
    // const auto start_gpu = std::chrono::steady_clock::now();
    std::vector<PlacementResult> placement_results;
    std::vector<std::string> committed_query_names(placement_queries.size());

    if (commit_to_tree && batch_insert_size > 0 && !placement_queries.empty()) {
        if (!jplace_out.empty()) {
            throw std::runtime_error("Batch insert mode does not support --jplace-out.");
        }
        const bool profile_batch_timing = []() {
            const char* env = std::getenv("MLIPPER_PROFILE_COMMIT_TIMING");
            return env && std::atoi(env) != 0;
        }();
        BatchCommitTimingStats batch_commit_timing;
        cudaEventRecord(start);
        std::vector<std::string> current_names = msa_names;
        std::vector<std::string> current_rows = rows;
        std::string current_tree_newick = newick;
        const int total_queries = static_cast<int>(placement_queries.size());
        placement_results.resize(placement_queries.size());

        for (int batch_start = 0; batch_start < total_queries; batch_start += batch_insert_size) {
            const int batch_end = std::min(batch_start + batch_insert_size, total_queries);
            ++batch_commit_timing.batches;
            std::vector<NewPlacementQuery> batch_queries;
            std::vector<std::string> batch_query_names;
            std::vector<int> batch_indices;
            batch_queries.reserve(batch_end - batch_start);
            batch_query_names.reserve(batch_end - batch_start);
            batch_indices.reserve(batch_end - batch_start);
            for (int idx = batch_start; idx < batch_end; ++idx) {
                batch_queries.push_back(placement_queries[(size_t)idx]);
                batch_query_names.push_back(placement_queries[(size_t)idx].msa_name);
                batch_indices.push_back(idx);
            }
            batch_commit_timing.queries += static_cast<int>(batch_queries.size());

            if (res.dev.N != 0) {
                const auto free_prev_start = BatchCommitClock::now();
                free_placement_op_buffer(placement_ops, stream);
                cudaStreamSynchronize(stream);
                free_device_tree(res.dev);
                batch_commit_timing.free_prev_ms += batch_commit_elapsed_ms(free_prev_start);
            }
            const auto build_start = BatchCommitClock::now();
            res = BuildAllToGPU(
                current_names,
                current_rows,
                current_tree_newick,
                Q,
                pi,
                rate_multipliers,
                rate_weights,
                pattern_weights_arg,
                sites,
                states,
                rate_cats,
                per_rate_scaling,
                batch_queries,
                true);
            batch_commit_timing.build_ms += batch_commit_elapsed_ms(build_start);
            if (res.tree.nodes.empty() || res.dev.N == 0) {
                throw std::runtime_error("BuildAllToGPU returned empty tree/device structures.");
            }
            if (res.tree.root_id < 0) {
                throw std::runtime_error("BuildAllToGPU produced tree with invalid root_id.");
            }

            placement_ops = PlacementOpBuffer{};
            placement_ops.profile_commit_timing = profile_batch_timing;
            const auto initial_update_start = BatchCommitClock::now();
            UpdateTreeClvs(
                res.dev,
                res.tree,
                res.hostPack,
                placement_ops,
                stream);
            batch_commit_timing.initial_update_ms +=
                batch_commit_elapsed_ms(initial_update_start);
            std::vector<PlacementResult> batch_results;
            std::vector<std::string> inserted_names(batch_queries.size());
            PlacementCommitContext batch_ctx;
            batch_ctx.tree = &res.tree;
            batch_ctx.host = &res.hostPack;
            batch_ctx.queries = &res.queries;
            batch_ctx.placement_ops = &placement_ops;
            batch_ctx.query_names = &batch_query_names;
            batch_ctx.inserted_query_names = &inserted_names;
            const auto evaluate_start = BatchCommitClock::now();
            EvaluatePlacementQueries(
                res.dev,
                res.eig,
                rate_multipliers,
                batch_ctx,
                &batch_results,
                1,
                true,
                stream);
            batch_commit_timing.evaluate_ms += batch_commit_elapsed_ms(evaluate_start);
            if (profile_batch_timing) {
                accumulate_commit_timing(batch_commit_timing.commit, placement_ops.timing);
            }

            const auto append_start = BatchCommitClock::now();
            for (size_t i = 0; i < batch_indices.size(); ++i) {
                const int qidx = batch_indices[i];
                committed_query_names[(size_t)qidx] = inserted_names[i];
                if (i < batch_results.size()) {
                    placement_results[(size_t)qidx] = batch_results[i];
                }
                current_names.push_back(inserted_names[i]);
                current_rows.push_back(placement_queries[(size_t)qidx].msa);
            }
            batch_commit_timing.append_ms += batch_commit_elapsed_ms(append_start);

            const auto newick_start = BatchCommitClock::now();
            current_tree_newick = mltreeio::write_tree_to_newick_string(res.tree);
            batch_commit_timing.newick_ms += batch_commit_elapsed_ms(newick_start);

            if (local_spr) {
                LocalSprBatchRunContext local_spr_ctx{
                    res,
                    placement_ops,
                    stream,
                    inserted_names,
                    current_names,
                    current_rows,
                    current_tree_newick,
                    pattern_weights_arg,
                    rate_weights,
                    rate_multipliers,
                    pi,
                    sites,
                    states,
                    rate_cats,
                    per_rate_scaling,
                    profile_batch_timing,
                    local_spr_fast,
                    local_spr_radius,
                    local_spr_cluster_threshold,
                    local_spr_topk_per_unit,
                    local_spr_dynamic_validation_conflicts,
                    local_spr_rounds,
                };
                run_local_spr_batch_refinement(local_spr_ctx);
            }
        }
        if (profile_batch_timing) {
            const CommitTimingStats& stats = batch_commit_timing.commit;
            std::printf(
                "Batch commit timing: batches=%d queries=%d free_prev=%.3f ms build=%.3f ms "
                "initial_update=%.3f ms evaluate=%.3f ms append=%.3f ms newick=%.3f ms\n",
                batch_commit_timing.batches,
                batch_commit_timing.queries,
                batch_commit_timing.free_prev_ms,
                batch_commit_timing.build_ms,
                batch_commit_timing.initial_update_ms,
                batch_commit_timing.evaluate_ms,
                batch_commit_timing.append_ms,
                batch_commit_timing.newick_ms);
            std::printf(
                "Batch query timing: evals=%d reset=%.3f ms build_query_clv=%.3f ms "
                "placement_kernel_total=%.3f ms\n",
                stats.query_evals,
                stats.query_reset_stage_ms,
                stats.query_build_clv_stage_ms,
                stats.query_kernel_total_ms);
            std::printf(
                "Batch insertion timing: initial_updates=%d insertion_updates=%d "
                "initial_up_host=%.3f ms initial_up_stage=%.3f ms initial_down_host=%.3f ms initial_down_stage=%.3f ms "
                "insert_pre_clv=%.3f ms insert_up_host=%.3f ms insert_up_stage=%.3f ms insert_down_host=%.3f ms insert_down_stage=%.3f ms\n",
                stats.initial_updates,
                stats.insertion_updates,
                stats.initial_upward_host_ms,
                stats.initial_upward_stage_ms,
                stats.initial_downward_host_ms,
                stats.initial_downward_stage_ms,
                stats.insertion_pre_clv_ms,
                stats.insertion_upward_host_ms,
                stats.insertion_upward_stage_ms,
                stats.insertion_downward_host_ms,
                stats.insertion_downward_stage_ms);
            std::printf(
                "Batch op summary: initial_up_ops=%lld initial_down_ops=%lld "
                "insert_up_ops=%lld insert_down_ops=%lld\n",
                stats.initial_upward_ops,
                stats.initial_downward_ops,
                stats.insertion_upward_ops,
                stats.insertion_downward_ops);
        }
    } else {
        res = BuildAllToGPU(
            msa_names,
            rows,
            newick,
            Q,
            pi,
            rate_multipliers,
            rate_weights,
            pattern_weights_arg,
            sites,
            states,
            rate_cats,
            per_rate_scaling,
            placement_queries,
            commit_to_tree);
        if (res.tree.nodes.empty() || res.dev.N == 0) {
            throw std::runtime_error("BuildAllToGPU returned empty tree/device structures.");
        }
        if (res.tree.root_id < 0) {
            throw std::runtime_error("BuildAllToGPU produced tree with invalid root_id.");
        }

        std::cout << "Uploaded. N=" << res.dev.N << ", tips=" << res.dev.tips
                    << ", per_node_elems=" << res.dev.per_node_elems() << "\n";

        if (mlenv::env_flag_enabled("MLIPPER_DEBUG_TREE_STRUCTURE")) {
            mltreeio::print_tree_structure(res.tree);
        }

        cudaEventRecord(start);
        placement_ops.profile_commit_timing = []() {
            const char* env = std::getenv("MLIPPER_PROFILE_COMMIT_TIMING");
            return env && std::atoi(env) != 0;
        }();
        UpdateTreeClvs(
            res.dev,
            res.tree,
            res.hostPack,
            placement_ops,
            stream);
        double logL = root_likelihood::compute_root_loglikelihood_total(
            res.dev,
            res.tree.root_id,
            res.dev.d_pattern_weights_u,
            nullptr,
            0.0,
            0);
        printf("Initial tree log-likelihood = %.12f\n", logL);
        std::vector<std::string> placement_query_names;
        if (commit_to_tree) {
            placement_query_names.reserve(placement_queries.size());
            for (const NewPlacementQuery& query : placement_queries) {
                placement_query_names.push_back(query.msa_name);
            }
        }
        PlacementCommitContext commit_ctx;
        commit_ctx.placement_ops = &placement_ops;

        bool actual_commit_to_tree = commit_to_tree;
        if (commit_to_tree) {
            commit_ctx.tree = &res.tree;
            commit_ctx.host = &res.hostPack;
            commit_ctx.queries = &res.queries;
            commit_ctx.query_names = &placement_query_names;
            commit_ctx.inserted_query_names = &committed_query_names;
        }
        EvaluatePlacementQueries(
            res.dev,
            res.eig,
            rate_multipliers,
            commit_ctx,
            &placement_results,
            1,
            actual_commit_to_tree,
            stream);
    }
    if (commit_to_tree) {
        const double committed_logL = root_likelihood::compute_root_loglikelihood_total(
            res.dev,
            res.tree.root_id,
            res.dev.d_pattern_weights_u,
            nullptr,
            0.0,
            0);
        printf("Committed tree log-likelihood = %.12f\n", committed_logL);
        if (placement_ops.profile_commit_timing) {
            const CommitTimingStats& stats = placement_ops.timing;
            printf(
                "Commit timing summary: initial_updates=%d insertion_updates=%d "
                "initial_up_host=%.3f ms initial_up_stage=%.3f ms initial_down_host=%.3f ms initial_down_stage=%.3f ms "
                "query_reset=%.3f ms query_build_clv=%.3f ms query_kernel_total=%.3f ms "
                "insert_pre_clv=%.3f ms insert_up_host=%.3f ms insert_up_stage=%.3f ms insert_down_host=%.3f ms insert_down_stage=%.3f ms\n",
                stats.initial_updates,
                stats.insertion_updates,
                stats.initial_upward_host_ms,
                stats.initial_upward_stage_ms,
                stats.initial_downward_host_ms,
                stats.initial_downward_stage_ms,
                stats.query_reset_stage_ms,
                stats.query_build_clv_stage_ms,
                stats.query_kernel_total_ms,
                stats.insertion_pre_clv_ms,
                stats.insertion_upward_host_ms,
                stats.insertion_upward_stage_ms,
                stats.insertion_downward_host_ms,
                stats.insertion_downward_stage_ms);
            printf(
                "Commit op summary: query_evals=%d initial_up_ops=%lld initial_down_ops=%lld "
                "insert_up_ops=%lld insert_down_ops=%lld\n",
                stats.query_evals,
                stats.initial_upward_ops,
                stats.initial_downward_ops,
                stats.insertion_upward_ops,
                stats.insertion_downward_ops);
        }

    }

    const double final_logL = root_likelihood::compute_root_loglikelihood_total(
        res.dev,
        res.tree.root_id,
        res.dev.d_pattern_weights_u,
        nullptr,
        0.0,
        0);
    printf("Final tree log-likelihood = %.12f\n", final_logL);
    if (mlenv::env_flag_enabled("MLIPPER_DEBUG_TREE_STRUCTURE")) {
        mltreeio::print_tree_structure(res.tree);
    }

    free_placement_op_buffer(placement_ops, stream);
    cudaStreamSynchronize(stream);

    if (!commit_tree_out.empty()) {
        const size_t collapsed_internal_branches = mltreeio::write_tree_to_newick_file(
            res.tree,
            commit_tree_out,
            commit_collapse_internal_epsilon);
        std::cout << "Wrote Newick tree to " << commit_tree_out;
        if (commit_collapse_internal_epsilon >= 0.0) {
            std::cout << " after collapsing " << collapsed_internal_branches
                      << " internal branches <= " << commit_collapse_internal_epsilon;
        }
        std::cout << "\n";
    }

    if (!jplace_out.empty()) {
        if (placement_results.size() != placement_queries.size()) {
            std::cerr << "placement result count mismatch before jplace export: got "
                      << placement_results.size() << ", expected " << placement_queries.size()
                      << ". Exporting available prefix only.\n";
        }

        const mljplace::JplaceTreeExport jplace_tree =
            mljplace::build_jplace_tree_export(res.tree);
        const std::vector<mljplace::JplacePlacementRecord> jplace_records =
            mljplace::build_jplace_records(
                res.tree,
                jplace_tree,
                placement_results,
                placement_queries);

        std::ostringstream invocation;
        for (int i = 0; i < argc; ++i) {
            if (i) invocation << ' ';
            invocation << argv[i];
        }
        mljplace::write_jplace(
            jplace_out,
            jplace_tree.tree,
            jplace_records,
            invocation.str());
        std::cout << "Wrote jplace to " << jplace_out << "\n";
    }

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time (milliseconds)
    cudaEventElapsedTime(&gpu_ms_kernel, start, stop);
    const auto end_gpu = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    printf("GPU kernel time = %.3f ms\n", gpu_ms_kernel);
    printf("GPU Wall Clock time = %.3f ms\n", gpu_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    free_device_tree(res.dev);

    return 0;
    } catch (const std::exception& e) {
        free_device_tree(res.dev);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
        std::cout.flush();
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
