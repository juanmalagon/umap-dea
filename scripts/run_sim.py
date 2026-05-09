import os
import gc
import argparse
import json
import logging
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from umap_dea.config import SimulationConfig
from umap_dea import dgp, dim_red, dea, eval

# Prevent numpy from spawning too many threads per worker,
# which causes thread contention and slowdown in parallel runs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


logger = logging.getLogger(__name__)

ParamsDict = dict[str, Any]
SimulationResult = dict[str, Any]


def run_simulation(params_dict: ParamsDict) -> pd.DataFrame:
    """
    Run a single simulation.
    """

    N = params_dict['N']
    M = params_dict['M']
    n = params_dict['n']
    alpha_1 = params_dict['alpha_1']
    gamma = params_dict['gamma']
    sigma_u = params_dict['sigma_u']
    rts = params_dict['rts']
    orientation = params_dict['orientation']
    seed = params_dict['seed']
    pca = params_dict['pca']
    umap_n_neighbors = params_dict.get('umap_n_neighbors', 15)
    umap_min_dist = params_dict.get('umap_min_dist', 0.1)
    umap_metric = params_dict.get('umap_metric', 'euclidean')

    # Data Generating Process
    data_dict = dgp.generate_data_dict(
        n=n,
        N=N,
        M=M,
        alpha_1=alpha_1,
        gamma=gamma,
        sigma_u=sigma_u,
        verbose=False
    )
    x = data_dict["x"]
    y = data_dict["y"]
    y_tilde = data_dict["y_tilde"]
    efficiency_score_by_design = (y/y_tilde).squeeze()

    # Dimensionality Reduction
    embeddings = dim_red.create_embeddings(
        x=x,
        seed=seed,
        pca=pca,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
    )
    embeddings_df_dict = embeddings['embeddings_df_dict']
    dims_for_embedding_dict = embeddings['dims_for_embedding_dict']

    # Calculate DEA
    efficiency_scores_dict = dea.calculate_dea_for_embeddings(
        embeddings_df_dict=embeddings_df_dict,
        y=y,
        rts=rts,
        orientation=orientation
    )

    # Evaluate Results
    evaluation_df = eval.create_evaluation_df(
        efficiency_scores_dict=efficiency_scores_dict,
        efficiency_score_by_design=efficiency_score_by_design,
        dims_for_embedding_dict=dims_for_embedding_dict,
    )

    return evaluation_df


def export_results(evaluation_df_list: list,
                   errors_list: list,
                   params_dict: ParamsDict,
                   run_serial: str,
                   results_dir: str) -> None:
    """
    Export results to csv files.

    .. deprecated::
        This function is kept for backward compatibility. The new
        ``wrapper_function`` writes results incrementally via streaming.
    """
    # Save parameters
    pd.DataFrame(params_dict, index=[0]).to_csv(
        os.path.join(results_dir, f'params_dict_{run_serial}.csv'),
        index=False)

    # Save errors
    errors_list_df = pd.DataFrame(errors_list, columns=['iteration'])
    errors_list_df.to_csv(
        os.path.join(results_dir, f'errors_list_{run_serial}.csv'),
        index=False)

    # Save evaluation
    evaluation_df = pd.concat(evaluation_df_list)
    evaluation_df.to_csv(
        os.path.join(results_dir, f'evaluation_df_{run_serial}.csv'),
        index=False)

    # Save summary
    summary_df = evaluation_df.groupby(['dim_reduction_level', 'dims']).agg(
        {
            'mae': ['mean', 'std'],
            'spearmanr': ['mean', 'std'],
            'pearsonr': ['mean', 'std'],
            'kendalltau': ['mean', 'std'],
            'nr_efficient': ['mean', 'std'],
            'prop_efficient': ['mean', 'std'],
            'nr_non_nan': ['mean', 'std'],
            'spearmanr_warning': 'sum',
            'kendalltau_warning': 'sum',
        }
    ).reset_index()
    summary_df.columns = [
        'dim_reduction_level',
        'dims',
        'mae_mean',
        'mae_std',
        'spearmanr_mean',
        'spearmanr_std',
        'pearsonr_mean',
        'pearsonr_std',
        'kendalltau_mean',
        'kendalltau_std',
        'nr_efficient_mean',
        'nr_efficient_std',
        'prop_efficient_mean',
        'prop_efficient_std',
        'nr_non_nan_mean',
        'nr_non_nan_std',
        'spearmanr_warning_count',
        'kendalltau_warning_count',
    ]
    summary_df.sort_values(by=['dims', 'dim_reduction_level']).to_csv(
        os.path.join(results_dir, f'summary_df_{run_serial}.csv'), index=False)

    return None


def run_simulation_wrapper(args: tuple[ParamsDict, int]) -> SimulationResult:
    """
    Wrapper function for run_simulation to handle exceptions in parallel processing.
    """
    params_dict, i = args
    try:
        # Set unique seed for this iteration in the worker process
        # This ensures reproducible but diverse data across iterations
        iteration_seed = params_dict['seed'] + i
        np.random.seed(iteration_seed)

        evaluation_df = run_simulation(params_dict)
        evaluation_df['iteration'] = i
        result = {'evaluation_df': evaluation_df, 'error': None, 'iteration': i}
        # Explicit cleanup to release memory in the worker process
        gc.collect()
        return result
    except Exception as e:
        logger.exception('Error in iteration %s: %s', i, str(e))
        gc.collect()
        return {'evaluation_df': None, 'error': str(e), 'iteration': i}


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

def wrapper_function(params_dict: ParamsDict, results_dir: str) -> None:
    """
    Parallelized wrapper function to run the simulation study.

    Uses batch processing with imap_unordered to stream results to disk
    incrementally, preventing memory accumulation from holding all 1000
    DataFrames in memory simultaneously.
    """
    run_serial = str(uuid4())

    logger.info('INITIAL SETUP')
    logger.info('Number of inputs: %s', params_dict["N"])
    logger.info('Number of outputs: %s', params_dict["M"])
    logger.info('Number of DMUs: %s', params_dict["n"])
    logger.info('Parameter alpha_1: %s', params_dict["alpha_1"])
    logger.info('Parameter gamma: %s', params_dict["gamma"])
    logger.info('Parameter sigma_u: %s', params_dict["sigma_u"])
    logger.info('Return to scale: %s', params_dict["rts"])
    logger.info('Orientation: %s', params_dict["orientation"])
    logger.info('Seed: %s', params_dict["seed"])
    logger.info('PCA enabled: %s', params_dict["pca"])
    logger.info('UMAP n_neighbors: %s', params_dict.get("umap_n_neighbors", 15))
    logger.info('UMAP min_dist: %s', params_dict.get("umap_min_dist", 0.1))
    logger.info('UMAP metric: %s', params_dict.get("umap_metric", "euclidean"))
    logger.info('Number of available CPUs: %s', cpu_count())
    logger.info('Number of simulations: %s', params_dict["nr_simulations"])

    # Save parameters immediately
    pd.DataFrame(params_dict, index=[0]).to_csv(
        os.path.join(results_dir, f'params_dict_{run_serial}.csv'),
        index=False,
    )

    # Set random seed for main process
    np.random.seed(params_dict['seed'])

    # Prepare arguments for parallel processing
    args_list = [
        (params_dict.copy(), i)
        for i in range(params_dict['nr_simulations'])
    ]

    # Determine number of processes to use (leave one CPU free)
    n_processes = max(1, cpu_count() - 1)

    # Batch size: recreate the Pool periodically to fully release worker memory
    batch_size = min(200, len(args_list))
    total_batches = (len(args_list) + batch_size - 1) // batch_size

    evaluation_csv_path = os.path.join(
        results_dir, f'evaluation_df_{run_serial}.csv'
    )
    # Remove stale file if it exists
    if os.path.exists(evaluation_csv_path):
        os.remove(evaluation_csv_path)

    errors_list = []
    header_written = False

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(args_list))
        batch_args = args_list[batch_start:batch_end]

        logger.info(
            'Batch %s/%s: iterations %s-%s of %s',
            batch_idx + 1,
            total_batches,
            batch_start,
            batch_end - 1,
            len(args_list),
        )

        with Pool(processes=n_processes) as pool:
            # Use imap_unordered to stream results as they complete,
            # instead of collecting all 1000 in memory at once.
            for result in pool.imap_unordered(run_simulation_wrapper, batch_args):
                if result['error'] is None:
                    df = result['evaluation_df']
                    # Append to CSV immediately — never hold more than one
                    # DataFrame in memory at a time.
                    df.to_csv(
                        evaluation_csv_path,
                        mode='a',
                        header=not header_written,
                        index=False,
                    )
                    header_written = True
                    del df
                else:
                    errors_list.append(result['iteration'])

        # Force garbage collection between batches to release any
        # memory that the Pool workers didn't free.
        gc.collect()
        logger.info(
            'Batch %s/%s complete. Errors so far: %s',
            batch_idx + 1,
            total_batches,
            len(errors_list),
        )

    logger.info('All batches complete. Writing final outputs...')

    # Save errors
    errors_list_df = pd.DataFrame(errors_list, columns=['iteration'])
    errors_list_df.to_csv(
        os.path.join(results_dir, f'errors_list_{run_serial}.csv'),
        index=False,
    )

    # Read back the evaluation CSV and compute summary
    if header_written:
        evaluation_df = pd.read_csv(evaluation_csv_path)
        summary_df = evaluation_df.groupby(['dim_reduction_level', 'dims']).agg(
            {
                'mae': ['mean', 'std'],
                'spearmanr': ['mean', 'std'],
                'pearsonr': ['mean', 'std'],
                'kendalltau': ['mean', 'std'],
                'nr_efficient': ['mean', 'std'],
                'prop_efficient': ['mean', 'std'],
                'nr_non_nan': ['mean', 'std'],
                'spearmanr_warning': 'sum',
                'kendalltau_warning': 'sum',
            }
        ).reset_index()
        summary_df.columns = [
            'dim_reduction_level',
            'dims',
            'mae_mean',
            'mae_std',
            'spearmanr_mean',
            'spearmanr_std',
            'pearsonr_mean',
            'pearsonr_std',
            'kendalltau_mean',
            'kendalltau_std',
            'nr_efficient_mean',
            'nr_efficient_std',
            'prop_efficient_mean',
            'prop_efficient_std',
            'nr_non_nan_mean',
            'nr_non_nan_std',
            'spearmanr_warning_count',
            'kendalltau_warning_count',
        ]
        summary_df.sort_values(by=['dims', 'dim_reduction_level']).to_csv(
            os.path.join(results_dir, f'summary_df_{run_serial}.csv'),
            index=False,
        )
    else:
        logger.warning(
            'No successful evaluations — skipping summary generation.'
        )

    logger.info('Completed all %s simulations!', params_dict["nr_simulations"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for UMAP-DEA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "config.json"),
        help="Path to the JSON configuration file (default: config.json)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO). Use WARNING to suppress per-step noise."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Load configuration from JSON file
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    # Create SimulationConfig from loaded dict
    config = SimulationConfig(**config_dict)
    
    # Convert config to dict for wrapper_function
    params_dict = asdict(config)
    
    # Set up results directory
    results_dir = os.path.join(_PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the simulations
    wrapper_function(params_dict, results_dir)
