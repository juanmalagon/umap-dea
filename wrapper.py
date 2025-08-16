import os
import pandas as pd
import numpy as np
from uuid import uuid4
from multiprocessing import Pool, cpu_count
import traceback

import data_generating_process as dgp
import dimensionality_reduction as dr
import calculate_dea as cdea
import evaluate_results as er


def run_simulation(params_dict: dict) -> pd.DataFrame:
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
    embeddings = dr.create_embeddings(x=x, seed=seed, pca=pca)
    embeddings_df_dict = embeddings['embeddings_df_dict']
    dims_for_embedding_dict = embeddings['dims_for_embedding_dict']

    # Calculate DEA
    efficiency_scores_dict = cdea.calculate_dea_for_embeddings(
        embeddings_df_dict=embeddings_df_dict,
        y=y,
        rts=rts,
        orientation=orientation
    )

    # Evaluate Results
    evaluation_df = er.create_evaluation_df(
        efficiency_scores_dict=efficiency_scores_dict,
        efficiency_score_by_design=efficiency_score_by_design,
        dims_for_embedding_dict=dims_for_embedding_dict,
    )

    return evaluation_df


def export_results(evaluation_df_list: list,
                   errors_list: list,
                   params_dict: dict,
                   run_serial: str,
                   results_dir: str) -> None:
    """
    Export results to csv files.
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
        {'mae': ['mean', 'std'],
         'spearmanr': ['mean', 'std'],
         'pearsonr': ['mean', 'std'],
         'kendalltau': ['mean', 'std']}).reset_index()
    summary_df.columns = ['dim_reduction_level', 'dims', 'mae_mean', 'mae_std',
                          'spearmanr_mean', 'spearmanr_std', 'pearsonr_mean',
                          'pearsonr_std', 'kendalltau_mean', 'kendalltau_std']
    summary_df.sort_values(by=['dims', 'dim_reduction_level']).to_csv(
        os.path.join(results_dir, f'summary_df_{run_serial}.csv'), index=False)

    return None


def run_simulation_wrapper(args):
    """
    Wrapper function for run_simulation to handle exceptions in parallel processing.
    """
    params_dict, i = args
    try:
        evaluation_df = run_simulation(params_dict)
        evaluation_df['iteration'] = i
        return {'evaluation_df': evaluation_df, 'error': None, 'iteration': i}
    except Exception as e:
        print(f'Error in iteration {i}: {str(e)}')
        traceback.print_exc()
        return {'evaluation_df': None, 'error': str(e), 'iteration': i}


def wrapper_function(params_dict: dict, results_dir: str):
    """
    Parallelized wrapper function to run the simulation study.
    """
    run_serial = str(uuid4())

    print('INITIAL SETUP \n')
    print(f'Number of inputs: {params_dict["N"]}')
    print(f'Number of outputs: {params_dict["M"]}')
    print(f'Number of DMUs: {params_dict["n"]}')
    print(f'Parameter alpha_1: {params_dict["alpha_1"]}')
    print(f'Parameter gamma: {params_dict["gamma"]}')
    print(f'Parameter sigma_u: {params_dict["sigma_u"]}')
    print(f'Return to scale: {params_dict["rts"]}')
    print(f'Orientation: {params_dict["orientation"]}')
    print(f'Seed: {params_dict["seed"]}')
    print(f'PCA enabled: {params_dict["pca"]}')
    print(f'Number of available CPUs: {cpu_count()}')
    print(f'Number of simulations: {params_dict["nr_simulations"]}')

    # Set random seed for main process
    np.random.seed(params_dict['seed'])

    # Prepare arguments for parallel processing
    args_list = [(params_dict.copy(), i) for i in range(params_dict['nr_simulations'])]

    # Determine number of processes to use (leave one CPU free)
    n_processes = max(1, cpu_count() - 1)

    evaluation_df_list = []
    errors_list = []

    print(f'Starting parallel processing with {n_processes} workers...')
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_simulation_wrapper, args_list)

    # Process results
    for result in results:
        if result['error'] is None:
            evaluation_df_list.append(result['evaluation_df'])
        else:
            errors_list.append(result['iteration'])

    export_results(evaluation_df_list, errors_list, params_dict, run_serial, results_dir)

    return None
