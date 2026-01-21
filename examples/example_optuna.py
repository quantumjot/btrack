import btrack
from btrack import datasets, constants, utils, config
import numpy as np
import pandas as pd
import json
from joblib import parallel_backend
import optuna

#import ctc-tools
import sys
sys.path.append('/Users/tim/Documents/ctc-tools')
import ctctools

# Import traccuracy specific libraries
from traccuracy.loaders import load_ctc_data
from traccuracy.loaders._ctc import ctc_to_graph, _get_node_attributes
from traccuracy._tracking_graph import TrackingGraph
from traccuracy import run_metrics
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics

def write_best_params_to_config(params, config_file_path):
    # Define the structure of the config file
    config = {
        "TrackerConfig": {
            "MotionModel": {
                "name": "cell_motion",
                "dt": params.get('dt', 1.0),
                "measurements": params.get('measurements', 3),
                "states": params.get('states', 6),
                "accuracy": params.get('accuracy', 7.5),
                "prob_not_assign": params.get('prob_not_assign', 0.1),
                "max_lost": params.get('max_lost', 5),
                "A": {
                    "matrix": [
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1
                    ]
                },
                "H": {
                    "matrix": [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0
                    ]
                },
                "P": {
                    "sigma": params.get('p_sigma', 150.0),
                    "matrix": [
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1
                    ]
                },
                "G": {
                    "sigma": params.get('g_sigma', 15.0),
                    "matrix": [
                        0.5,
                        0.5,
                        0.5,
                        1,
                        1,
                        1
                    ]
                },
                "R": {
                    "sigma": params.get('r_sigma', 5.0),
                    "matrix": [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1
                    ]
                }
            },
            "ObjectModel": {},
            "HypothesisModel": {
                "name": "cell_hypothesis",
                "hypotheses": [
                    "P_FP",
                    "P_init",
                    "P_term",
                    "P_link",
                    "P_branch",
                    "P_dead"
                ],
                "lambda_time": params.get('lambda_time', 5.0),
                "lambda_dist": params.get('lambda_dist', 3.0),
                "lambda_link": params.get('lambda_link', 10.0),
                "lambda_branch": params.get('lambda_branch', 50.0),
                "eta": params.get('eta', 1e-10),
                "theta_dist": params.get('theta_dist', 20.0),
                "theta_time": params.get('theta_time', 5.0),
                "dist_thresh": params.get('dist_thresh', 40),
                "time_thresh": params.get('time_thresh', 2),
                "apop_thresh": params.get('apop_thresh', 5),
                "segmentation_miss_rate": params.get('segmentation_miss_rate', 0.1),
                "apoptosis_rate": params.get('apoptosis_rate', 0.001),
                "relax": params.get('relax', True),
            }
        }
    }

    # Open the config file in write mode
    with open(config_file_path, 'w') as file:
        # Write the config to the file as JSON
        json.dump(config, file, indent=4)

def read_config_params(config_file_path):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"The configuration file {config_file_path} does not exist.")

    try:
        with open(config_file_path, 'r') as file:
            config_params = json.load(file)
        return config_params
    except json.JSONDecodeError:
        raise ValueError(f"The file {config_file_path} is not a valid JSON file.")

def scale_matrix(matrix: np.ndarray, original_sigma: float, new_sigma: float) -> np.ndarray:
    """
    Scales a matrix by first reverting the original scaling and then applying a new sigma value.

    Parameters:
    matrix (np.ndarray): The matrix to be scaled.
    original_sigma (float): The original sigma value used to scale the matrix.
    new_sigma (float): The new sigma value to scale the matrix.

    Returns:
    np.ndarray: The rescaled matrix.
    """

    # Revert the original scaling
    if original_sigma != 0:
        unscaled_matrix = matrix / original_sigma
    else:
        unscaled_matrix = matrix.copy()  # Avoid division by zero

    # Apply the new sigma scaling
    rescaled_matrix = unscaled_matrix * new_sigma

    return rescaled_matrix

def add_missing_attributes(graph):
    """
    Function to add attribute to nodes without any attributes, essentially inserting dummy nodes where necessary.
    """
    for node in graph.nodes:
        if not graph.nodes[node]:
            segmentation_id_str, time_str = node.split('_') # Split the node identifier into segmentation_id and time
            segmentation_id = int(segmentation_id_str)
            time = int(time_str)
            prev_node = f"{segmentation_id}_{time - 1}" # Find the previous time step node
            
            # Copy x and y from the previous time step if it exists, else initialize to None
            x = graph.nodes[prev_node].get('x') if graph.has_node(prev_node) else None
            y = graph.nodes[prev_node].get('y') if graph.has_node(prev_node) else None
            
            attributes = {
                'segmentation_id': segmentation_id,
                'x': x,
                'y': y,
                't': time
            }
            graph.nodes[node].update(attributes)

def run_cell_tracking_algorithm(objects, config, volume, max_search_radius=100, use_napari=False):
    """
    Function to run a cell tracking algorithm using the btrack library.

    Parameters:
    objects: The objects to be tracked.
    config: The configuration for the tracker.
    volume: The volume in which the objects are located.
    max_search_radius (optional, default is 100): The maximum search radius for the tracker.
    use_napari (optional, default is False): A flag indicating whether to use Napari.

    Returns:
    lbep: The lower bound of the highest posterior density.
    tracks: The tracks of the objects.
    nap_data, nap_properties, nap_graph: The tracker data in Napari format (only if use_napari is True).
    """
    if not objects:
        raise ValueError("objects must not be empty.")
    if not isinstance(max_search_radius, (int, float)) or max_search_radius <= 0:
        raise ValueError("max_search_radius must be a positive number.")
    if not isinstance(use_napari, bool):
        raise ValueError("use_napari must be a boolean.")

    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config)
        tracker.max_search_radius = max_search_radius
        tracker.append(objects)
        # Reverse the volume to fit btrack requirements. The volume determines whether or not objects can leave the field of view.
        # For example, if the bottom is bounded by glass, the z value needs to go from really negative to show the cells can't leave the field of view.
        tracker.volume = volume[::-1]
        tracker.track(step_size=100)
        tracker.optimize()
        tracks = tracker.tracks
        lbep = tracker.LBEP
        if use_napari:
            nap_data, nap_properties, nap_graph = tracker.to_napari()
            return lbep, tracks, nap_data, nap_properties, nap_graph
        else:
            return lbep, tracks
        

def calculate_accuracy(lbep, segmentation, ground_truth_data):
    """
    Function to calculate the accuracy of a cell tracking algorithm.

    Parameters:
    lbep: Lower bound of the highest posterior density.
    segmentation: Segmentation data.
    ground_truth_data: Ground truth data.

    Returns:
    results: A dictionary containing the results of the accuracy calculation.
    """
    # Create DataFrame from lbep data
    tracks_df = pd.DataFrame({
        "Cell_ID": lbep[:, 0],
        "Start": lbep[:, 1],
        "End": lbep[:, 2],
        "Parent_ID": [0 if lbep[idx, 3] == lbep[idx, 0] else lbep[idx, 3] for idx in range(lbep.shape[0])],
    })

    # Get node attributes from segmentation data and convert to graph
    detections_df = _get_node_attributes(segmentation)
    graph = ctc_to_graph(tracks_df, detections_df)
    add_missing_attributes(graph)

    # Run metrics on ground truth data and predicted data
    predicted_data = TrackingGraph(graph, segmentation)
    ctc_results = run_metrics(
        gt_data=ground_truth_data,
        pred_data=predicted_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )

    # Extract results for keys
    keys = ['fp_nodes', 'fn_nodes', 'ns_nodes', 'fp_edges', 'fn_edges', 'ws_edges', 'TRA', 'DET', 'AOGM']
    results = {key: next((m['results'][key] for m in ctc_results if key in m['results']), None) for key in keys}

    # Extract 'Mitotic Branching Correctness'
    mbc_key = 'Mitotic Branching Correctness'
    for m in ctc_results:
        if 'Frame Buffer 0' in m['results'] and mbc_key in m['results']['Frame Buffer 0']:
            results[mbc_key] = m['results']['Frame Buffer 0'][mbc_key]
            break

    return results

def objective(trial, dataset, gt_data, param_ranges):
    """
    Objective function for Bayesian Optimization. This function is used to optimize the parameters of a cell tracking algorithm.

    Parameters:
    trial: The current trial.
    dataset: The dataset to be used.
    gt_data: Ground truth data.
    param_ranges: The ranges for the parameters to be optimized.

    Returns:
    results: A dictionary containing the results of the accuracy calculation.
    """
    # Load param ranges
    params = {}
    for param, (low, high, *type_) in param_ranges.items():
        if type_ and type_[0] == 'int':
            params[param] = trial.suggest_int(param, low, high)
        else:
            params[param] = trial.suggest_float(param, low, high)

    # Segmentation
    objects = utils.segmentation_to_objects(dataset.segmentation, properties=('area', )) 

    # Load config to be modified
    conf = config.load_config('cell_config.json') #config

    # Task: add a way to optimise Z dimension during trial
    volume = dataset.volume

    # Set each attribute individually
    attributes = {
        'theta_dist': params['theta_dist'],
        'lambda_dist': params['lambda_dist'],
        'lambda_link': params['lambda_link'],
        'lambda_branch': params['lambda_branch'],
        'theta_time': params['theta_time'],
        'dist_thresh': params['dist_thresh'],
        'time_thresh': params['time_thresh'],
        'apop_thresh': params['apop_thresh'],
        'P': scale_matrix(conf.motion_model.P, 150.0, params['p_sigma']),
        'G': scale_matrix(conf.motion_model.G, 15.0, params['g_sigma']),
        'R': scale_matrix(conf.motion_model.R, 5.0, params['r_sigma']),
        'max_lost': params['max_lost'],
        'prob_not_assign': params['prob_not_assign']
    }

    # Set attributes for hypothesis model and motion model
    for attr, value in attributes.items():
        if attr in ['P', 'G', 'R', 'max_lost', 'prob_not_assign']: #add motion model attributes if they change
            setattr(conf.motion_model, attr, value)
        else:
            setattr(conf.hypothesis_model, attr, value)

    # Set division hypothesis
    hypotheses = [
        "P_FP",
        "P_init",
        "P_term",
        "P_link",
        "P_dead"
    ]
    if params['div_hypothesis'] == 1:
        hypotheses.append("P_branch")
    elif params['div_hypothesis'] != 0:
        raise ValueError(f"Invalid value for div_hypothesis: {params['div_hypothesis']}. It should be 0 or 1.")
    setattr(conf.hypothesis_model, 'hypotheses', hypotheses)

    lbep, tracks = run_cell_tracking_algorithm(objects, conf, volume, params['max_search_radius'])
    segm = utils.update_segmentation(np.asarray(dataset.segmentation), tracks)
    results = calculate_accuracy(lbep, segm, gt_data)

    # Store additional metrics in the trial's user attributes
    for attr, value in results.items():
        trial.set_user_attr(attr, value)

    return results['AOGM'], results['Mitotic Branching Correctness']

def perform_study(dataset_name, gt_data, dataset, param_ranges, n_trials = 256, use_parallel_backend = False):
    """
    Function to perform a study using Bayesian Optimization.

    Parameters:
    dataset_name: The name of the dataset.
    gt_data: Ground truth data.
    dataset: The dataset to be used.
    param_ranges: The ranges for the parameters to be optimized.
    n_trials: The number of trials to run. Default is 256.
    use_parallel_backend: Whether to use a parallel backend for optimization. Default is False.

    Returns:
    study: The study object from Optuna.
    """
    # Bayesian Optimization
    study = optuna.create_study(directions=["minimize", "maximize"], study_name=f"btrack_{dataset_name}", storage="sqlite:///btrack.db", load_if_exists=True)

    # Perform Bayesian optimization
    if use_parallel_backend:
        with parallel_backend('multiprocessing'):
            study.optimize(lambda trial: objective(trial, dataset, gt_data, param_ranges), n_trials=n_trials, n_jobs=4, gc_after_trial=True)
    else:
        study.optimize(lambda trial: objective(trial, dataset, gt_data, param_ranges), n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    return study

def get_optimized_params_df(best_trial, index_name):
    """
    Function to get a DataFrame with the parameters and results of a trial.

    Parameters:
    best_trial: The trial.
    index_name: The name to use for the index of the DataFrame.

    Returns:
    optimized_params_df: A DataFrame with the parameters and results of the trial.
    """
    optimized_params = {
        **best_trial.params, 
        'best_AOGM': best_trial.values[0], 
        'best_TRA': best_trial.user_attrs["TRA"], 
        'best_DET': best_trial.user_attrs["DET"],
        'AOGM': best_trial.user_attrs["AOGM"],
        'MBC': best_trial.user_attrs["Mitotic Branching Correctness"],
        'fp_nodes': best_trial.user_attrs["fp_nodes"],
        'fn_nodes': best_trial.user_attrs["fn_nodes"],
        'ns_nodes': best_trial.user_attrs["ns_nodes"],
        'fp_edges': best_trial.user_attrs["fp_edges"],
        'fn_edges': best_trial.user_attrs["fn_edges"],
        'ws_edges': best_trial.user_attrs["ws_edges"],
        'total_trials': len(study.trials)
    }

    # Convert the optimized parameters to a DataFrame and set the index to index_name
    optimized_params_df = pd.DataFrame([optimized_params], index=[index_name])

    return optimized_params_df

if __name__ == "__main__":

    dataset_path = 'path/to/dataset'  # Replace with the path to your dataset
    experiment = "0N"  # Replace with the experiment number
    
    gt_path1 = f'{dataset_path}/{experiment}_GT/TRA'  # This is the path to the ground truth data
    gt_path2 = f'{dataset_path}/{experiment}_GT/TRA/man_track.txt'  # This is the path to the man_track.txt file within the ground truth data
    gt_data = load_ctc_data(gt_path1, gt_path2) # Load ground truth data for the dataset
    dataset = ctctools.load(dataset_path, experiment=experiment, scale=(1., 1., 1.)) # Load the dataset

    # Define the number of trials
    n_trials = 10

    # Suggest ranges for each parameter
    param_ranges = {
        'theta_dist': (15, 25),
        'lambda_dist': (0, 15),
        'lambda_link': (0, 30),
        'lambda_branch': (0, 100),
        'theta_time': (0, 10),
        'dist_thresh': (1, 120, 'int'),
        'time_thresh': (1, 4, 'int'),
        'apop_thresh': (1, 8, 'int'),
        'p_sigma': (120, 200),
        'g_sigma': (10, 25),
        'r_sigma': (0.5, 40),
        'max_lost': (1, 10, 'int'), #if more than 10 causes leaked semaphore object error
        'prob_not_assign': (0.0, 0.5),
        'max_search_radius': (50, 200, 'int'),
        'div_hypothesis': (0, 1, 'int')  # do not change range as this functions as a boolean
    } 

    dataset_name = "dataset_name"  # Replace with the name of your dataset or trial

    # Run optimization
    study = perform_study(dataset_name, gt_data, dataset, param_ranges, n_trials, use_parallel_backend=True)

    #select trial with highest AOGM and MBC
    best_trial_0 = study.best_trials[0]
    best_trial_1 = study.best_trials[-1]

    # Initialize the DataFrame
    optimized_params_per_dataset = pd.DataFrame()

    # save parameters and results of trials in csv and json files
    for i, best_trial in enumerate(study.best_trials):
        index_name = f"{dataset_name}_{i}"
        optimized_params_df = get_optimized_params_df(best_trial, index_name)

        # Append it to the main DataFrame
        optimized_params_per_dataset = pd.concat([optimized_params_per_dataset, optimized_params_df])

    # Set path for the CSV file
    csv_file_path = 'results.csv'

    # Convert dictionary to DataFrame and save as CSV
    pd.DataFrame(optimized_params_per_dataset).to_csv(csv_file_path, index_label='Dataset')

    # Set paths for the config files
    config_0_path = 'config_0.json'
    config_1_path = 'config_1.json'

    write_best_params_to_config(config_0_path, best_trial_0.params)
    write_best_params_to_config(config_1_path, best_trial_1.params)