===============
``optuna``
===============

Setting up btrack parameter optimization using ``optuna``
-----------------------------------------------------------

To install the requirements for this script, install libraries via

.. code-block:: bash

    pip install numpy pandas btrack joblib optuna traccuracy


For the purpose of our example using datasets from the cell tracking challenge we use the ctctools package to load the data.

.. code-block:: bash

    pip install -q -U --no-deps git+https://github.com/lowe-lab-ucl/ctc-tools.git@main

Setting Up Your Dataset
-----------------------

- Instructions on preparing your dataset for optimization, including formatting and necessary files.
- you need to have a dataset in the format of the cell tracking challenge.

::

    Each dataset has the following structure of subdirectories:

    ├── 0N
    ├── 0N_GT          
    │   ├── SEG
    │   └── TRA       

Where:

- `0N`: Represents the original image data of the N-th sequence.
- `0N_GT`: Contains the gold truth for the N-th sequence.
    - `SEG`: ground truth for the SEG measure. Contains segmented images in a sequence of tiff images.
    - `TRA`: ground truth for the DET and TRA measures. contains the ground truth tracking data in a sequence of tiff images and ``man_track.txt``.
- For a complete and detailed explanation, please refer to the official conventions outlined on the `Cell Tracking Challenge website <https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf>`_.

Running the Optimization
------------------------

- Step-by-step instructions on executing the script. The script can be found in the file `script.py <https://github.com/quantumjot/btrack/btrack/examples/example_optuna.py>`_.
- In order to use your own dataset with this code, you need to specify the path of your data. This is done by assigning the path to the `dataset_path` variable and the experiment/sequence number to the `experiment` variable as shown in the code snippet below:

.. code-block:: python

    dataset_path = 'path/to/dataset'  # Replace with the path to your dataset
    experiment = "0N"  # Replace with the experiment number
    
    gt_path1 = f'{dataset_path}/{experiment}_GT/TRA'  # This is the path to the ground truth data
    gt_path2 = f'{dataset_path}/{experiment}_GT/TRA/man_track.txt'  # This is the path to the man_track.txt file within the ground truth data
    gt_data = load_ctc_data(gt_path1, gt_path2) # Load ground truth data for the dataset
    
    #load dataset
    dataset = ctctools.load(dataset_path, experiment=experiment, scale=(1., 1., 1.))

- To change the number of trials to run during your optimization, you can modify the `n_trials` variable in the code snippet below:

.. code-block:: python

    # Define the number of trials
    n_trials = 10

The `n_trials` parameter determines the number of trials that the optimization process will run. Typically, the process converges after about 100 trials. However, it's recommended to initially set `n_trials` to a lower number, such as 10, to verify that the optimization process works correctly. For larger and more complex datasets, you may need to increase `n_trials` to ensure the process converges.
- The next step is to assign the parameter ranges you would like to search over. This is done by modifying the `param_ranges` variable in the code snippet below:


.. code-block:: python

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
        'max_lost': (1, 10, 'int'),
        'prob_not_assign': (0.0, 0.5),
        'max_search_radius': (50, 200, 'int'),
        'div_hypothesis': (0, 1, 'int')
    } 

.. warning::

    1. If the maximum value for 'max_lost' is more than 10, it can cause a semaphore object error.
    2. Do not change the range for 'div_hypothesis' as this functions as a boolean.
    3. We suggest to not change the ranges on the first run and to adjust depending on the results.

Next, adjust the `dataset_name` and `use_parallel_backend` parameters. Set `dataset_name` to the name of your dataset or trial. This name will be used as the index in the resulting CSV file and to name the configuration file. The `use_parallel_backend` variable is a boolean that controls whether the optimization process is parallelized. Set this to `True` for parallel processing, and `False` otherwise.

.. code-block:: python

    dataset_name = "dataset_name"  # Replace with the name of your dataset or trial

    # Run optimization
    study = perform_study(dataset_name, gt_data, dataset, param_ranges, n_trials, use_parallel_backend=True)

You also need to specify the path where the results will be saved by modifying the `results_path` variable in the following code snippet:

.. code-block:: python

    # Set path for the CSV file
    csv_file_path = 'results.csv'

    # Convert dictionary to DataFrame and save as CSV
    pd.DataFrame(optimized_params_per_dataset).to_csv(csv_file_path, index_label='Dataset')


To save the best parameters for future use, set the paths for the configuration files and write the parameters to these files:

.. code-block:: python

    # Set paths for the config files
    config_0_path = 'config_0.json'  # Parameters that led to the best MBC score
    config_1_path = 'config_1.json'  # Parameters that led to the best AOGM score

    # Write the best parameters to the config files
    write_best_params_to_config(config_0_path, best_trial_0.params)
    write_best_params_to_config(config_1_path, best_trial_1.params)

In this code snippet, `config_0_path` and `config_1_path` are the paths where the configuration files will be saved. `best_trial_0.params` and `best_trial_1.params` are the best parameters obtained from the optimization process that led to the best MBC and AOGM scores respectively. The `write_best_params_to_config` function writes these parameters to the specified config files.

Interpreting the Results
------------------------

- An interactive graphical interface with detailed results can be accessed by running `sqlite:///btrack.db` in the terminal. After executing this command, click on the link that appears in the terminal to view the interface.
- The output file, `results.csv`, includes the optimized parameters for each dataset, along with the AOGM and MBC metrics. The `.json` files also contain the optimized parameters for each dataset. These parameters can be used to enhance your bTrack configuration for improved tracking results.
- The AOGM (Acyclic Oriented Graph Metric) and MBC (Mitotic Branching Correctness) metrics are key indicators used to evaluate the accuracy of the tracking results. A lower AOGM value signifies higher overall tracking accuracy, while a higher MBC value indicates better accuracy in detecting mitotic events. Understanding these metrics can help you interpret the optimization results more effectively. For the AOGM metric, refer to: `Matula et al. 2015 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144959>`_. For the MBC metric, refer to: `Ulicna et al. 2021 <https://doi.org/10.3389/fcomp.2021.734559>`_.

.. note::

    The AOGM and MBC metrics are calculated using the `traccuracy` package. For more information on how these metrics are calculated, refer to the `traccuracy documentation <https://traccuracy.readthedocs.io/en/latest/>`_.

Troubleshooting
---------------

- What are some common issues that may arise during the optimization process?
- How can you address these issues?

Google Colab Example
--------------------

For a hands-on example, on a dataset from the cell tracking challenge, check out our Google Colab notebook:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/YourUsername/YourRepository/blob/main/YourNotebook.ipynb
