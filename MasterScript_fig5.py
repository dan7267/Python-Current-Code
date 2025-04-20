import numpy as np
import scipy as sp
from paradigm_setting import paradigm_setting
from simulate_adaptation import simulate_adaptation
from repeffects_fig4 import produce_slopes
import sys
from ExperimentalData import create_pattern
from itertools import product
np.set_printoptions(threshold=sys.maxsize)
from joblib import Parallel, delayed
import os
print("Number of CPU cores:", os.cpu_count())
n_jobs = 4

def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, sub_num)
    # pattern = (out['pattern'].T + np.random.randn(v, len(j), sub_num) * noise).T
    pattern = (out['pattern'].T + np.random.randn(v, len(j), sub_num) * noise).T
    v = pattern.shape[1]


    inds = {
        'cond1_p1': np.arange(0, pattern.shape[1], 4),  # 0, 4, 8, ...
        'cond1_p2': np.arange(1, pattern.shape[1], 4),  # 1, 5, 9, ...
        'cond2_p1': np.arange(2, pattern.shape[1], 4),  # 2, 6, 10, ...
        'cond2_p2': np.arange(3, pattern.shape[1], 4)   # 3, 7, 11, ...
    }

    # Stack the four condition slices along a new axis → shape (18, 4, 8, 200)
    pattern_split = np.stack([
        pattern[:, inds['cond1_p1'], :],
        pattern[:, inds['cond1_p2'], :],
        pattern[:, inds['cond2_p1'], :],
        pattern[:, inds['cond2_p2'], :]
    ], axis=1)

    # Reshape to (18, 32, 200) by merging the 4 and 8 dims → 4×8 = 32
    pattern_out = pattern_split.reshape(pattern.shape[0], 32, pattern.shape[2])

    return pattern_out

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, n_jobs):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    noise = 0.03
    N = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    #     for sub in range(sub_num)
    # )
    #y is an 18 x 32 x 200
    # y = np.array([results[sub] for sub in range(sub_num)])

    return produce_slopes(y, 1)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one simulation set")

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_slopes_one_simulation)(paradigm, model_type, sigma, a, b, n_jobs)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems]).flatten()

    results_dict = {
        key: 3 if x[0] < 0 and x[1] < 0 else 
             1 if x[0] > 0 and x[1] > 0 else 
             2 if x[0] <0 < x[1] else 
             4
        for key, x in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci.reshape(-1, 2))
        }
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat
    return results_dict

def produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs):
    """Returns key data structures and values to produce the final figure, doing so with multiple simulations at each parameter combination. For one model"""
    num_combinations = np.prod([len(v) for v in parameters.values()])
    results_comparison = np.zeros((num_combinations, 6))
    max_same, parameters_of_max = [], []
    no_max_same = 0

    parameter_list = list(product(*parameters.values()))
    parameter_indices = {
        (sigma, a, b): [parameters['sigma'].index(sigma), parameters['a'].index(a), parameters['b'].index(b)]
        for sigma, a, b, in parameter_list
    }
    def process_combination(index, combination):
        sigma, a, b = combination
        results_dict = produce_confidence_intervals(paradigm, model, sigma, a, b, n_jobs, n_simulations)
        results_match = [1 if results_dict[feature] == experimental_results[feature] else 0 for feature in results_dict]
        return index, results_match

    parallel_results = Parallel(n_jobs)(
        delayed(process_combination)(i, combination) for i, combination in enumerate(parameter_list)
    )
    #For one model, produces an array for each parameter combo 

    for index, results_match in parallel_results:
        results_comparison[index] = results_match
        current_max = sum(results_match)

        if current_max > no_max_same:
            no_max_same = current_max
            max_same = [results_match]
            parameters_of_max = [parameter_indices[parameter_list[index]]]
        elif current_max == no_max_same:
            max_same.append(results_match)
            parameters_of_max.append(parameter_indices[parameter_list[index]])
    key_variables = {
        'model_comparison' : results_comparison,
        'max_same' : np.unique(max_same, axis=0),
        'no_max_same' : no_max_same,
        'parameters_of_max' : parameters_of_max
    }
    return key_variables

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    fig_5_dict = {model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_array = np.zeros((6, 12))
    fig_5_sets = []
    for model, model_data in fig_5_dict.items():
        fig_5_sets.append(model_data['max_same'])
        for i, max_match in enumerate(model_data['max_same'][0]):
            fig_5_array[i, model - 1] = max_match
    print(fig_5_array, fig_5_sets)
    print(fig_5_dict)
    return fig_5_dict

parameters = {
    'sigma' : [0.2],
    'a' : [0.7],
<<<<<<< HEAD
    'b' : [0.2, 1.5]
=======
    'b' : [0.2]
>>>>>>> d83f6e8874ca7e5c1ce983432dd88572e59c8151
}

good_spread_parameters = {
    'sigma' : [0.1, 0.3, 0.7, 5, 11],
    'a' : [0.1, 0.4, 0.7, 0.9],
    'b' : [0.1, 0.5, 0.9, 1.5]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}

model_1_2 = {
    'global scaling' : 1,
    'local scaling' : 2,
    'remote scaling' : 3,
    'global sharpening' : 4
}

models = {
    'global scaling' : 1,
    'local scaling' : 2,
    'remote scaling' : 3,
    'global sharpening' : 4,
    'local sharpening' : 5,
    'remote sharpening' : 6,
    'global repulsion' : 7,
    'local repulsion' : 8,
    'remote repulsion' : 9,
    'global attraction' : 10,
    'local attraction' : 11,
    'remote attraction' : 12
}

paradigms = ['face', 'grating']        


experimental_face_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 3,
    'AMS' : 1,
    'AMA' : 1
}

experimental_grating_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 1,
    'AMS' : 3,
    'AMA' : 1
}

<<<<<<< HEAD
n_simulations = 20
=======
<<<<<<< HEAD
n_simulations = 2
=======
<<<<<<< HEAD
n_simulations = 50
=======
n_simulations = 20
>>>>>>> 27b350a95b62b63133589867f2cc611c0f8f99bc
>>>>>>> d83f6e8874ca7e5c1ce983432dd88572e59c8151
>>>>>>> a0f33a40d51c85cf4e278e621e4e785af550d644

producing_fig_5(good_spread_parameters, 'face', n_simulations, n_jobs)