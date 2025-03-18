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
n_jobs = 5

def produce_statistics_one_simulation(paradigm, model_type, sigma, a, b):
    """Produces the slopes for one simulation at one parameter combination"""
    v = 200
    X = np.pi #Stimulus Dimension
    cond1 = 1/4*X #Condition 1 of paradigm
    cond2 = 3/4*X #Condition 2 of paradigm
    sub_num = 18 #The number of subjects. Keep at 18
    noise = 0.03 
    N = 8
    y = {}
    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2) #We don't use winning_params, we use sigma, a, b
    for sub in range(sub_num):
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N)
        pattern = (out['pattern'].T + np.random.randn(v, len(j)) * noise).T
        v = pattern.shape[1]
        if paradigm == 'face':
            #print("Simulating face data")
            #The core function that generates the initial and repeated patterns - outputs: [presentation x voxel] matrix
            #Now group trials according to conditions and presentations
            cond1_p = {
                1: pattern[::4, :v],  # Rows 1:4:end (0-based index)
                2: pattern[1::4, :v]  # Rows 2:4:end (0-based index)
            }
            cond2_p = {
                1: pattern[2::4, :v],  # Rows 3:4:end (0-based index)
                2: pattern[3::4, :v]  # Rows 4:4:end (0-based index)
            }
        elif paradigm == 'grating':
            #print("Simulating grating data")
            cond1_p = {
                1: pattern[ind['cond1_p1'], :v],  # Using the indices for condition 1, part 1
                2:pattern[ind['cond1_p3'], :v]  # Using the indices for condition 1, part 2
            }
            cond2_p = {
                1: pattern[ind['cond2_p1'], :v],  # Using the indices for condition 2, part 1
                2: pattern[ind['cond2_p3'], :v]  # Using the indices for condition 2, part 2
            }
        y[sub] = np.vstack([cond1_p[1], cond1_p[2], cond2_p[1], cond2_p[2]])
    return produce_slopes(y,1)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_statistics_one_simulation)(paradigm, model_type, sigma, a, b)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    print(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems]).flatten()

    results_dict = {key: 3 if x[0] < 0 and x[1] < 0 else 1 if x[0] > 0 and x[1] > 0 else 2 if x[0] <0 < x[1] else 4
                    for key, x in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci.reshape(-1, 2))}
    return results_dict

def run_simulation(model, parameters, paradigm, experimental_results, n_simulations, n_jobs):
    """Returns key data structures and values to produce the final figure, doing so with multiple simulations at each parameter combination"""
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
        print(results_dict)
        results_match = [1 if results_dict[feature] == experimental_results[feature] else 0 for feature in results_dict]
        return index, results_match

    parallel_results = Parallel(n_jobs)(
        delayed(process_combination)(i, combination) for i, combination in enumerate(parameter_list)
    )

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

    return {
        'model_comparison' : results_comparison,
        'max_same' : np.unique(max_same, axis=0),
        'no_max_same' : no_max_same,
        'parameters_of_max' : parameters_of_max
    }

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    fig_5_dict = {model: run_simulation(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_array = np.zeros((6, 12))
    fig_5_sets = []
    for model, model_data in fig_5_dict.items():
        fig_5_sets.append(model_data['max_same'])
        for i, max_match in enumerate(model_data['max_same'][0]):
            fig_5_array[i, model - 1] = max_match
    print(fig_5_array, fig_5_sets)
    return fig_5_dict

parameters = {
    'sigma' : [0.2, 1.0],
    'a' : [0.2, 0.7],
    'b' : [0.2, 1.5]
}

good_spread_parameters = {
    'sigma' : [0.1, 2, 11],
    'a' : [0.1, 0.5, 0.9],
    'b' : [0.1, 0.7, 1.5]
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

n_simulations = 2

producing_fig_5(good_spread_parameters, 'face', n_simulations, n_jobs)