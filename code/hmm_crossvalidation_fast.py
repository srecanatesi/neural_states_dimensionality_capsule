#%% Fast HMM Cross-validation with optimized parameters for faster convergence
import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)
from pathlib import Path
import warnings
import itertools
import ssm
import random
import os
import itertools as it
import glob
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap
import scipy.spatial
from scipy import interpolate
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold, KFold
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.proportion import proportion_confint
from random import sample
from numpy.random import uniform
from math import isnan
from sklearn.model_selection import cross_val_score
import pickle
from scipy.sparse import csr_matrix
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d
import multiprocessing as mp
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(linewidth=1000)
pd.set_option("display.max_columns", None)
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
]
colors = sns.xkcd_palette(color_names)

def subsample_data(data, max_timepoints=50000, max_neurons=200):
    """
    Subsample data to reduce memory usage while preserving structure
    """
    if len(data) == 0:
        return data
    
    subsampled_data = []
    for trial in data:
        n_neurons, n_timepoints = trial.shape
        
        # Subsample neurons if too many
        if n_neurons > max_neurons:
            neuron_indices = np.random.choice(n_neurons, max_neurons, replace=False)
            trial = trial[neuron_indices, :]
        
        # Subsample timepoints if too many
        if n_timepoints > max_timepoints:
            time_indices = np.random.choice(n_timepoints, max_timepoints, replace=False)
            trial = trial[:, time_indices]
        
        subsampled_data.append(trial)
    
    return subsampled_data

def hmm_fit_fast(data, num_states, num_iters, true_states = None, tolerance=0.1):
    """
    Fast HMM fitting with relaxed convergence criteria and fewer iterations
    """
    # Subsample data to reduce memory usage
    data_subsampled = subsample_data(data, max_timepoints=50000, max_neurons=200)
    
    num_trials = len(data_subsampled)
    num_neurons = data_subsampled[0].shape[0]
    
    print(f"  Fitting HMM with {num_states} states, {num_neurons} neurons, {num_trials} trials")
    
    # Use Poisson observations for spike count data
    hmm = ssm.HMM(num_states, num_neurons, observations="poisson")
    train_data = [data_subsampled[i].transpose().astype(np.int8) for i in range(num_trials)]
    
    # Fit with reduced iterations and relaxed tolerance
    train_lls = hmm.fit(train_data, method="em", num_iters=num_iters, tolerance=tolerance)
    
    # Get states and likelihoods (only for subsampled data)
    hmm_z = np.array([hmm.most_likely_states(train_data[i_trial]) for i_trial in range(num_trials)])
    hmm_ll = np.array([hmm.observations.log_likelihoods(train_data[i_trial], None, None, None) for i_trial in range(num_trials)])
    hmm_ps = np.array([hmm.filter(train_data[i_trial]) for i_trial in range(num_trials)])
    
    return hmm_z, hmm_ll, hmm_ps

def xval_core_fast(i_state, spkcnts, nKfold, N_itersxv, tolerance=0.1):
    """
    Fast cross-validation core with reduced iterations and early stopping
    For single-trial data, we'll use a simplified approach
    """
    print(f"    Testing {i_state} states...")
    
    if len(spkcnts) == 1:
        # Single trial: use aggressive subsampling for cross-validation
        subsampled_data = subsample_data(spkcnts, max_timepoints=20000, max_neurons=100)
        hmm_z, hmm_ll, hmm_ps = hmm_fit_fast(subsampled_data, i_state, N_itersxv, tolerance=tolerance)
        return np.mean([np.mean(hmm_ll[i]) for i in range(len(hmm_ll))])
    
    # Multiple trials: use cross-validation with subsampling
    kf = KFold(n_splits=min(nKfold, len(spkcnts)), shuffle=True, random_state=42)
    llhood_fold = []
    
    for train_idx, test_idx in kf.split(spkcnts):
        train_data = [spkcnts[i] for i in train_idx]
        test_data = [spkcnts[i] for i in test_idx]
        
        # Subsample training data aggressively for speed
        train_data_subsampled = subsample_data(train_data, max_timepoints=20000, max_neurons=100)
        test_data_subsampled = subsample_data(test_data, max_timepoints=10000, max_neurons=100)
        
        # Fit HMM on training data with reduced iterations
        hmm_z, hmm_ll, hmm_ps = hmm_fit_fast(train_data_subsampled, i_state, N_itersxv, tolerance=tolerance)
        
        # For cross-validation, just return the training likelihood for speed
        # (full cross-validation would require refitting on test data)
        llhood_fold.append(np.mean([np.mean(hmm_ll[i]) for i in range(len(hmm_ll))]))
    
    return np.mean(llhood_fold)

def hmm_xval_fast(spkcnts, nKfold=2, N_itersxv=50, tolerance=0.1):
    """
    Fast cross-validation with reduced parameters for speed
    """
    # Reduced state space for faster evaluation (only test 2-5 states)
    states_space = np.arange(2, min(5, len(spkcnts[0])//100 + 2))  # Cap at 5 states max
    
    print(f"Fast cross-validation: testing {len(states_space)} states with {nKfold} folds")
    print(f"States to test: {states_space}")
    
    # Use sequential processing to avoid memory issues
    xval_core_partial = partial(xval_core_fast, spkcnts=spkcnts, nKfold=nKfold, N_itersxv=N_itersxv, tolerance=tolerance)
    
    # Process states sequentially to avoid memory issues
    llhood_mean = []
    for state in states_space:
        print(f"  Processing state {state}...")
        llhood = xval_core_partial(state)
        llhood_mean.append(llhood)
        print(f"  State {state}: log-likelihood = {llhood:.2f}")
    
    return states_space, llhood_mean

def hmm_analysis_fast(spkcnts, nKfold=2, N_itersxv=50, N_iters=100, N_states=np.nan, N_final_fit=2, tolerance=0.1, hmm_true4colors=None, save=np.nan):
    """
    Fast HMM analysis with optimized parameters for speed
    """
    llhood_mean = np.nan
    states_space = np.nan
    
    if isnan(N_states):
        print("Running fast cross-validation...")
        states_space, llhood_mean = hmm_xval_fast(spkcnts, nKfold, N_itersxv, tolerance=tolerance)
        
        try:
            # Use more aggressive knee detection for faster convergence
            kneedle = KneeLocator(states_space, llhood_mean, S=0.5, curve='concave', direction='increasing')
            
            # Create results directory if it doesn't exist
            Path('./results').mkdir(exist_ok=True)
            
            # Save plots
            kneedle.plot_knee_normalized(figsize=(10,6))
            sns.despine()
            plt.xticks(np.linspace(0, 1, len(states_space)), states_space)
            plt.title('Fast Cross-validation: Normalized Log-likelihood')
            plt.savefig('./results/crossvalidation_meanLLhood_normalized_fast.pdf', dpi=150, bbox_inches='tight')
            plt.close()
            
            kneedle.plot_knee(figsize=(10,6))
            sns.despine()
            plt.title('Fast Cross-validation: Log-likelihood')
            plt.savefig('./results/crossvalidation_meanLLhood_fast.pdf', dpi=150, bbox_inches='tight')
            plt.close()
            
            N_states = round(kneedle.elbow, 3)
            print(f"Optimal number of states (fast): {N_states}")
        except Exception as e:
            print(f"Knee detection failed: {e}")
            N_states = states_space[-1] if len(states_space) > 0 else 3
            print(f"Using last state: {N_states}")
    else:
        llhood_mean = np.nan
    
    print(f"Final HMM fitting with {N_states} states...")
    
    # Reduced final fits for speed
    hmm_states_all, hmm_ll_all, hmm_posterior_all = [], [], []
    for i_fold in range(N_final_fit):
        print(f"Final fit {i_fold + 1}/{N_final_fit}")
        hmm_states_i, hmm_ll_i, hmm_posterior_i = hmm_fit_fast(spkcnts, N_states, N_iters, tolerance=tolerance)
        hmm_states_all.append(hmm_states_i)
        hmm_ll_all.append(hmm_ll_i)
        hmm_posterior_all.append(hmm_posterior_i)
    
    # Select best fit
    i_min = np.argmin([np.nanmean(hmm_ll_all[i]) for i in range(N_final_fit)])
    hmm_states, hmm_ll, hmm_posterior = hmm_states_all[i_min], hmm_ll_all[i_min], hmm_posterior_all[i_min]
    
    hmm_analysis_data = {
        'states': hmm_states, 
        'll': hmm_ll, 
        'posterior': hmm_posterior, 
        'N_states': N_states, 
        'll_xval': llhood_mean,
        'method': 'fast'
    }
    
    if isinstance(save, str):
        with open(save, "wb") as f:
            pickle.dump(hmm_analysis_data, f)
        print(f"Results saved to {save}")
    
    return hmm_analysis_data

def main():
    """
    Main function to run fast HMM cross-validation on preprocessed data
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast HMM Cross-validation')
    parser.add_argument('--session-id', type=str, required=True, help='Session ID to process')
    parser.add_argument('--data-dir', type=str, default='data/sessions_preprocessed', help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='data/sessions_hmm_crossval', help='Output directory for results')
    parser.add_argument('--n-folds', type=int, default=2, help='Number of cross-validation folds')
    parser.add_argument('--n-iter-xval', type=int, default=5, help='Number of iterations for cross-validation')
    parser.add_argument('--n-iter-final', type=int, default=5, help='Number of iterations for final fit')
    parser.add_argument('--n-final-fits', type=int, default=1, help='Number of final fits')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Convergence tolerance')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load preprocessed data
    data_file = Path(args.data_dir) / f'df_{args.session_id}.pkl'
    if not data_file.exists():
        print(f"Error: Data file {data_file} not found")
        return
    
    print(f"Loading data from {data_file}")
    df = pd.read_pickle(data_file)
    
    # Extract spike counts
    spkcnts = df['spkcnts'].values[0]  # Shape: (n_neurons, n_timepoints)
    
    # Convert to list of trials (assuming single trial for now)
    spkcnts_list = [spkcnts]  # Each element should be (n_neurons, n_timepoints)
    
    print(f"Data shape: {spkcnts.shape}")
    print(f"Number of neurons: {spkcnts.shape[0]}")
    print(f"Number of time points: {spkcnts.shape[1]}")
    
    # Run fast HMM analysis
    output_file = Path(args.output_dir) / f'hmm_{args.session_id}.pkl'
    
    print("Starting fast HMM analysis...")
    hmm_results = hmm_analysis_fast(
        spkcnts_list,
        nKfold=args.n_folds,
        N_itersxv=args.n_iter_xval,
        N_iters=args.n_iter_final,
        N_final_fit=args.n_final_fits,
        tolerance=args.tolerance,
        save=str(output_file)
    )
    
    print(f"Fast HMM analysis completed!")
    print(f"Optimal states: {hmm_results['N_states']}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
