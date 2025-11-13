#%%
# !pip install allensdk
import xarray as xr
import os
import itertools as it
from pathlib import Path
import numpy as np
import glob
import shutil
import pandas as pd
import pynwb
from pynwb import NWBHDF5IO
import warnings
warnings.simplefilter(action='ignore')

# Set up paths - support both container /data and local data/
data_dir_abs = Path('/data')
data_dir_rel = Path(__file__).resolve().parents[1] / 'data'

# Prefer relative path if it exists (for local development)
# Otherwise use absolute path (for container environments)
if data_dir_rel.exists() and data_dir_rel.is_dir():
    datafolder = data_dir_rel
elif data_dir_abs.exists() and data_dir_abs.is_dir():
    datafolder = data_dir_abs
else:
    # Default to relative for local development
    datafolder = data_dir_rel

# Output folder - use /results in container, results/ locally
results_dir_abs = Path('/results')
results_dir_rel = Path(__file__).resolve().parents[1] / 'results'

if results_dir_rel.exists() and results_dir_rel.is_dir():
    results_dir = results_dir_rel
elif results_dir_abs.exists() and results_dir_abs.is_dir():
    results_dir = results_dir_abs
else:
    # Default to relative for local development
    results_dir = results_dir_rel

savefolder = results_dir / 'sessions_preprocessed'
nwb_folder = datafolder / 'sessions_nwb'
unit_table_file = datafolder / 'unit_table_all.csv'
sessions_file = datafolder / 'sessions.csv'

# Create output directory
savefolder.mkdir(parents=True, exist_ok=True)

# Load sessions table to filter for functional_connectivity sessions
def load_functional_connectivity_sessions():
    """Load session IDs that are labeled as functional_connectivity"""
    if not sessions_file.exists():
        print(f"Warning: {sessions_file} not found, processing all NWB files")
        return None
    
    sessions_df = pd.read_csv(sessions_file)
    fc_sessions = sessions_df[sessions_df['session_type'] == 'functional_connectivity']
    # Use 'id' column (not 'ecephys_session_id')
    fc_session_ids = set(fc_sessions['id'].astype(str))
    print(f"Found {len(fc_session_ids)} functional_connectivity sessions")
    return fc_session_ids

#%%
def get_spikecounts_during_spontaneous_epochs(nwb_file, uID_list, bSize=0.5, binarize=False, dtype=None):
    """Get spike counts during spontaneous epochs from NWB file"""
    
    # Extract stimulus table from NWB intervals
    if hasattr(nwb_file, 'intervals') and 'spontaneous_presentations' in nwb_file.intervals:
        spontaneous_table = nwb_file.intervals['spontaneous_presentations']
        start_times = spontaneous_table.start_time[:] if hasattr(spontaneous_table.start_time, '__getitem__') else spontaneous_table.start_time
        stop_times = spontaneous_table.stop_time[:] if hasattr(spontaneous_table.stop_time, '__getitem__') else spontaneous_table.stop_time
        durations = stop_times - start_times
    else:
        print("No spontaneous presentations found in NWB file")
        return [], []
    
    # Ensure it was for long enough (1500 seconds = 25 minutes)
    iBlocks = np.where(durations > 1500)[0]
    nBlocks = len(iBlocks)
    print(f"Found {nBlocks} epochs longer than 1500 seconds")
    
    if nBlocks == 0:
        print("No spontaneous epochs longer than 1500 seconds found, skipping session")
        return [], []
    
    # Create mapping from unit ID to position in NWB units table
    unit_id_to_position = {}
    for i in range(len(nwb_file.units)):
        unit_data = nwb_file.units[i]
        unit_id = unit_data.index[0] if hasattr(unit_data, 'index') and len(unit_data.index) > 0 else i
        unit_id_to_position[unit_id] = i
    
    spikecount_list = []
    timecourse_list = []
    
    # Loop through spontaneous blocks
    for iEpoch in iBlocks:
        tStart = start_times[iEpoch]
        tStop = stop_times[iEpoch]
        duration = tStop - tStart
        
        # Bin spikes into windows to calculate simple FR vector for each neuron
        bin_edges = np.arange(tStart, tStop, bSize)
        starts = bin_edges[:-1]
        ends = bin_edges[1:]
        tiled_data = np.zeros((bin_edges.shape[0] - 1, len(uID_list)),
                              dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype)
        
        # Loop through each neuron
        for ii, unit_id in enumerate(uID_list):
            if unit_id in unit_id_to_position:
                # Get spike times for this unit from NWB using position
                unit_position = unit_id_to_position[unit_id]
                unit_data = nwb_file.units[unit_position]
                spike_times = unit_data['spike_times'].values[0] if hasattr(unit_data['spike_times'], 'values') else unit_data['spike_times']
                
                # Ignore invalid spike times
                pos = np.where(spike_times > 0)[0]
                data = spike_times[pos]
                
                # Ensure spike times are sorted
                sort_indices = np.argsort(data)
                start_positions = np.searchsorted(data, starts.flat, sorter=sort_indices)
                end_positions = np.searchsorted(data, ends.flat, side="right", sorter=sort_indices)
                counts = (end_positions - start_positions)
                
                tiled_data[:, ii].flat = counts > 0 if binarize else counts
            else:
                print(f"Warning: Unit ID {unit_id} not found in NWB file")
                tiled_data[:, ii].flat = 0
        
        # Save matrix to list
        spikecount_list.append(tiled_data.T)
        timecourse_list.append(bin_edges[:-1] + np.diff(bin_edges) / 2)
    
    return spikecount_list, timecourse_list

def midbins(x):
    return x[:-1] + (x[1] - x[0]) / 2

def find_permutation_seq(x, y):
    x_el = np.unique(x)
    y_el = np.unique(y)
    x_elpos = [np.nanmean([np.mean(np.where(row == el)[0]) for row in x]) for el in x_el]
    y_elpos = [np.nanmean([np.mean(np.where(row == el)[0]) for row in y]) for el in y_el]
    x_elids = np.argsort(x_elpos)
    y_elids = np.argsort(y_elpos)
    permutation = [np.where(x_elids == el)[0][0] for el in y_elids]
    return permutation

def hmm_fit(data, num_states, num_iters, true_states = None):
    import ssm
    num_trials = len(data)
    num_neurons = data[0].shape[0]
    # ini_state = np.zeros(num_states)
    # ini_state[0] = 1
    # init_dist = ssm.init_state_distns.FixedInitialStateDistribution(num_states, num_neurons, pi0=ini_state)
    hmm = ssm.HMM(num_states, num_neurons, observations="poisson")#, init_state_distn=init_dist)
    train_data = [data[i].transpose().astype(np.int8) for i in range(num_trials)]
    train_lls = hmm.fit(train_data, method="em", num_iters = num_iters)
    hmm_z = np.array([hmm.most_likely_states(train_data[i_trial]) for i_trial in range(num_trials)])
    hmm_ll = np.array([hmm.observations.log_likelihoods(train_data[i_trial], None, None, None) for i_trial in range(num_trials)])
    hmm_ps = np.array([hmm.filter(train_data[i_trial]) for i_trial in range(num_trials)])
    # hmm_ps = np.exp(hmm_ll) / np.sum(np.exp(hmm_ll), axis=2)[:, :, np.newaxis]
    return hmm_z, hmm_ll, hmm_ps

def sequence2midpoints(sequence):
    final_points = np.array(np.where(np.diff(sequence) != 0)[0])
    initial_points = final_points+1
    initial_points = np.insert(initial_points, 0, 0)
    final_points = np.append(final_points, len(sequence)-1)
    mid_points = np.round((initial_points + final_points) /2).astype('int')
    # This fixes initial and last state to have middle point respectively at beginning and end of array
    # mid_points[0] = 0
    # mid_points[-1] = len(sequence) - 1
    mid_points = np.insert(mid_points, 0, 0)
    mid_points = np.append(mid_points, len(sequence)-1)
    mid_values = sequence[mid_points]
    return mid_points, mid_values

def isininterval(x, a, b, y=None, axis = 0):
    if len(x.shape) > 0:
        x = np.swapaxes(x, 0, axis)
    if np.array(y == None).any():
        x = x[(x >= a) & (x <= b)]
    else:
        x = x[(y >= a) & (y <= b)]
    if len(x.shape) > 0:
        x = np.swapaxes(x, 0, axis)
    return x

def area2region(units, field):
    dict = {'Thalamus': ['LGd', 'LGn', 'LP', 'LD', 'POL', 'MD', 'VPL', 'PO', 'VPM', 'RT', 'MG', 'MGv', 'MGd', 'Eth', 'SGN', 'TH'],
           'others': ['RSP', 'OLF', 'BLA', 'ZI', 'grey'],
            'Hippocampus': ['DG', 'CA3', 'CA1', 'SUB', 'POST', 'ProS'],
            'FrontalCortex': ['ACA', 'MOs', 'PL', 'ILA', 'ORB', 'MOp', 'SSp'],
            'VisualCortex' : ['VISp', 'VISl', 'VISpm', 'VISam', 'VISrl', 'VISa', 'VISal', 'VIS', 'VISli', 'VISlm'],
            'Midbrain' : ['SCs', 'SCm', 'MRN', 'APN', 'PAG', 'MB'],
            'BasalGanglia' : ['CP', 'GPe', 'SNr', 'ACB', 'LS']}

    df = pd.DataFrame.from_dict(dict.items())
    df = df.explode(1)
    df = df.rename(columns= {0:'region', 1:'area'})
    df = df.merge(units, left_on = 'area', right_on = field)
    return df

def grpBySameConsecutiveItem(l, max_length = 15, min_length = 3, value=True):
    rv= []
    rv_idx = []
    last = None
    last_idx = None
    for i_elem, elem in enumerate(l):
        if last == None:
            last = [elem]
            last_idx = [i_elem]
            continue
        if (elem == last[0]) & (len(last) < max_length):
            last.append(elem)
            last_idx.append(i_elem)
            continue
        if (len(last) >= min_length) & (last[0]==value):
            rv.append(last)
            rv_idx.append(last_idx)
        last = [elem]
        last_idx = [i_elem]
    return rv, rv_idx

def classify_waveform(units_details):
    from sklearn.mixture import GaussianMixture
    X = units_details['waveform_duration'].values
    X = X.reshape(-1,1)
    gm = GaussianMixture(n_components=3, random_state=0, covariance_type='full').fit(X)
    clu = gm.predict(X)
    clu_p = np.max(gm.predict_proba(X), axis=1)

    ini_idx = (clu == np.argmin(gm.means_)) & (clu_p > 0.95)
    exc_idx = (clu == np.argsort(gm.means_)[1][0]) & (clu_p > 0.95)
    oth_idx = (clu == np.argmax(gm.means_)) & (clu_p > 0.95)
    units_details['EI_type'] = np.nan
    units_details['EI_type'][exc_idx] = 'Exc'
    units_details['EI_type'][ini_idx] = 'Ini'
    units_details['EI_type'][oth_idx] = 'Oth'
    return units_details

def optotagging_spike_counts(bin_edges, trials, session, units):
    time_resolution = np.mean(np.diff(bin_edges))
    spike_matrix = np.zeros((len(trials), len(bin_edges)-1, len(units)))
    for unit_idx, unit_id in enumerate(units.index.values):
        spike_times = session.spike_times[unit_id]
        for trial_idx, trial_start in enumerate(trials.start_time.values):
            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))
            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            binned_times, counts = np.unique(binned_times, return_counts=True)
            spike_matrix[trial_idx, binned_times, unit_idx] = counts
    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={'trial_id': trials.index.values, 'time_relative_to_stimulus_onset': (bin_edges[:-1]+bin_edges[1:])/2, 'unit_id': units.index.values},
        dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id'])

def classify_10mspulses(units_details, nwb_file):
    # Extract optogenetic stimulation epochs from NWB file
    units_details['opto_10ms'] = np.nan
    
    try:
        # Get optogenetic stimulation epochs
        if hasattr(nwb_file, 'processing') and 'optotagging' in nwb_file.processing:
            opto_module = nwb_file.processing['optotagging']
            if 'optogenetic_stimulation' in opto_module.data_interfaces:
                stim_epochs = opto_module.data_interfaces['optogenetic_stimulation'].to_dataframe()
                
                # Filter for 10ms pulses (9-20ms duration)
                ten_ms_pulses = stim_epochs[(stim_epochs['duration'] > 0.009) & (stim_epochs['duration'] < 0.02)]
                
                if len(ten_ms_pulses) > 0:
                    print(f'Found {len(ten_ms_pulses)} optogenetic 10ms pulses')
                    
                    # Get genotype from subject information
                    genotype = 'wt/wt'  # Default
                    if hasattr(nwb_file, 'subject'):
                        subject = nwb_file.subject
                        if hasattr(subject, 'genotype'):
                            genotype = subject.genotype
                    
                    # Extract first 3 characters of genotype
                    genotype_short = genotype[:3] if len(genotype) >= 3 else genotype
                    
                    # For now, we'll need to implement the spike analysis
                    # This is a simplified version that would need the full optotagging_spike_counts implementation
                    print(f'Genotype: {genotype} -> {genotype_short}')
                    print('Note: Full optogenetic analysis requires implementing spike count analysis around stimulation times')
                    print('This would involve extracting spike times and analyzing responses to optogenetic stimulation')
                else:
                    print('No 10ms optogenetic pulses found')
            else:
                print('No optogenetic stimulation data found in NWB file')
        else:
            print('No optotagging module found in NWB file')
    except Exception as e:
        print(f'Error in optogenetic analysis: {e}')
    
    return units_details

def create_units_dataframe(nwb_file, session_id):
    """Create units dataframe from NWB file similar to Allen SDK format"""
    units = nwb_file.units
    unit_data = []
    
    for i, unit in enumerate(units):
        unit_info = {}
        
        # Basic unit information - use the actual unit ID from NWB DataFrame index
        unit_info['unit_id'] = unit.index[0] if hasattr(unit, 'index') and len(unit.index) > 0 else i
        unit_info['session_id'] = session_id
        
        # Spike times - access correctly from the units table
        if hasattr(unit, 'spike_times') and unit.spike_times is not None:
            spike_times = unit.spike_times
            if isinstance(spike_times, pd.Series):
                spike_times = spike_times.values[0] if len(spike_times.values) > 0 else np.array([])
            unit_info['spike_times'] = spike_times
            unit_info['num_spikes'] = len(spike_times)
        else:
            unit_info['spike_times'] = np.array([])
            unit_info['num_spikes'] = 0
        
        # Waveform information
        unit_info['waveform_mean'] = unit['waveform_mean'].values[0] if hasattr(unit, 'waveform_mean') else None
        unit_info['waveform_std'] = None  # Not available in NWB
        unit_info['waveform_duration'] = unit['waveform_duration'].values[0] if hasattr(unit, 'waveform_duration') else None
        unit_info['waveform_halfwidth'] = unit['waveform_halfwidth'].values[0] if hasattr(unit, 'waveform_halfwidth') else None
        
        # Electrode information
        unit_info['electrodes'] = None  # Not directly available
        unit_info['peak_channel_id'] = unit['peak_channel_id'].values[0] if hasattr(unit, 'peak_channel_id') else None
        
        # Quality metrics
        unit_info['quality'] = unit['quality'].values[0] if hasattr(unit, 'quality') else None
        unit_info['isi_violations'] = unit['isi_violations'].values[0] if hasattr(unit, 'isi_violations') else None
        unit_info['firing_rate'] = unit['firing_rate'].values[0] if hasattr(unit, 'firing_rate') else None
        unit_info['amplitude_cutoff'] = unit['amplitude_cutoff'].values[0] if hasattr(unit, 'amplitude_cutoff') else None
        unit_info['presence_ratio'] = unit['presence_ratio'].values[0] if hasattr(unit, 'presence_ratio') else None
        unit_info['isolation_distance'] = unit['isolation_distance'].values[0] if hasattr(unit, 'isolation_distance') else None
        unit_info['l_ratio'] = unit['l_ratio'].values[0] if hasattr(unit, 'l_ratio') else None
        unit_info['d_prime'] = unit['d_prime'].values[0] if hasattr(unit, 'd_prime') else None
        unit_info['snr'] = unit['snr'].values[0] if hasattr(unit, 'snr') else None
        
        # Additional metrics
        unit_info['halfwidth'] = unit['waveform_halfwidth'].values[0] if hasattr(unit, 'waveform_halfwidth') else None
        unit_info['PT_ratio'] = unit['PT_ratio'].values[0] if hasattr(unit, 'PT_ratio') else None
        unit_info['repolarization_slope'] = unit['repolarization_slope'].values[0] if hasattr(unit, 'repolarization_slope') else None
        unit_info['recovery_slope'] = unit['recovery_slope'].values[0] if hasattr(unit, 'recovery_slope') else None
        unit_info['amplitude'] = unit['amplitude'].values[0] if hasattr(unit, 'amplitude') else None
        
        # Depth information
        unit_info['depth'] = None  # Not directly available in NWB
        
        # Probe information
        unit_info['probe_id'] = None  # Not directly available
        unit_info['probe_description'] = None  # Not directly available
        
        # Location information
        unit_info['location'] = None
        unit_info['ecephys_structure_acronym'] = None
        unit_info['ecephys_structure_id'] = None
        
        # CCF coordinates
        unit_info['anterior_posterior_ccf_coordinate'] = None
        unit_info['dorsal_ventral_ccf_coordinate'] = None
        unit_info['left_right_ccf_coordinate'] = None
        
        # Layer and area
        unit_info['layer'] = None
        unit_info['area'] = None
        
        unit_data.append(unit_info)
    
    # Create DataFrame
    df_units = pd.DataFrame(unit_data)
    return df_units

def preprocess_data(nwb_file, session_id, region=None, area=None, layer=0., N_min_neurons=1, bin_size=0.005):
    """Preprocess data from NWB file similar to original preprocessing"""
    import time
    preprocess_start = time.time()
    
    print(f"\n[PREPROCESS {session_id}] Starting preprocessing pipeline")
    print(f"[PREPROCESS {session_id}] Parameters: region={region}, area={area}, layer={layer}, bin_size={bin_size}")
    print(f"[PREPROCESS {session_id}] {'-'*50}")
    
    # Create units dataframe from NWB file
    print(f"[PREPROCESS {session_id}] Step 1: Extracting units from NWB file...")
    step_start = time.time()
    units = create_units_dataframe(nwb_file, session_id)
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Extracted {len(units)} units in {step_time:.1f}s")
    
    if units.empty:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: No units data available")
        return None
    
    # Load unit details from CSV and merge with NWB data
    print(f"[PREPROCESS {session_id}] Step 2: Loading and merging unit details...")
    step_start = time.time()
    if unit_table_file.exists():
        print(f"[PREPROCESS {session_id}]   Loading unit details from {unit_table_file}")
        units_details = pd.read_csv(unit_table_file)
        units_details = units_details.rename(columns={'Unnamed: 0': 'unit_id'})
        
        print(f"[PREPROCESS {session_id}]   Units details shape: {units_details.shape}")
        print(f"[PREPROCESS {session_id}]   Units dataframe shape before merge: {units.shape}")
        common_ids = len(set(units['unit_id']) & set(units_details['unit_id']))
        print(f"[PREPROCESS {session_id}]   Common unit_ids: {common_ids}")
        
        # Classify waveforms and optogenetic responses on units_details before merge
        print(f"[PREPROCESS {session_id}]   Classifying waveforms...")
        if 'waveform_duration' in units_details.columns:
            units_details = classify_waveform(units_details)
            print(f"[PREPROCESS {session_id}]   [OK] Waveform classification complete")
        else:
            units_details['EI_type'] = 'Unknown'
            print(f"[PREPROCESS {session_id}]   ⚠ No waveform_duration column, skipping classification")
        
        # Classify optogenetic responses (for NWB files, this will set all to NaN)
        print(f"[PREPROCESS {session_id}]   Classifying optogenetic responses...")
        units_details = classify_10mspulses(units_details, nwb_file)
        
        # Merge on unit_id
        print(f"[PREPROCESS {session_id}]   Merging unit data...")
        units = pd.merge(units, units_details, on=['unit_id'], how='inner')
        print(f"[PREPROCESS {session_id}]   Units dataframe shape after merge: {units.shape}")
        
        if units.shape[0] == 0:
            print(f"[PREPROCESS {session_id}]   ⚠ WARNING: No matching units found after merge, using basic unit data")
            units['EI_type'] = 'Unknown'
            units['opto_10ms'] = np.nan
    else:
        print(f"[PREPROCESS {session_id}]   ⚠ WARNING: No unit_table_all.csv found, using basic unit data")
        units['EI_type'] = 'Unknown'
        units['opto_10ms'] = np.nan
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 2 completed in {step_time:.1f}s")
    
    # Add region information if ecephys_structure_acronym exists
    print(f"[PREPROCESS {session_id}] Step 3: Assigning region and area information...")
    step_start = time.time()
    ecephys_col = None
    if 'ecephys_structure_acronym_y' in units.columns:
        ecephys_col = 'ecephys_structure_acronym_y'
    elif 'ecephys_structure_acronym_x' in units.columns:
        ecephys_col = 'ecephys_structure_acronym_x'
    elif 'ecephys_structure_acronym' in units.columns:
        ecephys_col = 'ecephys_structure_acronym'
    
    if ecephys_col is not None:
        valid_acronyms = units[ecephys_col].dropna()
        if len(valid_acronyms) > 0:
            # Use the original ecephys_structure_acronym as area
            units['area'] = units[ecephys_col]
            
            # Create region mapping manually
            region_mapping = {
                'LGd': 'Thalamus', 'LGn': 'Thalamus', 'LP': 'Thalamus', 'LD': 'Thalamus', 'POL': 'Thalamus', 
                'MD': 'Thalamus', 'VPL': 'Thalamus', 'PO': 'Thalamus', 'VPM': 'Thalamus', 'RT': 'Thalamus', 
                'MG': 'Thalamus', 'MGv': 'Thalamus', 'MGd': 'Thalamus', 'Eth': 'Thalamus', 'SGN': 'Thalamus', 'TH': 'Thalamus',
                'RSP': 'others', 'OLF': 'others', 'BLA': 'others', 'ZI': 'others', 'grey': 'others',
                'DG': 'Hippocampus', 'CA3': 'Hippocampus', 'CA1': 'Hippocampus', 'SUB': 'Hippocampus', 'POST': 'Hippocampus', 'ProS': 'Hippocampus',
                'ACA': 'FrontalCortex', 'MOs': 'FrontalCortex', 'PL': 'FrontalCortex', 'ILA': 'FrontalCortex', 'ORB': 'FrontalCortex', 'MOp': 'FrontalCortex', 'SSp': 'FrontalCortex',
                'VISp': 'VisualCortex', 'VISl': 'VisualCortex', 'VISpm': 'VisualCortex', 'VISam': 'VisualCortex', 'VISrl': 'VisualCortex', 
                'VISa': 'VisualCortex', 'VISal': 'VisualCortex', 'VIS': 'VisualCortex', 'VISli': 'VisualCortex', 'VISlm': 'VisualCortex',
                'SCs': 'Midbrain', 'SCm': 'Midbrain', 'MRN': 'Midbrain', 'APN': 'Midbrain', 'PAG': 'Midbrain', 'MB': 'Midbrain',
                'CP': 'BasalGanglia', 'GPe': 'BasalGanglia', 'SNr': 'BasalGanglia', 'ACB': 'BasalGanglia', 'LS': 'BasalGanglia'
            }
            
            # Map regions
            units['region'] = units[ecephys_col].map(region_mapping).fillna('Unknown')
            unique_regions = units['region'].unique()
            print(f"[PREPROCESS {session_id}]   Found {len(unique_regions)} unique regions: {', '.join(unique_regions)}")
        else:
            print(f"[PREPROCESS {session_id}]   ⚠ WARNING: All ecephys_structure_acronym values are None, skipping region assignment")
            units['region'] = 'Unknown'
            units['area'] = 'Unknown'
    else:
        print(f"[PREPROCESS {session_id}]   ⚠ WARNING: No ecephys_structure_acronym column found, skipping region assignment")
        units['region'] = 'Unknown'
        units['area'] = 'Unknown'
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 3 completed in {step_time:.1f}s")
    
    # Set index
    units = units.set_index('unit_id')
    
    # Filter by region
    print(f"[PREPROCESS {session_id}] Step 4: Filtering units by region/area/layer...")
    step_start = time.time()
    if region is not None:
        unitsregion = units[units['region'] == region]
        print(f"[PREPROCESS {session_id}]   Filtered by region '{region}': {len(unitsregion)} units")
    else:
        unitsregion = units
        print(f"[PREPROCESS {session_id}]   No region filter: {len(unitsregion)} units")
    
    if area is not None:
        unitsregion = unitsregion[unitsregion['area'] == area]
        print(f"[PREPROCESS {session_id}]   Filtered by area '{area}': {len(unitsregion)} units")
    
    if layer == 0:
        unitssel = unitsregion
    else:
        unitssel = unitsregion[unitsregion['cortical_layer'] == layer]
        print(f"[PREPROCESS {session_id}]   Filtered by layer {layer}: {len(unitssel)} units")

    print(f"[PREPROCESS {session_id}]   Final selected units: {len(unitssel)}")
    if unitssel.shape[0] < N_min_neurons:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: Number of neurons ({unitssel.shape[0]}) below minimum ({N_min_neurons})")
        return None
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 4 completed in {step_time:.1f}s")

    # Extract spike counts during spontaneous epochs
    print(f"[PREPROCESS {session_id}] Step 5: Extracting spike counts from spontaneous epochs...")
    step_start = time.time()
    spkspont = get_spikecounts_during_spontaneous_epochs(nwb_file, unitssel.index.values, bSize=bin_size)
    
    if not spkspont[0]:  # No spontaneous epochs found
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: No spontaneous epochs found, skipping session")
        return None
    
    num_epochs = len(spkspont[0])
    total_timepoints = sum([epoch.shape[1] for epoch in spkspont[0]])
    print(f"[PREPROCESS {session_id}]   Found {num_epochs} spontaneous epochs")
    print(f"[PREPROCESS {session_id}]   Total timepoints: {total_timepoints}")
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 5 completed in {step_time:.1f}s")
    
    print(f"[PREPROCESS {session_id}] Step 6: Concatenating spike counts and extracting behavioral data...")
    step_start = time.time()
    spkcnts = np.concatenate(spkspont[0], axis=1)
    all_times = np.concatenate([spkspont[1][i] for i in range(len(spkspont[1]))])
    print(f"[PREPROCESS {session_id}]   Concatenated spike counts shape: {spkcnts.shape} (neurons x timepoints)")
    
    # Extract running speed (if available)
    running = np.nan
    if hasattr(nwb_file, 'processing') and 'running' in nwb_file.processing:
        running_module = nwb_file.processing['running']
        if 'running_speed' in running_module.data_interfaces:
            running_data = running_module.data_interfaces['running_speed']
            if hasattr(running_data, 'data') and hasattr(running_data, 'timestamps'):
                running_times = running_data.timestamps[:]
                running_values = running_data.data[:]
                # Interpolate to match spike times
                running = np.interp(all_times, running_times, running_values)
                print(f"[PREPROCESS {session_id}]   [OK] Extracted running speed data")
    else:
        print(f"[PREPROCESS {session_id}]   No running speed data available")
    
    # Extract pupil data (if available)
    pupil = np.nan
    if hasattr(nwb_file, 'processing') and 'filtered_gaze_mapping' in nwb_file.processing:
        gaze_module = nwb_file.processing['filtered_gaze_mapping']
        if 'pupil_area' in gaze_module.data_interfaces:
            pupil_data = gaze_module.data_interfaces['pupil_area']
            if hasattr(pupil_data, 'data') and hasattr(pupil_data, 'timestamps'):
                pupil_times = pupil_data.timestamps[:]
                pupil_values = pupil_data.data[:]
                # Interpolate to match spike times
                pupil = np.interp(all_times, pupil_times, pupil_values)
                print(f"[PREPROCESS {session_id}]   [OK] Extracted pupil area data")
    else:
        print(f"[PREPROCESS {session_id}]   No pupil area data available")
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 6 completed in {step_time:.1f}s")
    
    # Prepare output data
    print(f"[PREPROCESS {session_id}] Step 7: Preparing output dataframe...")
    step_start = time.time()
    EI_type = unitssel['EI_type'].values if 'EI_type' in unitssel.columns else np.array(['Unknown'] * len(unitssel))
    opto_10ms = unitssel['opto_10ms'].values if 'opto_10ms' in unitssel.columns else np.array([np.nan] * len(unitssel))
    
    # Use ecephys_structure_acronym for areas (this comes from the merge with unit_table_all.csv)
    if 'ecephys_structure_acronym' in unitssel.columns:
        areas_values = unitssel['ecephys_structure_acronym'].values
    elif 'area' in unitssel.columns:
        areas_values = unitssel['area'].values
    else:
        areas_values = np.array(['Unknown'] * len(unitssel))
    
    layers_values = unitssel['cortical_layer'].values if 'cortical_layer' in unitssel.columns else np.array([0] * len(unitssel))
    
    df = pd.DataFrame({
        'session_id': session_id, 
        'stimulus': 'spontaneous', 
        'region': region, 
        'area': area, 
        'layer': layer,
        'epoch': 'all', 
        'state': 'all', 
        'spkcnts': [spkcnts], 
        'times': [all_times], 
        'EI_type': [EI_type], 
        'opto_10ms': [opto_10ms], 
        'areas': [areas_values], 
        'layers': [layers_values],
        'running': [running], 
        'pupil': [pupil], 
        'N_trials': np.nan
    })
    step_time = time.time() - step_start
    print(f"[PREPROCESS {session_id}] [OK] Step 7 completed in {step_time:.1f}s")
    
    total_preprocess_time = time.time() - preprocess_start
    print(f"\n[PREPROCESS {session_id}] {'='*50}")
    print(f"[PREPROCESS {session_id}] [OK] PREPROCESSING COMPLETE")
    print(f"[PREPROCESS {session_id}] Total preprocessing time: {total_preprocess_time:.1f}s ({total_preprocess_time/60:.1f} minutes)")
    print(f"[PREPROCESS {session_id}] Output shape: {df.shape}")
    print(f"[PREPROCESS {session_id}] {'='*50}\n")
    
    return df

#%%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=str, help='Session ID (e.g., "766640955") or index (integer)')
    parser.add_argument('--all-sessions', action='store_true', 
                       help='Process all sessions, not just functional_connectivity')
    args = parser.parse_args()
    session_arg = args.session
    process_all_sessions = args.all_sessions

#%%
    bin_size = 0.005
    region = 'VisualCortex'
    area = None  # 'VISp'
    layers = [0]  # np.arange(7)
    regions_of_interest = ['Thalamus', 'others', 'Hippocampus', 'FrontalCortex', 'VisualCortex', 'Midbrain', 'BasalGanglia']
    areas_of_interest = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam', 'LGn', 'LGd', 'CA1', 'CA3', 'DG']
    
    # Load functional connectivity sessions
    fc_session_ids = load_functional_connectivity_sessions()
    
    # Find NWB files and filter for functional_connectivity sessions
    all_nwb_files = list(nwb_folder.glob('*.nwb'))
    if not all_nwb_files:
        print("No NWB files found in", nwb_folder)
        exit(1)
    
    # Filter NWB files based on session type
    nwb_files = []
    if process_all_sessions:
        print("Processing all available NWB files (--all-sessions flag used)")
        nwb_files = all_nwb_files
    else:
        # Filter NWB files to only include functional_connectivity sessions
        for nwb_file in all_nwb_files:
            session_id = nwb_file.stem.replace('session_', '')
            if fc_session_ids is None or session_id in fc_session_ids:
                nwb_files.append(nwb_file)
    
    if not nwb_files:
        print(f"No functional_connectivity NWB files found. Available files: {len(all_nwb_files)}")
        print("Available NWB files:")
        for nwb_file in all_nwb_files:
            session_id = nwb_file.stem.replace('session_', '')
            print(f"  {session_id}")
        print("\nFunctional connectivity session IDs from CSV:")
        if fc_session_ids:
            for session_id in sorted(list(fc_session_ids)):
                print(f"  {session_id}")
        print("\nTo process functional_connectivity sessions, you need to download the corresponding NWB files first.")
        exit(1)
    
    print(f"Found {len(nwb_files)} functional_connectivity NWB files out of {len(all_nwb_files)} total files")
    print(f"NWB folder: {nwb_folder}")
    print(f"Output folder: {savefolder}")
    
    # Determine if session_arg is a session_id or an index
    # First, check if it's a session_id by looking for matching NWB file
    session_id_str = str(session_arg)
    nwb_file_path = nwb_folder / f'session_{session_id_str}.nwb'
    
    print(f"\n[PREPROCESS] Determining session from argument: '{session_arg}'")
    if nwb_file_path.exists():
        # It's a session_id and the file exists
        session_id = session_id_str
        file_size_mb = nwb_file_path.stat().st_size / (1024 * 1024)
        print(f"[PREPROCESS] [OK] Found NWB file: {nwb_file_path.name} ({file_size_mb:.1f} MB)")
        print(f"[PREPROCESS] Processing session_id: {session_id}")
    else:
        # Try to parse as integer index
        try:
            i_iterator = int(session_arg)
            if i_iterator < 0 or i_iterator >= len(nwb_files):
                print(f"[PREPROCESS] [ERROR] ERROR: Session index {i_iterator} out of range. Available functional_connectivity sessions: {len(nwb_files)}")
                print(f"[PREPROCESS] Available NWB files:")
                for nwb_file in nwb_files:
                    print(f"[PREPROCESS]   {nwb_file.stem.replace('session_', '')}")
                exit(1)
            nwb_file_path = nwb_files[i_iterator]
            session_id = nwb_file_path.stem.replace('session_', '')
            file_size_mb = nwb_file_path.stat().st_size / (1024 * 1024)
            print(f"[PREPROCESS] [OK] Using index {i_iterator} -> {nwb_file_path.name} ({file_size_mb:.1f} MB)")
            print(f"[PREPROCESS] Processing functional_connectivity session {i_iterator + 1}/{len(nwb_files)}")
            print(f"[PREPROCESS] Session ID: {session_id}")
        except ValueError:
            # Not a valid integer and file doesn't exist
            print(f"[PREPROCESS] [ERROR] ERROR: NWB file not found: {nwb_file_path}")
            print(f"[PREPROCESS] Available NWB files:")
            for nwb_file in nwb_files:
                print(f"[PREPROCESS]   {nwb_file.stem.replace('session_', '')}")
            exit(1)
    
    # Load NWB file
    print(f"\n[PREPROCESS {session_id}] Loading NWB file: {nwb_file_path}")
    import time
    load_start = time.time()
    try:
        io = NWBHDF5IO(str(nwb_file_path), mode='r')
        nwb_file = io.read()
        load_time = time.time() - load_start
        file_size_mb = nwb_file_path.stat().st_size / (1024 * 1024)
        print(f"[PREPROCESS {session_id}] [OK] NWB file loaded ({file_size_mb:.1f} MB) in {load_time:.1f}s")
    except Exception as e:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR loading NWB file {nwb_file_path}: {e}")
        exit(1)
    
    # Process data
    df = preprocess_data(nwb_file, session_id, region=region, area=area, layer=layers[0], bin_size=bin_size)
    
    # Save results
    if df is not None:
        print(f"\n[PREPROCESS {session_id}] Saving preprocessed data...")
        save_start = time.time()
        output_file = savefolder / f'df_{session_id}.pkl'
        df.to_pickle(output_file)
        save_time = time.time() - save_start
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"[PREPROCESS {session_id}] [OK] Saved preprocessed data: {output_file}")
        print(f"[PREPROCESS {session_id}]   - File size: {output_size_mb:.2f} MB")
        print(f"[PREPROCESS {session_id}]   - DataFrame shape: {df.shape}")
        print(f"[PREPROCESS {session_id}]   - Spike counts shape: {df['spkcnts'].values[0].shape}")
        print(f"[PREPROCESS {session_id}]   - Time points: {len(df['times'].values[0])}")
        print(f"[PREPROCESS {session_id}]   - Save time: {save_time:.1f}s")
    else:
        print(f"[PREPROCESS {session_id}] [ERROR] ERROR: No data processed for this session")
    
    # Clean up
    io.close()
    print(f"[PREPROCESS {session_id}] [OK] NWB file closed")

# %%