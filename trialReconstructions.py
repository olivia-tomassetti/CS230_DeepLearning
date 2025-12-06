# Analyze trained model performance on test data
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import json

# Open trial length dictionary 
with open('trial_length_map.json','r') as f:
    trial_length_map = json.load(f)
    
# Load test data
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
starts_test = np.load('starts_test.npy', allow_pickle=True)
trialIDs = np.load('trial_ids_train.npy', allow_pickle=True)
trialIDs_test = np.load('trial_ids_test.npy', allow_pickle=True)
predictions = np.load('predictions.npz', allow_pickle=True)
history = np.load('training_history.npz', allow_pickle=True)

# Helper functions ----------------------------------------------------------------------------
def reassembleTrials(predictionProbs, starts, Tx, trialLength):
    """
    predictionProbs: (n_windows,) or (n_windows, ) floats in [0,1] - predicted probability for each window element or
                if pred per-timestep: (n_windows, Tx) probabilities
    starts: (n_windows,) start indices (int) relative to trial (0-based)
    Tx: window length
    trialLength: total length of trial in samples (optional - used to size output)
    Returns: avgerage probabilities 
    """
    # Convert predictions and starts to numpy arrays
    preds = np.asarray(predictionProbs)
    starts = np.asarray(starts, dtype=int)
    
    L = int(trialLength) # length of the original trial
    acc = np.zeros(L, dtype=float) # initialize accumulated probabilities
    cnt = np.zeros(L, dtype=float) # initialize counts of contributions per timestep
    
    # Assume per window per timestep predictions
    window_idx = (starts[:, None] + np.arange(Tx)[None, :]).ravel()  
    values = preds.ravel()                                             

    # Accumulate predicted probabilities at each sample index
    np.add.at(acc, window_idx, values)
    # Add increment counts at each same index
    np.add.at(cnt, window_idx, 1)

    # Compute averaged probability per-sample, avoiding divide-by-zero
    # 'Return a full array with the same shape and type as a given array.'
    avg = np.full_like(acc, np.nan, dtype=float)
    nonzero = cnt > 0
    avg[nonzero] = acc[nonzero] / cnt[nonzero]

    return avg, cnt

def reassembleTrialsMultiClass(predictionProbs, starts, Tx, trialLength):
    """
    predictionProbs: (n_windows,) or (n_windows, ) floats in [0,1] - predicted probability for each window element or
                if pred per-timestep: (n_windows, Tx) probabilities
    starts: (n_windows,) start indices (int) relative to trial (0-based)
    Tx: window length
    trial_length: total length of trial in samples (optional - used to size output)
    Returns: avgerage probabilities per class
    """
    # Convert predictions and starts to numpy arrays
    preds = np.asarray(predictionProbs, dtype=float)
    starts = np.asarray(starts, dtype=int)
    
    m, numClasses = preds.shape
    
    L = int(trialLength) # length of the original trial
    acc = np.zeros((L, numClasses), dtype=float) # initialize accumulated probabilities
    cnt = np.zeros(L, dtype=float) # initialize counts of contributions per timestep
    
    # Assume per window per timestep predictions
    window_idx = (starts[:, None] + np.arange(Tx)[None, :]).ravel()  # length n_w * Tx
    values = np.repeat(preds, Tx, axis = 0) # preds.ravel()
    
    # Loop through classes and accumulate predicted probabilities at each sample index
    for c in range(numClasses):
        np.add.at(acc[:,c], window_idx, values[:, c])
  
    # And increment counts at each same index
    np.add.at(cnt, window_idx, 1)

    # Compute averaged probability per-sample, avoiding divide-by-zero:
    # 'Return a full array with the same shape and type as a given array.'
    avg = np.full_like(acc, np.nan, dtype=float)
    nonzero = cnt > 0
    avg[nonzero, :] = acc[nonzero, :] / cnt[nonzero, None]

    return avg, cnt

def transitionLabels(Y):
    '''Labels each timestep as 0,1,2,3 based on the classification.
        0: no intubation
        1: intubation ongoing
        2: start of intubation
        3: end of intubation
        Y: (m, Tx, 1) binary labels
        
        Returns: transitions: (m, Tx) integer labels per timestep
    '''
    m, Tx, _ = Y.shape
    Y_bin = Y.reshape(m, Tx)
    transitions = np.zeros((m, Tx))

    # Find changes in Y
    dY = np.diff(Y, axis=1)
    startIntubation = np.where(dY == 1)
    endIntubation = np.where(dY == -1) 
    
    # Label start and stop (transitions) of intubation
    transitions[startIntubation[0], startIntubation[1] + 1] = 2 
    transitions[endIntubation[0], endIntubation[1] + 1] = 3

    # Fill in 1s between the start and stop of intubation
    transitions[(transitions == 0) & (Y_bin == 1)] = 1
    
    return transitions 

def windowLabels(Y):
    '''Generates winow-level labels from original per-timestep labels.
    '''
    
    # Flatten Y yo shape (m, Tx)
    Y_flat = Y.reshape(Y.shape[0], Y.shape[1]) 
    m, Tx = Y_flat.shape
    
    # Find where the labels are all 0, all 1, or mixed for each example
    allZero = np.all(Y_flat == 0, axis=1)
    allOne = np.all(Y_flat == 1, axis=1)
    
    # Create a window label array that is m values long and contains all 2s
    yWindowLabels = np.full(m, -1, dtype=int)  # Default to mixed (2)
    # Replace with 0 where allZero is True
    yWindowLabels[allZero] = 0
    # Replace with 1 where allOne is True
    yWindowLabels[allOne] = 1
    
    boundaryIndices = np.where(~allZero & ~allOne)[0]

    for i in boundaryIndices:
        y_seq = Y_flat[i]

        # Look at first and last value in the sequence
        first = y_seq[0]
        last  = y_seq[-1]

        if first == 0 and last == 1:
            # likely a start window: 0 -> 1
            yWindowLabels[i] = 2
        elif first == 1 and last == 0:
            # likely a stop window: 1 -> 0
            yWindowLabels[i] = 3
    
    return yWindowLabels
    
# ---------------------------------------------------------------------------------------------

# Read in prediction information
predProbsTest = predictions["pred_probs"]
Y_windowLabels_test = predictions["Y_windowLabels_test"]

predClassesPerWindow = predictions['pred_classes']   
predConfPerWindow = predictions['pred_confidences']

os.makedirs('trialProbs', exist_ok=True)

# Loop through each trial 
for trial in np.unique(trialIDs_test):
    # Get length of trial
    key = trial if trial in trial_length_map else str(trial)
    L = int(trial_length_map[key])
    # Find indices of windows belonging to this trial
    idxs = np.where(trialIDs_test == trial)[0]
    if idxs.size == 0:
        continue

    # Use a local slice so we don't overwrite the master predictions array
    local_probs = predProbsTest[idxs, :]  # shape (n_windows, Tx)
    local_starts = starts_test[idxs]
    
    # Reassemble predicted probabilities
    avgProbs, counts = reassembleTrialsMultiClass(
        local_probs,
        local_starts,
        Tx=128,
        trialLength=L
    )
    
    # Label per timestep
    trueTransitionLabels = transitionLabels(Y_test)
    
    # Reassemble true labels (use averaged fraction per-sample)
    local_Y_windows = trueTransitionLabels[idxs, :].astype(float)
    avgTrue, countsTrue = reassembleTrials(
        local_Y_windows, 
        local_starts, 
        Tx=128, 
        trialLength=L
    )

    # Save the reconstructed arrays for later plotting/analysis
    out_data_path = os.path.join('trialProbs', f'trial_{trial}.npz')
    np.savez_compressed(out_data_path,
                        avgProbs=avgProbs,
                        counts=counts,
                        avgTrue=avgTrue,
                        countsTrue=countsTrue,
                        predClassesPerWindow=predClassesPerWindow,
                        predConfPerWindow=predConfPerWindow,
                        Y_windowLabels_test=Y_windowLabels_test,
                        length=L)
    


    



