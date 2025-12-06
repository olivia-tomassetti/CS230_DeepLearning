# Pre-processing intubation force/torque data.
# This file is used to identify intubation trials, label the data as either during or not during intubation, downsample the data,
# normalize the data, and create examples (windows of data) for the model to train on. 

# This file creates both training and development data sets. 

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split

# Helper functions
def isolateTrial(forceTorqueData, trialStart, trialStop):
    '''
    Isolate each trial with even amounts of data before and after the trial. This 
    should result in an even number of positive and negative samples.
    
    Arguments:
    forceTorqueData -- a pandas data frame with the force and torque data.
    trialStart -- the start time of the trial.
    trialStop -- the stop time of the trial.
    
    Returns:
    isolatedTrial -- a pandas data frame with the isolated trial data.
    '''
    trialLength = trialStop - trialStart
    lowerBound = trialStart - trialLength/2
    upperBound = trialStop + trialLength/2
    
    return forceTorqueData[(forceTorqueData["Sequence"] >= lowerBound) & (forceTorqueData["Sequence"] <= upperBound)]
    

def downsampleFTData(data, fs= 7000, fsDesired = 250, dataColumns = ["Fx","Fy","Fz","Tx","Ty","Tz"]):
    '''
    Downsample the force and torque data from 7000 Hz to 250 Hz.
    
    Arguments:
    data -- a pandas data frame with the force and torque data.
    fs -- the frequency the data was sampled at.
    fsDesired -- the frequency we want the data at.
    dataColumns -- the names of the columns with the data to downsample. 
    
    Returns:
    downsampledFT -- a pandas data frame with the downsampled data.
    '''
    up = 1
    down = int(round(fs/fsDesired))
    
    downsampled = {}
    for column in data:
        x = data[column].to_numpy()
        # Use signal.resample_poly(x, up, down) to downsample each column - has built in anti-aliasing 
        downsampled[column] = signal.resample_poly(x, up=up, down=down)
        
    downsampledFT = pd.DataFrame(downsampled)
    
    # Also downsample the Sequence number and y label columns 
    sequences = data['Sequence'].to_numpy()[::down]
    y = np.array([data['y'].iloc[i:i+down].max() for i in range(0, len(data['y']), down)])
    downsampledFT['Sequence'] = sequences
    downsampledFT['y'] = y
    
    return downsampledFT

def normalizeData(data, columns):
    '''
    Normalize data columns using z-score (mean=0, std=1) normalization.
    
    Arguments:
    data -- pandas DataFrame with the data to normalize (for entire task)
    columns -- list of column names to normalize ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    
    Returns:
    stats -- dictionary with mean/std or min/max for each column (for inverse transform if needed)
    '''
    stats = {}
    
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        stats[col] = {'mean': mean, 'std': std}
        
    return stats

def createExamples(data, columns, Tx=128, stride=16):
    '''
    Arguments:
    data -- pandas DataFrame
    columns -- list of columns that will be included in example ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    Tx -- window length for each example
    stride -- spacing between overlapping windows 
    
    Retruns: 
    X -- np.array with shape (nx, m, Tx) where nx is the length of columns, m is the number of examples, and Tx is the length of each example
    '''
    
    # Convert the dataframe to a numpy array and set equal to input X 
    X = data[columns].to_numpy()
    Y = data['y'].to_numpy()
    
    # N is the total length of the data and nx is number of elements in each data point 
    N, nx = X.shape 
    
    # Create an array of the window start indices (m, 1)
    startIdx = np.arange(0, N - Tx + 1, stride)[:, None] 
    
    starts = startIdx.flatten()
    
    # Create WindowIdx - add startIdx to each value of Tx array 
    windowIdx = startIdx + np.arange(Tx)[None, :]
    
    X = X[windowIdx,:]
    
    Y = Y[windowIdx]
     
    return X, Y, starts

def createMixedExamples(data, columns, Tx=128, stride=16, transitionStride=1):
    '''
    Arguments:
    data -- pandas DataFrame
    columns -- list of columns that will be included in example ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    Tx -- window length for each example
    stride -- spacing between overlapping windows 
    transitionStride -- stride to use for dense sampling around the transition points
    
    Retruns: 
    X -- np.array with shape (nx, m, Tx) where nx is the length of columns, m is the number of examples, and Tx is the length of each example
    '''
    # First create examples with the normal stride
    Xbase, Ybase, startsbase = createExamples(data, columns, Tx, stride)
    
    # Now look for the transition indices 
    yVector = data['y'].to_numpy().flatten()
    N = len(yVector)
    transitionIdices = np.where(yVector[:-1] != yVector[1:])[0]
    
    # Return the base examples if there are no transitions
    if transitionIdices.size == 0:
        return Xbase, Ybase, startsbase
    
    mixedStarts = []
    # Find all the start indices for sampling around the first transition
    for t in transitionIdices:
        startMin = max(0, (t + 1) - Tx - 1)
        startMax = min(len(yVector) - Tx, t + 1)
        
        
        if (startMin <= startMax):
            mixedStarts.extend(list(range(startMin, startMax + 1, transitionStride)))
            
    # Remove duplicates and sort
    mixedStarts = sorted(list(set(mixedStarts)))
    
    # Remove starts that exactly match base starts (avoid exact duplicates)
    exampleStarts = set(range(0, N - Tx + 1, stride))
    mixedStarts = [s for s in mixedStarts if s not in exampleStarts]
    
    
    X = data[columns].to_numpy()
    mixedWindowIdx = np.array(mixedStarts)[:, None] + np.arange(Tx)[None, :]
    Xmixed = X[mixedWindowIdx, :]
    Ymixed = yVector[mixedWindowIdx]
    
    # Combine
    X = np.concatenate([Xbase, Xmixed], axis=0)
    Y = np.concatenate([Ybase, Ymixed], axis=0)
    starts = np.concatenate([startsbase, np.array(mixedStarts)], axis=0)
    
    return X, Y, starts

#  ---------------------------------------------------------------------------------------------------------------
# All subjects with "good" data - determined by plotting and observing clear force/torque changes during intubation trials
subjects = ["13", "35",	"36",	"31",	"47",	"67",	"75",	"74",	"53",	
            "23",	"39",	"52",	"11",	"2",	"64",	"9",	"45",	"19",	
            "68",	"73",	"44",	"58",	"50",	"49",	"40",	"14",	"25"]

# Read start/stop frame file
startStopDP = pd.read_csv("StartStopSequenceNumbers.csv")

# List indices for each trial
trialIndices = {
    "trial1Pre":  (0, 1),
    "trial2Pre":  (2, 3),
    "trial3Pre":  (4, 5),
    "trial4Term": (6, 7),
    "trial5Term": (8, 9),
    "trial6Term": (10, 11),
}

# Columns in the force/torque data 
fColumns = ['Fx', 'Fy', 'Fz']
tColumns = ['Tx', 'Ty', 'Tz']
countsPerForceUnit = 224809
countsPerTorqueUnit = 8850746

# Dictionary to store all trials for all subjects
allSubjectTrials = {}

# Loop through each subject 
for subj_id in subjects:
    # Get the column for this subject
    pStartStopDP = startStopDP[subj_id]
    pStartStopDP = pd.to_numeric(pStartStopDP, errors="coerce")
    
    # Load this subject's force/torque data
    forceTorqueData = pd.read_csv(
        rf"OMSNI_ForceData\OMSNI {subj_id}",
        skiprows=12,
        sep=r'\s+',
    )
    
    # Convert counts to force/torque units
    for column in fColumns:
        forceTorqueData[column] = pd.to_numeric(forceTorqueData[column], errors="coerce")/countsPerForceUnit
    
    for column in tColumns:
        forceTorqueData[column] = pd.to_numeric(forceTorqueData[column], errors="coerce")/countsPerTorqueUnit

    # Add label column initialized to -1 (unknown)
    y = -1*np.ones((len(forceTorqueData), 1))
    forceTorqueData["y"] = y
    
    subjectTrials = {
        name: isolateTrial(
            forceTorqueData,
            pStartStopDP[i],
            pStartStopDP[j]
        ).copy()  # Make explicit copy to avoid SettingWithCopyWarning
        for name, (i, j) in trialIndices.items()
    }
    
    # Store this subject's trials in the main dictionary
    allSubjectTrials[subj_id] = subjectTrials
    
    
## LABELING
# Loop through each subject for labeling
for subj_id in subjects:
    # Get start/stop sequences for this subject
    pStartStopDP = startStopDP[subj_id]
    pStartStopDP = pd.to_numeric(pStartStopDP, errors="coerce")

    # Get this subject's trials
    subjectTrials = allSubjectTrials[subj_id]
    
    # Start by labeling all 0s
    for trialName, trialDf in subjectTrials.items():
        y = np.zeros((len(trialDf), 1))
        trialDf["y"] = y
        
        i, j = trialIndices[trialName]
        
        # Then determine when intubation is happening and label those as 1s
        location = trialDf.loc[
        trialDf["Sequence"].between(pStartStopDP[i], pStartStopDP[j], inclusive="both")
        ]
        trialDf.loc[location.index, "y"] = 1
    

# Plot some example trials to verify labeling - this was just for verification and debugging purposes
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
ax = axes.flat[0]
ax.plot(allSubjectTrials["75"]["trial1Pre"]['Sequence'], allSubjectTrials["75"]["trial1Pre"]['Fx'])
ax.plot(allSubjectTrials["75"]["trial1Pre"]['Sequence'], allSubjectTrials["75"]["trial1Pre"]['Fy'])
ax.plot(allSubjectTrials["75"]["trial1Pre"]['Sequence'], allSubjectTrials["75"]["trial1Pre"]['Fz'])
ax = axes.flat[1]
ax.plot(allSubjectTrials["75"]["trial1Pre"]['Sequence'] , allSubjectTrials["75"]["trial1Pre"]['y'])

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
ax = axes.flat[0]
ax.plot(allSubjectTrials["67"]["trial1Pre"]['Sequence'], allSubjectTrials["67"]["trial1Pre"]['Fx'])
ax.plot(allSubjectTrials["67"]["trial1Pre"]['Sequence'], allSubjectTrials["67"]["trial1Pre"]['Fy'])
ax.plot(allSubjectTrials["67"]["trial1Pre"]['Sequence'], allSubjectTrials["67"]["trial1Pre"]['Fz'])
ax = axes.flat[1]
ax.plot(allSubjectTrials["67"]["trial1Pre"]['Sequence'] , allSubjectTrials["67"]["trial1Pre"]['y'])

# Display all plots
# plt.show()

# These are the columns that contain data
columns = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# Define task groups: Pre (trials 1-3) and Term (trials 4-6)
taskGroups = {
    'Pre': ['trial1Pre', 'trial2Pre', 'trial3Pre'],
    'Term': ['trial4Term', 'trial5Term', 'trial6Term']
}

# Initialize X and Y as None to track first example
X = None
Y = None
starts = None
trialIDs = None

# Track which trial and task each example came from to ensure even distribution later
trial_labels = []  
trial_length_map = {}  

yArrays = []  
trial_ids_per_trial = [] 


for subj_id in subjects:
    # Get this subject's trials
    subjectTrials = allSubjectTrials[subj_id]
    
    # Normalize by task: for each task, compute stats from all trials, then apply to each
    for taskName, trialNames in taskGroups.items():
        # Collect all trials for this task
        taskTrials = [subjectTrials[trialName] for trialName in trialNames]
        
        # Concatenate all trials in this task to compute normalization statistics
        taskData = pd.concat(taskTrials, ignore_index=True)
        
        # Compute normalization statistics from the combined task data
        stats = normalizeData(taskData, columns)
        
        # Apply the same normalization to each individual trial in this task
        for trialName in trialNames:
            trialDf = subjectTrials[trialName].copy()
            for col in columns:
                if 'mean' in stats[col] and 'std' in stats[col]:
                    trialDf[col] = (trialDf[col] - stats[col]['mean']) / (stats[col]['std'] + 1e-8) # Add small epsilon to avoid divide by zero error
            subjectTrials[trialName] = trialDf
    
    # Now downsample and create examples for each trial
    for trialName, trialDf in subjectTrials.items():
        trialDf = downsampleFTData(trialDf)
        
        Xt, Yt, startst = createMixedExamples(trialDf, columns)
        
        # Build trial ID array
        trialIDstr = f"{subj_id}_{trialName}"
        trialIDt = np.array([trialIDstr] * len(startst), dtype=object)
        
        trial_length_map[trialIDstr] = len(trialDf)
        
        # Save per-trial y array and the trial id string
        yArrays.append(trialDf['y'].to_numpy().ravel())
        trial_ids_per_trial.append(trialIDstr)
        
        # Determine which task this trial belongs to
        task_name = 'Pre' if trialName in taskGroups['Pre'] else 'Term'
        
        # Track trial information for each example
        num_examples = Xt.shape[0]
        trial_labels.extend([(subj_id, trialName, task_name)] * num_examples)
        
        # If first trial, initialize X and Y
        if X is None:
            X = Xt
            Y = Yt
            starts = startst
            trialIDs = trialIDt
        # Else, concatenate to existing arrays
        else:
            X = np.concatenate((X, Xt))
            Y = np.concatenate((Y, Yt))
            starts = np.concatenate((starts, startst))
            trialIDs = np.concatenate((trialIDs, trialIDt))
            

# print(X.shape)
m = X.shape[0]
Y = Y.reshape(m, 128, 1)
# print("Y reshaped:", Y.shape)  # (62238, 128, 1)
# print(Y.shape)
# print(m)

# Categorize examples by averaging labels over timestep dimension
Y_mean = np.mean(Y, axis=1).flatten() 

# Categorize examples: all 0s (mean=0), all 1s (mean=1), transition (0 < mean < 1)
epsilon = 1e-6
all_zeros = Y_mean < epsilon
all_ones = Y_mean > (1 - epsilon)
mixed = ~(all_zeros | all_ones)

# Create label category: 0 = all zeros, 1 = all ones, 2 = transition
label_categories = np.zeros(m, dtype=int)
label_categories[all_ones] = 1
label_categories[mixed] = 2

# Extract task information for each example
task_labels = np.array([task_name for _, _, task_name in trial_labels])
# Create task category: 0 = Pre, 1 = Term
task_categories = np.zeros(m, dtype=int)
task_categories[task_labels == 'Term'] = 1

# Create combined stratification: combines label category and task type
# This ensures balanced distribution of both label types and task types
combinedStratify = label_categories * 10 + task_categories

# Print distribution
print("\nExample distribution by label type:")
print(f"All 0s (negative): {np.sum(all_zeros)} ({100*np.sum(all_zeros)/m:.1f}%)")
print(f"All 1s (positive): {np.sum(all_ones)} ({100*np.sum(all_ones)/m:.1f}%)")
print(f"Mixed (boundary): {np.sum(mixed)} ({100*np.sum(mixed)/m:.1f}%)")

print("\nExample distribution by task type:")
print(f"Pre task (trials 1-3): {np.sum(task_categories == 0)} ({100*np.sum(task_categories == 0)/m:.1f}%)")
print(f"Term task (trials 4-6): {np.sum(task_categories == 1)} ({100*np.sum(task_categories == 1)/m:.1f}%)")

print("\nExample distribution by combined category:")
for label_name, label_val in [("All 0s", 0), ("All 1s", 1), ("Mixed", 2)]:
    for task_name, task_val in [("Pre", 0), ("Term", 1)]:
        combined_val = label_val * 10 + task_val
        count = np.sum(combinedStratify == combined_val)
        print(f"  {label_name} + {task_name}: {count} ({100*count/m:.1f}%)")

# Shuffle and split into train/test sets with stratified sampling
test_size = 0.3
random_state = 42

# Get indices first to track which examples go to train/test
indices = np.arange(m)
train_indices, test_indices = train_test_split(
    indices,
    test_size=test_size,
    random_state=random_state,
    stratify=combinedStratify 
)

# Split X and Y using the same indices
X_train, X_test = X[train_indices], X[test_indices]
Y_train, Y_test = Y[train_indices], Y[test_indices]
starts_train, starts_test = starts[train_indices], starts[test_indices]
trialIDs_train, trialIDs_test = trialIDs[train_indices], trialIDs[test_indices]

# Verify the distribution in train and test sets
Y_train_mean = np.mean(Y_train, axis=1).flatten()
Y_test_mean = np.mean(Y_test, axis=1).flatten()

train_all_zeros = Y_train_mean < epsilon
train_all_ones = Y_train_mean > (1 - epsilon)
train_mixed = ~(train_all_zeros | train_all_ones)

test_all_zeros = Y_test_mean < epsilon
test_all_ones = Y_test_mean > (1 - epsilon)
test_mixed = ~(test_all_zeros | test_all_ones)

# Get task distribution for train and test
train_task_labels = task_labels[train_indices]
test_task_labels = task_labels[test_indices]

print(f"\n{'='*60}")
print(f"Train set: {len(X_train)} examples")
print(f"Label distribution:")
print(f"All 0s: {np.sum(train_all_zeros)} ({100*np.sum(train_all_zeros)/len(X_train):.1f}%)")
print(f"All 1s: {np.sum(train_all_ones)} ({100*np.sum(train_all_ones)/len(X_train):.1f}%)")
print(f"Mixed: {np.sum(train_mixed)} ({100*np.sum(train_mixed)/len(X_train):.1f}%)")
print(f"Task distribution:")
print(f"Pre: {np.sum(train_task_labels == 'Pre')} ({100*np.sum(train_task_labels == 'Pre')/len(X_train):.1f}%)")
print(f"Term: {np.sum(train_task_labels == 'Term')} ({100*np.sum(train_task_labels == 'Term')/len(X_train):.1f}%)")

print(f"\nTest set: {len(X_test)} examples")
print(f"Label distribution:")
print(f"All 0s: {np.sum(test_all_zeros)} ({100*np.sum(test_all_zeros)/len(X_test):.1f}%)")
print(f"All 1s: {np.sum(test_all_ones)} ({100*np.sum(test_all_ones)/len(X_test):.1f}%)")
print(f"Mixed: {np.sum(test_mixed)} ({100*np.sum(test_mixed)/len(X_test):.1f}%)")
print(f"Task distribution:")
print(f"Pre: {np.sum(test_task_labels == 'Pre')} ({100*np.sum(test_task_labels == 'Pre')/len(X_test):.1f}%)")
print(f"Term: {np.sum(test_task_labels == 'Term')} ({100*np.sum(test_task_labels == 'Term')/len(X_test):.1f}%)")
print(f"{'='*60}")

# Save train and test sets, starts, and trial IDs
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)
np.save('starts_train.npy', starts_train)
np.save('starts_test.npy',  starts_test)
np.save('trial_ids_train.npy', trialIDs_train)
np.save('trial_ids_test.npy',  trialIDs_test)

print(f"\nSaved train and test sets:")
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")


# Save trial length map to JSON for easy lookup
import json
with open('trial_length_map.json','w') as f:
    json.dump(trial_length_map, f, indent=2)

# Save per-trial arrays alongside their trial ids
np.savez_compressed('dataY_by_trial.npz',
                    trial_ids=np.array(trial_ids_per_trial, dtype=object),
                    y_arrays=np.array(yArrays, dtype=object))
