# Training script for intubation detection model
# This file loads the preprocessed training data and trains a deep learning model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# Add callbacks for AWS training
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
'''Mixed precision training is the use of lower-precision operations 
(float16 and bfloat16) in a model during training to make it run faster 
and use less memory. Using mixed precision can improve performance by 
more than 3 times on modern GPUs and 60% on TPUs.'''

# Load training and test data
X = np.load('X_train.npy')
Y = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
starts = np.load('starts_train.npy')
starts_test = np.load('starts_test.npy')
trialIDs = np.load('trial_ids_train.npy')
trialIDs_test = np.load('trial_ids_test.npy')

print(f"  X (train) shape: {X.shape}")
print(f"  Y (train) shape: {Y.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  Y_test shape: {Y_test.shape}")

# Build training model
n_a = 64 # chosen from Improvise Jazz Solo
m, Tx, n_values = X.shape # forces and torques

reshaper = Reshape((1, n_values))      
LSTM_cell = LSTM(n_a, return_state=True) # Used in Step 2.B of djmodel(), below
densor = Dense(1, activation='sigmoid')     # Used in Step 2.D


# Modified from CS230 Improvise Jazz Solo Model

def intuModel(Tx, n_a):
    """
    Implement the model composed of Tx LSTM cells where each cell is responsible
    for learning whether the force/torque data is occuring during an intubation 
    based on the previous force/torque reading in the sequence.
    Each cell has the following schema: 
            [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
    Arguments:
        Tx -- length of the sequences in the corpus
        LSTM_cell -- LSTM layer instance
        densor -- Dense layer instance
        reshaper -- Reshape layer instance
    
    Returns:
        model -- a keras instance model with inputs [X, a0, c0]
    """
    # Get the shape of input values
    n_values = 6
    
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values)) 
    
    h = LSTM(n_a, return_sequences=True)(X)
    
    # TimeDistributed applies "a layer to every temporal slice of an input"
    out = TimeDistributed(Dense(1, activation='sigmoid'))(h)
    
    model = Model(inputs=X, outputs=out)
    
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    # a0 = Input(shape=(n_a,), name='a0')
    # c0 = Input(shape=(n_a,), name='c0')
    # a = a0
    # c = c0

    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    # outputs = []
    
    # Step 2: Loop over tx
    # for t in range(Tx):
        
    #     # Step 2.A: select the "t"th time step vector from X. 
    #     # x = Lambda(lambda z: z[:, t, :])(X)
    #     x = X[:,t,:]
    #     # Step 2.B: Use reshaper to reshape x to be (1, n_values) (≈1 line)
    #     x = reshaper(x)
    #     # Step 2.C: Perform one step of the LSTM_cell
    #     _, a, c = LSTM_cell(x, initial_state=[a,c])
    #     # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
    #     out = densor(a)
    #     # Step 2.E: append the output to "outputs"
    #     outputs.append(out)
        
    # # Step 3: Create model instance
    
    # # stacked = tf.stack(outputs, axis=1)         # shape (batch, Tx, 1)
    # model = Model(inputs=[X, a0, c0], outputs=outputs)

    
    return model

import numpy as np

def reassembleTrials(predictionProbs, starts, Tx, trialLength):
    """
    pred_probs: (n_windows,) or (n_windows, ) floats in [0,1] - predicted probability for each window element or
                if pred per-timestep: (n_windows, Tx) probabilities
    window_starts: (n_windows,) start indices (int) relative to trial (0-based)
    Tx: window length
    trial_length: total length of trial in samples (optional - used to size output)
    Returns: avg_prob (length = trial_length or max covered index+1), counts (same length)
    """
    # Support either per-window scalar prob or per-window per-timestep probs
    preds = np.asarray(predictionProbs)
    starts = np.asarray(starts, dtype=int)
    
    L = int(trialLength) # length of the original trial
    acc = np.zeros(L, dtype=float) # accumulated probabilities
    cnt = np.zeros(L, dtype=float) # counts of contributions per timestep
    
    # Assume per window per timestep predictions
    
    # Two cases: preds are per-window per-timestep (n_w, Tx), or scalar per-window (n_w,)
    # if preds.ndim == 2 and preds.shape[1] == Tx:
        # Vectorized accumulation:
        # Compute flattened sample indices for every window timestep:
        #   window_idx = [s0+0, s0+1, ..., s0+Tx-1, s1+0, s1+1, ...]
    window_idx = (starts[:, None] + np.arange(Tx)[None, :]).ravel()  # length n_w * Tx
    values = preds.ravel()                                             # matching length

        # Mask to ensure indices don't exceed L (useful if trial_length < max_required)
        # mask_valid = window_idx < L
        # if not mask_valid.all():
        # window_idx = window_idx[mask_valid]
        # values = values[mask_valid]

        # np.add.at performs in-place scattered addition:
    #   acc[window_idx[i]] += values[i] for every i
    np.add.at(acc, window_idx, values)
    # and increment counts at each same index
    np.add.at(cnt, window_idx, 1)

    # else:
    #     # Scalar-per-window case: project the scalar value across the Tx timesteps
    #     for i, s in enumerate(starts):
    #         v = float(preds[i])
    #         end = min(L, s + Tx)
    #         acc[s:end] += v
    #         cnt[s:end] += 1

    # Compute averaged probability per-sample, avoiding divide-by-zero:
    # 'Return a full array with the same shape and type as a given array.'
    avg = np.full_like(acc, np.nan, dtype=float)
    nonzero = cnt > 0
    avg[nonzero] = acc[nonzero] / cnt[nonzero]

    return avg, cnt

# -----------------------------------------------------------------------------------

# Create model
# model = intuModel(Tx=128, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)

model = intuModel(Tx=128, n_a=n_a)

# Set parameter optimization
# opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999) # No decay for now 
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999) # No decay for now 
# model.compile(optimizer=opt, loss=['binary_crossentropy']*Tx, metrics=['accuracy']*Tx)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

# Model outputs a list of length Tx, so targets must be a list of length Tx
# Y_list = [Y[:, t, :] for t in range(Tx)]
# Y_test_list = [Y_test[:, t, :] for t in range(Tx)]


# Model outputs a single stacked tensor of shape (batch, Tx, 1), so use
# the stacked target arrays directly.
Y_stack = Y  # shape (m, Tx, 1)
Y_test_stack = Y_test


# Prepare initial states for test set
m_test = X_test.shape[0]
a0_test = np.zeros((m_test, n_a))
c0_test = np.zeros((m_test, n_a))


# Save best model during training
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate if loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Train the model - start with 50 epochs
print("\nStarting training...")
history = model.fit(
    X,
    # [X, a0, c0], 
    Y_stack, # Y_list, #
    validation_data=(X_test, Y_test_stack),
    # validation_data=([X_test, a0_test, c0_test], Y_test_list), #Y_test_stack),
    epochs=50,  # Increased epochs since we have early stopping
    batch_size=64, #32,  # Add batch size for better memory management
    verbose=0  # SSet to 1 to see progress on AWS
    #callbacks=callbacks
)

# Print training summary
print(f"\nTraining completed!")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")

# Save final model
model.save('final_model.h5')
print("Model saved to 'final_model.h5'")

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = model.evaluate(X_test, Y_test_stack, verbose=1)
# test_results = model.evaluate([X_test, a0_test, c0_test], Y_test_stack, verbose=1)
print(f"Test loss: {test_results[0]:.4f}")

# Plot training history (save instead of show for AWS)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
# Plot average accuracy across all timesteps if available

# Plot flattened elementwise accuracy recorded by Keras
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Train acc (flattened)')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Val acc (flattened)')
    
# if 'accuracy' in history.history:
#     # Average accuracy across all timesteps
#     train_acc = np.mean([history.history[f'output_{i}_accuracy'] for i in range(Tx)], axis=0) if any(f'output_{i}_accuracy' in history.history for i in range(Tx)) else None
#     val_acc = np.mean([history.history[f'val_output_{i}_accuracy'] for i in range(Tx)], axis=0) if any(f'val_output_{i}_accuracy' in history.history for i in range(Tx)) else None
#     if train_acc is not None:
#         plt.plot(train_acc, label='Training Accuracy')
#         plt.plot(val_acc, label='Validation Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.title('Model Accuracy')
#         plt.legend()
#         plt.grid(True)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Training history plot saved to 'training_history.png'")

# Don't use plt.show() on AWS - it requires a display
# plt.show()  # Uncomment if running locally with display

# Get predictions for test set (or train set)
# model.predict returns ndarray shape (m_test, Tx, 1)
preds = model.predict(X_test) 
# preds = model.predict([X_test, a0_test, c0_test])  # shape (m_test, Tx, 1)
preds = preds.reshape(preds.shape[0], preds.shape[1])  # (m_test, Tx)
pred_bin = (preds >= 0.5).astype(int)

# True labels shape -> (m_test, Tx)
Y_test_mat = Y_test.reshape(Y_test.shape[0], Y_test.shape[1]) if Y_test.ndim == 3 else Y_test

# Plot some sample predictions vs true labels
sample_indices = [10, 100, 1000]  # change or random sample
for idx in sample_indices:
    plt.figure(figsize=(8,2))
    plt.plot(Y_test_mat[idx,:], label='true', marker='o')
    plt.plot(pred_bin[idx,:], label='pred', marker='x', alpha=0.8)
    plt.ylim(-0.1,1.1)
    plt.xlabel('Timestep'); plt.title(f'Sample {idx} true vs pred')
    plt.legend(); plt.grid(True)
    plt.savefig(f'sample_{idx}_true_vs_pred.png', dpi=150)

# Compute and save per-timestep accuracy (test set)
per_timestep_acc = np.mean(pred_bin == Y_test_mat, axis=0)  # shape (Tx,)
plt.figure(figsize=(12,4))
plt.plot(per_timestep_acc, marker='o')
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.title('Per-timestep Accuracy (Test set)')
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('per_timestep_accuracy.png', dpi=150, bbox_inches='tight')
print("Saved per-timestep accuracy to 'per_timestep_accuracy.png'")

# Compute and save exact-match (window-level) accuracy
exact_match_acc = np.mean(np.all(pred_bin == Y_test_mat, axis=1))
plt.figure(figsize=(4,4))
plt.bar(['Exact match'], [exact_match_acc], color='tab:blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title(f'Exact-match accuracy: {exact_match_acc:.3f}')
plt.savefig('exact_match_accuracy.png', dpi=150, bbox_inches='tight')
print(f"Exact-match accuracy: {exact_match_acc:.4f} (saved to 'exact_match_accuracy.png')")


eps = 1e-6
# 1) window-level true label type: 0=all0, 1=all1, 2=mixed
Y_mean = Y_test_mat.mean(axis=1)          # shape (m_test,)
is_all0 = Y_mean < eps
is_all1 = Y_mean > (1 - eps)
is_mixed = ~(is_all0 | is_all1)
window_type = np.zeros(len(Y_mean), dtype=int)
window_type[is_all1] = 1
window_type[is_mixed] = 2

# 2) correctness measures
exact_match = np.all(pred_bin == Y_test_mat, axis=1)         # True if every timestep matches
majority_true = (Y_test_mat.mean(axis=1) >= 0.5).astype(int)
majority_pred = (pred_bin.mean(axis=1) >= 0.5).astype(int)
majority_correct = (majority_pred == majority_true)

# 3) summary counts & accuracies per type
type_names = {0: 'All-0', 1: 'All-1', 2: 'Mixed'}
summary = {}
for t in (0,1,2):
    idx = np.where(window_type == t)[0]
    n = len(idx)
    if n == 0:
        summary[t] = {'count':0, 'exact_match_accuracy':None, 'majority_accuracy':None}
        continue
    exact_acc = exact_match[idx].mean()
    maj_acc = majority_correct[idx].mean()
    summary[t] = {'count': n, 'exact_match_accuracy': float(exact_acc), 'majority_accuracy': float(maj_acc)}

# Print readable table
print("Window-type breakdown (count, exact-match acc, majority-vote acc):")
for t in (0,1,2):
    s = summary[t]
    print(f"  {type_names[t]:6s}: {s['count']:5d}  | exact-match: {s['exact_match_accuracy']}  | majority: {s['majority_accuracy']}")