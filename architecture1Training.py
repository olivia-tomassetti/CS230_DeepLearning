# Training script for intubation detection model
# This file loads the preprocessed training data and trains a deep learning model

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, TimeDistributed, Bidirectional
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
starts = np.load('starts_train.npy', allow_pickle=True)
starts_test = np.load('starts_test.npy', allow_pickle=True)
trialIDs = np.load('trial_ids_train.npy', allow_pickle=True)
trialIDs_test = np.load('trial_ids_test.npy', allow_pickle=True)

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
   
    Arguments:
        Tx -- length of the sequences in the corpus
        n_a -- the number of activations used in the model's first LSTM cells
    
    Returns:
        model -- a keras instance model
    """
    # Get the shape of input values
    n_values = 6
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values)) 
    
    # Bidirectional LSTM layer
    h = Bidirectional(LSTM(n_a, return_sequences=False))(X)
    
    # Output layer
    out = (Dense(4))(h)
    out = tf.keras.layers.Softmax(axis=-1)(out)
    
    model = Model(inputs=X, outputs=out)

    return model
   

def windowLabels(Y):
    
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
        y_seq = Y_flat[i]  # shape (Tx,)

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

    
# -----------------------------------------------------------------------------------

# Create model
model = intuModel(Tx=128, n_a=n_a)

# Set parameter optimization
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create window-level labels for training and test sets
Y_windowLabels = windowLabels(Y)
Y_windowLabels_test = windowLabels(Y_test)

print("X:", X.shape)             
print("y_window:", Y_windowLabels.shape)     
print("unique labels:", np.unique(Y_windowLabels))

print("X_test:", X_test.shape)       
print("y_window_test:", Y_windowLabels_test.shape) 

model.summary()

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

# Train the model (started with 50 epochs then increased and found 125 empirically)
print("\nStarting training...")
history = model.fit(
    X,
    Y_windowLabels, 
    validation_data=(X_test, Y_windowLabels_test),
    epochs=125, 
    batch_size=64, 
    verbose=0, 
    callbacks=callbacks,
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
print("Evaluating on test set...")
test_results = model.evaluate(X_test, Y_windowLabels_test, verbose=1)
print(f"Test loss: {test_results[0]:.4f}")
if len(test_results) > 1:
    print(f"Test accuracy: {test_results[1]:.4f}")
    
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

# Plot flattened elementwise accuracy recorded by Keras
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Train acc (flattened)')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Val acc (flattened)')
    
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Training history plot saved to 'training_history.png'")

# Save training history for later analysis (both npz and json)
try:
    np.savez_compressed('training_history.npz', **history.history)
    print("Saved training history to 'training_history.npz'")
except Exception as e:
    print(f"Warning: failed to save training history: {e}")

# Get predictions for test set 
# model.predict returns ndarray shape (m_test, 4) since it is per window and 4 classes
pred_probs = model.predict(X_test) 
pred_classes = np.argmax(pred_probs, axis=1)  # predicted class for each window 
pred_confidences = np.max(pred_probs, axis=1)  # confidence of predicted class

# Save both class probabilities and discrete labels for later analysis
try:
    np.savez_compressed('predictions.npz',
                        pred_probs=pred_probs,
                        pred_classes=pred_classes,
                        pred_confidences=pred_confidences,
                        Y_windowLabels_test=Y_windowLabels_test)
    print("Saved predictions to 'predictions.npz' (pred_probs, pred_classes)")
except Exception as e:
    print(f"Warning: failed to save predictions: {e}")

    
