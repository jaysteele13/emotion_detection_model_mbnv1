
# imports

print('start imports')
from sklearn.model_selection import train_test_split
import h5py
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

import numpy as np
import cv2 as cv
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


# config

NUM_CLASSES = 7
IMG_SIZE = 224
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 7

TRAIN_EPOCH = 100
TRAIN_LR = 1e-4
TRAIN_ES_PATIENCE = 7
TRAIN_LR_PATIENCE = 3
TRAIN_MIN_LR = 1e-6
TRAIN_DROPOUT = 0.2

FT_EPOCH = 500
FT_LR = 1e-5
FT_LR_DECAY_STEP = 80.0
FT_LR_DECAY_RATE = 1
FT_ES_PATIENCE = 20
FT_DROPOUT = 0.3

ES_LR_MIN_DELTA = 0.001 # determines what acuracy jumps must happen before early stopping also


print("load data...")
version_model = 8
tl_file = f'../../../../../../mnt/scratch2/users/jsteele/facerecV2_models/face_rec_model_{version_model}_TL.keras'
ft_file = f'../../../../../../mnt/scratch2/users/jsteele/facerecV2_models/face_rec_model_{version_model}_FT.keras'

data_train = "../../../../../../../mnt/scratch2/users/jsteele/facerec-2_data/facerec_train.h5"
data_test_and_val = "../../../../../../../mnt/scratch2/users/jsteele/facerec-2_data/facerec_test_and_val.h5"

with h5py.File(data_train, 'r') as f:
    X_train = f['X_train'][:]
    y_train = f['y_train'][:]

with h5py.File(data_test_and_val, 'r') as f:
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]
    X_val = f['X_val'][:]
    y_val = f['y_val'][:]

# data already shuffled
def stratified_k_fold(X_train, y_train, n_splits=5):
    # Shuffle X_train and y_train together, maintaining their pairing
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=2)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits)

    # List to store splits
    splits = []

    # Iterate over the splits
    for train_idx, val_idx in skf.split(X_train_shuffled, y_train_shuffled):
        # Split the data based on indices
        X_train_fold, y_train_fold = X_train_shuffled[train_idx], y_train_shuffled[train_idx]

        # Append the split to the list
        splits.append((X_train_fold, y_train_fold))

    # Return shuffled training data and the list of cross-validation splits
    return X_train_shuffled, y_train_shuffled, splits


# shuffle once more
print("Shuffle!")
X_train, y_train, train_splits = stratified_k_fold(X_train, y_train)
X_val, y_val, train_splits = stratified_k_fold(X_val, y_val)


# Take third of val and add it to X train and y train

X_val, X_train_a, y_val, y_train_a = train_test_split(X_val, y_val, test_size=0.33)

X_train = np.concatenate((X_train, X_train_a))
y_train = np.concatenate((y_train, y_train_a))


print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_val.shape))
print("Shape of valid_label: {}".format(y_val.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))



# Model Building
input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')


sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(mode='horizontal'),
                                        tf.keras.layers.RandomContrast(factor=0.3)], name="augmentation")
preprocess_input = tf.keras.applications.mobilenet.preprocess_input

backbone = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
patch_extraction = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu')
], name='patch_extraction')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'),
                                          tf.keras.layers.BatchNormalization()], name='pre_classification')
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

inputs = input_layer
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(TRAIN_DROPOUT)(x)
x = pre_classification(x)

# fix attention
# å
x = tf.keras.layers.Reshape((1, 32))(x)  # Ensure 3D shape
x = self_attention([x, x])  # Attention output shape: (None, 1, 32)
x = tf.keras.layers.Flatten()(x)  # Convert to (None, 32)


outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs, name='train-head')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()

# Training Procedure
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) # changing lr_min_delta to be bigger
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
history = model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=1,
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback, tensorboard_callback])
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'§ Test Accuracy: {test_acc:.4f}')
model.save(tl_file)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)


print("\n🚀 Starting Dynamic Fine-tuning ...")

# Make base model trainable
base_model.trainable = True
total_layers = len(base_model.layers)

# Fine-tuning parameters
initial_unfreeze = 20
increment = 10
max_unfreeze = 60  # or use total_layers
epochs_per_stage = 8
total_epochs = 0
current_unfreeze = initial_unfreeze
val_accuracies = []

# Data augmentation for fine-tuning
FT_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.3)
])

# Freeze helper
def freeze_layers(model, unfreeze_count):
    fine_tune_from = total_layers - unfreeze_count
    for i, layer in enumerate(model.layers):
        if i < fine_tune_from:
            layer.trainable = False
        else:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

# Initial freeze before first compile
freeze_layers(base_model, current_unfreeze)

# Model building (only once)
inputs = input_layer
x = FT_data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
x = tf.keras.layers.SpatialDropout2D(FT_DROPOUT)(x)
x = global_average_layer(x)
x = pre_classification(x)
x = tf.keras.layers.Dropout(FT_DROPOUT)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs, name="finetune_model")

# Compile once initially
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FT_LR, global_clipnorm=3.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Static callbacks (used across all stages)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=ES_LR_MIN_DELTA,
    patience=20,
    restore_best_weights=True
)

callbacks_static = [early_stopping_callback, learning_rate_callback]

# Fine-tuning loop
while current_unfreeze <= max_unfreeze:
    print(f"\n🔓 Unfreezing last {current_unfreeze} layers ...")
    freeze_layers(base_model, current_unfreeze)
    
    # Recompile with new layer trainability
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FT_LR, global_clipnorm=3.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # TensorBoard log for current stage
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_unfreeze_{current_unfreeze}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Combine callbacks
    callbacks = callbacks_static + [tensorboard_callback]

    # Train
    history_finetune = model.fit(
        X_train, y_train,
        epochs=total_epochs + epochs_per_stage,
        initial_epoch=total_epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Update epoch counter
    total_epochs += epochs_per_stage

    # Track validation performance
    val_acc = history_finetune.history.get('val_accuracy', [None])[-1]
    val_accuracies.append(val_acc)
    print(f"✅ Finished stage unfreezing {current_unfreeze} layers - val_acc: {val_acc:.4f}")

    # Optional: Early break if performance stagnates across stages
    if len(val_accuracies) > 3 and val_accuracies[-1] < max(val_accuracies[-4:-1]):
        print("🛑 No improvement over 3 stages. Stopping early.")
        break

    # Next stage
    current_unfreeze += increment

# Final Evaluation
print("\n📊 Final Evaluation:")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'✅ Test Accuracy: {test_acc:.4f}')


# Save History to plot later
with h5py.File('tl_model_history.h5', 'w') as f:
    for key, value in history.history.items():
        f.create_dataset(key, data=value)

with h5py.File('ft_model_history.h5', 'w') as f:
    for key, value in history_finetune.history.items():
        f.create_dataset(key, data=value)

#--------------save model
model.save(ft_file)

# -------------------------------------------
