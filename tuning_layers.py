print("starting imports...")
import tensorflow as tf
import numpy as np
import h5py
import math
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import regularizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input, Resizing, Layer
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
print("finished laaaaaaaaaad...")
# -------- load data -----------------
print("load data...")
# train_path = 'drive/MyDrive/UNI/FYP/Demo/Initial_Model_With_Fer_affwild/Code/Affwild2/affwild_local_to_train.hdf5'
train_path = '../../../../../../mnt/scratch2/users/jsteele/affwild_preprocessed/cross_val_batch2_datasets2.hdf5'

with h5py.File(train_path, 'r') as f: 
    X = f['X_train'][:]
    y = f['y_train'][:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("data grabbed")
print(f"x_train shape {X_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"x_test shape {X_test.shape}")
print(f"y_test shape {y_test.shape}")

# config values
CONFIGURATION = {
    "BATCH_SIZE": 64,
    "IM_SIZE": 224,
    "LEARNING_RATE": 1e-3,
    "TL_N_EPOCHS": 150,
    "FT_N_EPOCHS": 75,
    "DROPOUT_RATE": 0.5,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 384,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 7,
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "AFF_NAMES": ["Neutral", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]   
}


# ---------- reset values START ---------
# Model architecture
# ---------- reset values ---------
base_model = None
model = None

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
num_classes = CONFIGURATION['NUM_CLASSES']
base_model.trainable = False
# ---------- reset values END ---------


class NormalizeLayer(Layer):
    def __init__(self, mean=None, scale=255.0, **kwargs):
        super(NormalizeLayer, self).__init__(**kwargs)
        self.mean = mean
        self.scale = scale

    def call(self, inputs):
        if self.mean is not None:
            inputs -= self.mean
        return inputs / self.scale



# Build Model

# Input & Resizing: Prepares and normalizes input images.
# Base Model: Leverages pre-trained feature extraction to reduce the need for training from scratch.
# Convolutional Blocks: Capture hierarchical features, with regularization and pooling to prevent overfitting and reduce dimensions.
# Global Max Pooling: Converts feature maps to a fixed-length vector.
# Fully Connected Layers: Extract high-level representations and enable classification while using dropout and regularization to prevent overfitting.

def build_jay_net_t1(out_dim: int, learning_rate: float) -> tf.keras.Model:

    DIMX = CONFIGURATION['IM_SIZE']
    DIMY = CONFIGURATION['IM_SIZE']
    L2_REG = 0.01  # Regularization strength

    # Input layer that accepts fixed image shape
    input_layer = Input(shape=(DIMX, DIMY, 3))  # Assuming DIMX, DIMY = 224

    # Resize layer
    x = Resizing(DIMX, DIMY)(input_layer)

    # Load the base model (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(DIMX, DIMY, 3))
    base_model.trainable = False  # Freeze the base model

    mean = K.constant([123.68, 116.779, 103.939])  # ImageNet mean values for normalization
    x = NormalizeLayer(mean=mean, scale=255.0)(x)

    x = Conv2D(96, (3, 3), activation="selu", padding="same", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # This will work when the spatial dimension is large enough
    x = Dropout(0.3)(x)

    x = Conv2D(196, (3, 3), activation="selu", padding="same", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Conv2D(196, (3, 3), activation="selu", padding="same", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    # If feature map size is very small, use GlobalMaxPooling2D to avoid errors
    x = GlobalMaxPooling2D()(x)  # This will handle small spatial dimensions like (1, 1)

    # Continue with the rest of the model
    x = Flatten()(x)
    x = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='selu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Final output layer
    predictions = Dense(out_dim, activation='softmax', kernel_regularizer=regularizers.l2(L2_REG), name="output")(x)

    # Model compilation
    model = Model(inputs=input_layer, outputs=predictions, name="jay_net_t1")
    model = compile_model(model, learning_rate)

    return model


def compile_model(model: tf.keras.Model, learning_rate: float,
    optimizer=tf.keras.optimizers.Adam) -> tf.keras.Model:
    # Create the ADAM optimizer with the new learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100000,
        decay_rate=0.93,
        staircase=True
    )
    model.compile(optimizer=optimizer(learning_rate=lr_schedule),
      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def unfreeze_base_layers(model: tf.keras.Model,
    layers: int, learning_rate: float) -> tf.keras.Model:

    # Access the base model
    base_model = model.layers[1]

    # Unfreeze the last `layers` number of layers
    for layer in base_model.layers[-layers:]:
        layer.trainable = True

    # Compile the model again to apply the changes
    model = compile_model(model, learning_rate)
    return model


# ----------- Transfer learning -> train top layers --------------

# build model
model = build_jay_net_t1(out_dim=7, learning_rate=0.0001)

# model.summary()

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., 'val_loss', 'val_accuracy')
    patience=3,  # Number of epochs to wait for improvement before stopping
    restore_best_weights=True,  # Restore the model weights from the best epoch
    verbose=1  # Print a message when stopping
)

# fit with history to report
fit_history_t = model.fit(X_train, y_train, epochs=CONFIGURATION['TL_N_EPOCHS'], validation_data=(X_test, y_test), callbacks=[early_stopping])

# get loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)
# ----------- END Transfer learning --------------
learning_rate = 0.000001
unfreeze=40

model = unfreeze_base_layers(model, layers=unfreeze,
        learning_rate=learning_rate)

fit_history_f = model.fit(X_train, y_train, epochs=CONFIGURATION['FT_N_EPOCHS'], validation_data=(X_test, y_test), callbacks=[early_stopping])

# --------------- UnFreese first 92 layers then train again ---------------

# get loss and accuracy
fine_tune_history = fit_history_f.history
transfer_learning_history = fit_history_t.history
loss, accuracy = model.evaluate(X_test, y_test)

# --------------- END Freese  40 layers  ---------------

# Save History to plot later
with h5py.File('tl_model_history.h5', 'w') as f:
    for key, value in transfer_learning_history.items():
        f.create_dataset(key, data=value)

with h5py.File('ft_model_history.h5', 'w') as f:
    for key, value in fine_tune_history.items():
        f.create_dataset(key, data=value)

#--------------save model
model.save('../../../../../../mnt/scratch2/users/jsteele/aff_models/aff_model_1.keras')
