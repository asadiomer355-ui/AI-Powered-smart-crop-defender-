import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# 1. Basic Configurations
# ======================================================
dataset_path = "static"          # Path to your dataset folder
img_size = (224, 224)
batch_size = 32
initial_epochs = 10             # First phase (freeze base)
fine_tune_epochs = 20           # Second phase (unfreeze top layers)
total_epochs = initial_epochs + fine_tune_epochs

# ======================================================
# 2. Data Generators with Extended Augmentation
# ======================================================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,           # reduced from 45 to avoid extreme rotations
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# Compute class weights if dataset is imbalanced
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# ======================================================
# 3. Build a Transfer Learning Model (EfficientNetB0)
# ======================================================
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Freeze the entire base model initially
base_model.trainable = False

# Add classification head on top
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Dropout to reduce overfitting
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# ======================================================
# 4. Compile Model (Phase 1: Frozen base)
# ======================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ======================================================
# 5. Callbacks
# ======================================================
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor='val_loss',         # often more stable to track val_loss
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ======================================================
# 6. Train Phase 1: Frozen Base
# ======================================================
history_frozen = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, tensorboard_callback, lr_scheduler]
)

# ======================================================
# 7. Fine-Tune: Unfreeze top layers of the base model
# ======================================================
# Let's unfreeze the top block(s) of EfficientNetB0
# (Adjust how many layers to unfreeze based on your data size)
fine_tune_start = 100  # e.g., unfreeze from layer 100 onward
for layer in base_model.layers[fine_tune_start:]:
    layer.trainable = True

# Recompile with a potentially lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=total_epochs,  # total epochs = initial + fine_tune
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, tensorboard_callback, lr_scheduler],
    initial_epoch=history_frozen.epoch[-1]  # resume from previous training
)

# ======================================================
# 8. Save Final Model
# ======================================================
model.save("disease_model_final.keras")
print("✅ Model training completed! Saved as 'disease_model_final.keras'")
