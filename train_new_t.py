import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# 1. Basic Configurations
# ===============================
dataset_path = "static"   # Path to your dataset folder
img_size = (224, 224)         
batch_size = 32
initial_epochs = 5              # Fewer epochs for frozen base
fine_tune_epochs = 10           # Fine-tune phase
total_epochs = initial_epochs + fine_tune_epochs

# ===============================
# 2. Data Augmentation
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=15,       # Moderate rotation
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
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
    class_mode="categorical",  # Multi-class classification
    subset="training"
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# 3. Class Weights (If Imbalanced)
# ===============================
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# ===============================
# 4. Define a Smaller Pretrained Model (MobileNetV2)
# ===============================
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)

# Freeze the base model initially
base_model.trainable = False

# ===============================
# 5. Build the Classification Head
# ===============================
inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)  # Dropout to reduce overfitting
# For multi-class classification, final Dense = number of classes with softmax
outputs = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# ===============================
# 6. Compile (Phase 1: Frozen Base)
# ===============================
# If mixed precision is enabled, the default float policy might be float16
# so you might need to adjust the final output layer's dtype or add a 
# tf.keras.layers.Activation('softmax', dtype='float32') if required.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 7. Callbacks
# ===============================
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop earlier if no improvement
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# ===============================
# 8. Train (Phase 1: Frozen Base)
# ===============================
history_frozen = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, tensorboard_callback, lr_scheduler]
)

# ===============================
# 9. Fine-Tune: Unfreeze Top Layers
# ===============================
# Decide how many layers to unfreeze. MobileNetV2 has ~154 layers.
# We'll unfreeze from somewhere near the top, e.g. last ~40 layers:
fine_tune_start = 100  # Adjust if you want to unfreeze more or fewer layers
for layer in base_model.layers[fine_tune_start:]:
    layer.trainable = True

# Re-compile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=total_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, tensorboard_callback, lr_scheduler],
    initial_epoch=history_frozen.epoch[-1]
)

# ===============================
# 10. Save the Final Model
# ===============================
model.save("Plant_disease_model.keras")
print("✅ Model training completed! Saved as 'Plant_disease_model.keras'")
