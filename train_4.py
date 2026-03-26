import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 1. Basic Configurations
dataset_path = "Data Set"  # Folder with your dataset; should contain three subfolders:
                           # "Healthy", "Bacterial wilt disease", "Manganese Toxicity"
img_size = (224, 224)
batch_size = 32
initial_epochs = 5            # Frozen base epochs
fine_tune_epochs = 10         # Fine-tuning epochs
total_epochs = initial_epochs + fine_tune_epochs

# Define the classes explicitly, matching the folder names in your dataset.
classes_list = ["Healthy", "Bacterial wilt disease", "Manganese Toxicity"]

# 2. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,             # Increased rotation range for more diversity
    width_shift_range=0.3,         # Increased horizontal shift
    height_shift_range=0.3,        # Increased vertical shift
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
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
    classes=classes_list,  # Force the desired ordering of classes
    class_mode="categorical",
    subset="training"
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    classes=classes_list,
    class_mode="categorical",
    subset="validation"
)

# 3. Class Weights (If Imbalanced)
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# 4. Define a Smaller Pretrained Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)
base_model.trainable = False

# 5. Build the Classification Head
inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Increased dropout to reduce overfitting
# Add L2 regularization to the Dense layer
outputs = layers.Dense(train_generator.num_classes, activation='softmax',
                       kernel_regularizer=regularizers.l2(0.001))(x)
model = tf.keras.Model(inputs, outputs)

# 6. Compile (Phase 1: Frozen Base)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# 8. Train (Phase 1: Frozen Base)
history_frozen = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint, tensorboard_callback, lr_scheduler]
)

# 9. Fine-Tune: Unfreeze only the last 20 layers of the base model
fine_tune_start = len(base_model.layers) - 20
for layer in base_model.layers[fine_tune_start:]:
    layer.trainable = True

# Re-compile with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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

# 10. Save the Final Model
model.save("Plant_disease_model_4.keras")
print("✅ Model training completed! Saved as 'Plant_disease_model_4.keras'")
