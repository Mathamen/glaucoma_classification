import tensorflow as tf
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

DATA_DIR = "data/training_set"
TEST_DIR = "data/test_set"
IMG_SIZE = (300, 300)
BATCH_SIZE = 12
EPOCHS = 25



train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

print("Detected classes:", train_ds.class_names)



data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])



def mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    weight = tf.random.uniform([], 0, alpha, dtype=images.dtype)
    
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = weight * images + (1 - weight) * tf.gather(images, indices)
    
    labels_float = tf.cast(labels, images.dtype)
    mixed_labels = weight * labels_float + (1 - weight) * tf.gather(labels_float, indices)
    
    return mixed_images, mixed_labels

def apply_preprocessing(x, y, augment=False):
    x = preprocess_input(x)
    if augment:
        x = data_augmentation(x, training=True)
    return x, y

train_ds = train_ds.map(lambda x, y: apply_preprocessing(x, y, augment=True))
train_ds = train_ds.map(mixup)  # Apply mixup
val_ds = val_ds.map(lambda x, y: apply_preprocessing(x, y, augment=False))
test_ds = test_ds.map(lambda x, y: apply_preprocessing(x, y, augment=False))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)



all_labels = []
for _, y in tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
).unbatch():
    all_labels.append(int(y.numpy()))

all_labels = np.array(all_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(all_labels),
    y=all_labels
)

class_weight = {i: weights[i] for i in range(len(weights))}




base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)

base.trainable = False

x = base.output

gap = tf.keras.layers.GlobalAveragePooling2D()(x)
gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
concat = tf.keras.layers.Concatenate()([gap, gmp])

x = tf.keras.layers.Dense(256, activation="relu")(concat)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid", dtype='float32')(x)

model = tf.keras.Model(base.input, output)
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        "accuracy", 
        tf.keras.metrics.AUC(name="AUC"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)



callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_AUC", 
        patience=8, 
        mode="max", 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", 
        monitor="val_AUC", 
        mode="max", 
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_AUC",
        factor=0.5,
        patience=4,
        mode="max",
        min_lr=1e-7,
        verbose=1
    )
]




history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)




base.trainable = True

for layer in base.layers[:-100]:
    layer.trainable = False

optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        "accuracy", 
        tf.keras.metrics.AUC(name="AUC"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks
)




for layer in base.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-6, weight_decay=1e-5)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        "accuracy", 
        tf.keras.metrics.AUC(name="AUC"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

history3 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks
)



def tta_predict(model, dataset, num_augmentations=5):
    predictions = []
    
    for x, y in dataset:
        batch_preds = []
        # Original prediction
        batch_preds.append(model.predict(x, verbose=0))
        
        # Augmented predictions
        for _ in range(num_augmentations - 1):
            augmented = data_augmentation(x, training=True)
            batch_preds.append(model.predict(augmented, verbose=0))
        
        # Average predictions
        avg_pred = np.mean(batch_preds, axis=0)
        predictions.append(avg_pred)
    
    return np.concatenate(predictions)




loss, acc, auc, precision, recall = model.evaluate(test_ds)

model.save("glaucoma_model_final.h5")