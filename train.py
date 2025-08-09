# train.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- Paths & params ----------------
TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"
TEST_DIR = "dataset_split/test"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # increase for better results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Output files (Streamlit expects certain names)
TRAIN_ACC_PATH = os.path.join(MODELS_DIR, "training_accuracy.png")
TRAIN_LOSS_PATH = os.path.join(MODELS_DIR, "training_loss.png")
CM_EFF_PATH = os.path.join(MODELS_DIR, "confusion_matrix_efficientnet.png")
CM_RES_PATH = os.path.join(MODELS_DIR, "confusion_matrix_resnet.png")
SAMPLE_PRED_PATH = os.path.join(MODELS_DIR, "sample_predictions.png")

# ---------------- Data generators ----------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb', seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb', seed=SEED
)
# Important: shuffle=False so predictions align with filenames/classes
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb', shuffle=False
)

CLASS_LABELS = list(train_gen.class_indices.keys())
NUM_CLASSES = len(CLASS_LABELS)
print("Classes:", CLASS_LABELS)

# ---------------- Build models ----------------
def build_efficientnet(num_classes):
    base = EfficientNetB0(weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

def build_resnet(num_classes):
    base = ResNet50(weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

model_eff = build_efficientnet(NUM_CLASSES)
model_res = build_resnet(NUM_CLASSES)

# Freeze base layers initially (we used weights=None; still freeze for consistency)
for layer in model_eff.layers[:-4]:
    layer.trainable = False
for layer in model_res.layers[:-4]:
    layer.trainable = False

model_eff.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model_res.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------- Train ----------------
print("Training EfficientNetB0...")
history_eff = model_eff.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("Training ResNet50...")
history_res = model_res.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ---------------- Save models ----------------
eff_path = os.path.join(MODELS_DIR, "efficientnet_model.h5")
res_path = os.path.join(MODELS_DIR, "resnet_model.h5")
model_eff.save(eff_path)
model_res.save(res_path)
print(f"Saved models to:\n - {eff_path}\n - {res_path}")

# ---------------- Plots: accuracy & loss ----------------
plt.figure(figsize=(8,6))
plt.plot(history_eff.history.get('accuracy', []), label='EfficientNet_Train')
plt.plot(history_eff.history.get('val_accuracy', []), label='EfficientNet_Val')
plt.plot(history_res.history.get('accuracy', []), label='ResNet_Train')
plt.plot(history_res.history.get('val_accuracy', []), label='ResNet_Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(TRAIN_ACC_PATH)
plt.close()
print("Saved:", TRAIN_ACC_PATH)

plt.figure(figsize=(8,6))
plt.plot(history_eff.history.get('loss', []), label='EfficientNet_Train_Loss')
plt.plot(history_eff.history.get('val_loss', []), label='EfficientNet_Val_Loss')
plt.plot(history_res.history.get('loss', []), label='ResNet_Train_Loss')
plt.plot(history_res.history.get('val_loss', []), label='ResNet_Val_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(TRAIN_LOSS_PATH)
plt.close()
print("Saved:", TRAIN_LOSS_PATH)

# ---------------- Evaluate on test set ----------------
# Predict on full test generator (keeps order because shuffle=False)
print("Predicting on test set (EfficientNet)...")
y_pred_eff = model_eff.predict(test_gen, verbose=1)
y_pred_classes_eff = np.argmax(y_pred_eff, axis=1)

print("Predicting on test set (ResNet)...")
# reload generator (to avoid state issues) - recreate with shuffle=False
test_gen_for_res = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb', shuffle=False
)
y_pred_res = model_res.predict(test_gen_for_res, verbose=1)
y_pred_classes_res = np.argmax(y_pred_res, axis=1)

y_true = test_gen.classes  # from earlier generator (shuffle=False)
print("True labels count:", len(y_true))

# ---------------- Confusion Matrices & reports ----------------
cm_eff = confusion_matrix(y_true, y_pred_classes_eff)
plt.figure(figsize=(8,6))
sns.heatmap(cm_eff, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - EfficientNet")
plt.tight_layout()
plt.savefig(CM_EFF_PATH)
plt.close()
print("Saved:", CM_EFF_PATH)

cm_res = confusion_matrix(y_true, y_pred_classes_res)
plt.figure(figsize=(8,6))
sns.heatmap(cm_res, annot=True, fmt="d", cmap="Greens",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ResNet")
plt.tight_layout()
plt.savefig(CM_RES_PATH)
plt.close()
print("Saved:", CM_RES_PATH)

print("\nClassification Report (EfficientNet):\n")
print(classification_report(y_true, y_pred_classes_eff, target_names=CLASS_LABELS))

print("\nClassification Report (ResNet):\n")
print(classification_report(y_true, y_pred_classes_res, target_names=CLASS_LABELS))

# ---------------- Sample predictions visualization (ensemble) ----------------
# Create ensemble probabilities by averaging both models' predicted probabilities
y_pred_avg = (y_pred_eff + y_pred_res) / 2.0
y_pred_classes_avg = np.argmax(y_pred_avg, axis=1)

# pick up to 6 random samples from test set for display
n_show = 6
indices = random.sample(range(len(test_gen.filepaths)), min(n_show, len(test_gen.filepaths)))

plt.figure(figsize=(12, 8))
for i, idx in enumerate(indices):
    img_path = test_gen.filepaths[idx]
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0

    pred_idx = int(y_pred_classes_avg[idx])
    true_idx = int(y_true[idx])
    pred_label = CLASS_LABELS[pred_idx]
    true_label = CLASS_LABELS[true_idx]
    confidence = y_pred_avg[idx][pred_idx] * 100.0

    plt.subplot(2, 3, i+1)
    plt.imshow(arr)
    title = f"P: {pred_label} ({confidence:.1f}%)\nT: {true_label}"
    plt.title(title, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig(SAMPLE_PRED_PATH)
plt.close()
print("Saved:", SAMPLE_PRED_PATH)

print("\nAll evaluation images saved in the 'models/' folder. Done.")
