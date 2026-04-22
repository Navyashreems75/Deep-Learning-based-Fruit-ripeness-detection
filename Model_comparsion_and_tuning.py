# ── STEP 1: Import Libraries ──────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
import cv2
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import shutil

print('✅ TensorFlow:', tf.__version__)
print('✅ GPU:', tf.config.list_physical_devices('GPU'))


# ── STEP 2: Find Correct Path Automatically ───────────────────
# This auto-detects the exact path so you don't have to guess
def find_dataset_path(base='/kaggle/input'):
    """Walk input folder and find where train/ and test/ live."""
    for root, dirs, files in os.walk(base):
        if 'train' in dirs and 'test' in dirs:
            print(f'✅ Found dataset at: {root}')
            return root
    raise FileNotFoundError('❌ Could not find train/ and test/ folders!')

BASE_DIR  = find_dataset_path()
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')
VAL_DIR   = '/kaggle/working/val'

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.0001
NUM_CLASSES = 9
SEED        = 42

# ── Verify all 9 classes ──────────────────────────────────────
print('\n📂 Train classes:')
for cls in sorted(os.listdir(TRAIN_DIR)):
    n = len([f for f in os.listdir(os.path.join(TRAIN_DIR, cls))
             if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f'  {cls:20s} → {n} images')

print('\n📂 Test classes:')
for cls in sorted(os.listdir(TEST_DIR)):
    n = len([f for f in os.listdir(os.path.join(TEST_DIR, cls))
             if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f'  {cls:20s} → {n} images')


# ── STEP 3: Carve 15% of Train → Validation ──────────────────
print('\n⏳ Creating validation set (15% of train)...')
os.makedirs(VAL_DIR, exist_ok=True)

for cls in sorted(os.listdir(TRAIN_DIR)):
    src_cls_dir = os.path.join(TRAIN_DIR, cls)
    val_cls_dir = os.path.join(VAL_DIR,   cls)
    os.makedirs(val_cls_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(src_cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]

    _, val_imgs = train_test_split(all_imgs, test_size=0.15, random_state=SEED)

    for img in val_imgs:
        src  = os.path.join(src_cls_dir, img)
        dest = os.path.join(val_cls_dir, img)
        if not os.path.exists(dest):
            shutil.copy2(src, dest)

    print(f'  {cls:20s} → {len(val_imgs)} val images')

print('✅ Validation set ready!')


# ── STEP 4: Data Augmentation & Generators ───────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=SEED
)
val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print(f'\n✅ Train : {train_gen.samples} images')
print(f'✅ Val   : {val_gen.samples} images')
print(f'✅ Test  : {test_gen.samples} images')
print(f'\n📋 Classes ({len(class_names)}): {class_names}')


# ── STEP 5: Visualize Sample Images ──────────────────────────
def plot_samples(generator, class_indices):
    idx_to_class = {v: k for k, v in class_indices.items()}
    images, labels = next(generator)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle('📸 Sample Images per Class', fontsize=15, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(idx_to_class[np.argmax(labels[i])], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('/kaggle/working/sample_images.png', dpi=150)
    plt.show()

plot_samples(train_gen, train_gen.class_indices)


# ── STEP 6: Build Transfer Learning Models ───────────────────
def build_model(base_fn, name, num_classes=9, lr=0.0001):
    base = base_fn(weights='imagenet', include_top=False,
                   input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(512, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(f'✅ {name} — params: {model.count_params():,}')
    return model

models_dict = {
    'DenseNet121' : build_model(DenseNet121,  'DenseNet121'),
    'VGG16'       : build_model(VGG16,        'VGG16'),
    'ResNet50'    : build_model(ResNet50,      'ResNet50'),
    'MobileNetV2' : build_model(MobileNetV2,  'MobileNetV2'),
}


# ── STEP 7: Callbacks ─────────────────────────────────────────
def get_callbacks(name):
    return [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'/kaggle/working/{name}_best.keras',
                        monitor='val_accuracy', save_best_only=True)
    ]


# ── STEP 8: Train All Models ──────────────────────────────────
histories = {}
for name, model in models_dict.items():
    print(f'\n{"="*55}')
    print(f'🏋️  Training {name}...')
    print(f'{"="*55}')
    history = model.fit(
        train_gen, epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(name),
        verbose=1
    )
    histories[name] = history
    print(f'✅ {name} done!')


# ── STEP 9: Fine-Tune DenseNet121 ────────────────────────────
print('\n🔓 Fine-tuning DenseNet121 (unfreeze last 50 layers)...')
densenet   = models_dict['DenseNet121']
base_layer = densenet.layers[1]
for layer in base_layer.layers[-50:]:
    layer.trainable = True

densenet.compile(optimizer=Adam(1e-5),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

ft_history = densenet.fit(
    train_gen, epochs=15,
    validation_data=val_gen,
    callbacks=get_callbacks('DenseNet121_ft'),
    verbose=1
)
histories['DenseNet121_FT'] = ft_history
print('✅ Fine-tuning complete!')


# ── STEP 10: Plot Training Curves ────────────────────────────
def plot_history(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'📈 {name} Training History', fontsize=13, fontweight='bold')
    ax[0].plot(history.history['accuracy'],     label='Train', color='#2D6A4F', lw=2)
    ax[0].plot(history.history['val_accuracy'], label='Val',   color='#F4A261', lw=2)
    ax[0].set_title('Accuracy'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(history.history['loss'],         label='Train', color='#2D6A4F', lw=2)
    ax[1].plot(history.history['val_loss'],     label='Val',   color='#E76F51', lw=2)
    ax[1].set_title('Loss'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/{name}_history.png', dpi=150)
    plt.show()

for name, hist in histories.items():
    plot_history(hist, name)


# ── STEP 11: Evaluate All Models ─────────────────────────────
def evaluate_model(model, generator, name):
    generator.reset()
    y_pred    = np.argmax(model.predict(generator, verbose=0), axis=1)
    y_true    = generator.classes
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
    recalls   = recall_score(y_true, y_pred, average=None)
    gmean     = np.prod(recalls) ** (1.0 / len(recalls))

    print(f'\n📊 {name}:')
    print(f'  Accuracy   : {acc*100:.2f}%')
    print(f'  Precision  : {precision*100:.2f}%')
    print(f'  Recall     : {recall*100:.2f}%')
    print(f'  F1-Score   : {f1*100:.2f}%')
    print(f'  G-Mean     : {gmean*100:.2f}%')
    print(f'  Error Rate : {(1-acc)*100:.2f}%')

    return {
        'Model'      : name,
        'Accuracy'   : round(acc*100, 2),
        'Precision'  : round(precision*100, 2),
        'Recall'     : round(recall*100, 2),
        'F1-Score'   : round(f1*100, 2),
        'G-Mean'     : round(gmean*100, 2),
        'Error Rate' : round((1-acc)*100, 2),
        'y_true'     : y_true,
        'y_pred'     : y_pred
    }

results = {name: evaluate_model(m, test_gen, name)
           for name, m in models_dict.items()}


# ── STEP 12: Comparison Table & Chart ────────────────────────
cols = ['Model','Accuracy','Precision','Recall','F1-Score','G-Mean','Error Rate']
df   = pd.DataFrame([{k: v for k,v in r.items() if k in cols}
                     for r in results.values()])
df   = df.sort_values('Accuracy', ascending=False)

print('\n🏆 MODEL COMPARISON:')
print(df.to_string(index=False))
df.to_csv('/kaggle/working/model_comparison.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df)); w = 0.2
colors = ['#2D6A4F','#52B788','#F4A261','#E76F51']
for i, (m, c) in enumerate(zip(['Accuracy','Precision','Recall','F1-Score'], colors)):
    ax.bar(x + i*w, df[m], w, label=m, color=c)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(df['Model'], fontsize=11)
ax.set_ylabel('Score (%)'); ax.legend()
ax.set_title('📊 Model Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(85, 101); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/model_comparison.png', dpi=150)
plt.show()


# ── STEP 13: Confusion Matrix ─────────────────────────────────
def plot_cm(y_true, y_pred, labels, name):
    cm_val = confusion_matrix(y_true, y_pred)
    cm_pct = cm_val.astype('float') / cm_val.sum(axis=1)[:, None] * 100
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='YlGn',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'🔥 Confusion Matrix — {name} (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/{name}_cm.png', dpi=150)
    plt.show()

plot_cm(results['DenseNet121']['y_true'],
        results['DenseNet121']['y_pred'],
        class_names, 'DenseNet121')

print('\n📋 Classification Report — DenseNet121:')
print(classification_report(
    results['DenseNet121']['y_true'],
    results['DenseNet121']['y_pred'],
    target_names=class_names))
