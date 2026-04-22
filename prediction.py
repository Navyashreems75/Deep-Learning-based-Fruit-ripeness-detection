import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm

from tensorflow import keras

# Load BEST model (DenseNet121 - 99% accuracy 💯)
model = keras.models.load_model(
    '/kaggle/input/models/navyashrems/densenetb/keras/default/1/DenseNet121_ft_best.keras')

# Create dictionary (needed for your code)
models_dict = {'DenseNet121': model}

print("✅ Model loaded successfully!")

# test directory path
TEST_DIR = '/kaggle/input/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten/fruit_ripeness_dataset/archive (1)/dataset/test'

class_names = sorted(os.listdir(TEST_DIR))
print("Classes:", class_names)

sample_cls = class_names[2]

sample_img = os.path.join(
    TEST_DIR,
    sample_cls,
    os.listdir(os.path.join(TEST_DIR, sample_cls))[0]
)

print("Sample image:", sample_img)

# predict the picture of the fruit
def predict_image(img_path, model, class_names):
    IMG_SIZE = (224, 224)

    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img) / 255.0

    preds = model.predict(np.expand_dims(arr, 0))[0]

    top = np.argmax(preds)

    print(f"✅ Prediction: {class_names[top]} ({preds[top]*100:.2f}%)")

    plt.imshow(img)
    plt.title(class_names[top])
    plt.axis('off')
    plt.show()

# predicts each picture of all classes
for cls in class_names:
    img_path = os.path.join(
        TEST_DIR,
        cls,
        os.listdir(os.path.join(TEST_DIR, cls))[6]
    )
    
    print(f"\n🔹 Class: {cls}")
    predict_image(img_path, model, class_names)
