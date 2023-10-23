import os
import cv2
import selective_search
import numpy as np
import tensorflow as tf
import pywt
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, Iterator
from xml.etree import ElementTree


# Implementing the selective search:
def generate_roi(image_path, xml_path):
    image = cv2.imread(image_path)
    _, regions = selective_search.selective_search(image)


    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    bboxes = []

    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))

    positive_regions = []
    negative_regions = []

    for region in regions:
        x, y, w, h = region['rect']

        is_positive = False
        for bbox in bboxes:
            if (x < bbox[2] and x + w > bbox[0] and y < bbox[3] and y + h > bbox[1]):
                is_positive = True
                break

        if is_positive:
            positive_regions.append(image[y:y+h, x:x+w])
        else:
            negative_regions.append(image[y:y+h, x:x+w])

    return positive_regions, negative_regions

# Path to save positive and negative regions:
train_positive_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/train/weap'
train_negative_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/train/norm'
train_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/train/weap'
test_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/test'
# Generate ROIs for training images:
for image_name in os.listdir(train_dir):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        base_name = os.path.splitext(image_name)[0]
        xml_name = base_name + '.xml'

        pos_regions, neg_regions = generate_roi(os.path.join(train_dir, image_name), os.path.join(train_dir, xml_name))
        print(1)
        # for index, region in enumerate(pos_regions):
        #     cv2.imwrite(os.path.join(train_positive_dir, base_name + f"_pos_{index}.jpg"), region)
        #
        # for index, region in enumerate(neg_regions):
        #     cv2.imwrite(os.path.join(train_negative_dir, base_name + f"_neg_{index}.jpg"), region)

def custom_preprocessing(img):
    img = (img * 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    coeffs = pywt.dwt2(gray, 'db1')  # Using Daubechies wavelet
    cA, (cH, cV, cD) = coeffs

    combined_edges = np.sqrt(cH**2 + cV**2 + cD**2)

    if combined_edges.min() != combined_edges.max():
        combined_edges_rescaled = ((combined_edges - combined_edges.min()) * (255 / (combined_edges.max() - combined_edges.min()))).astype(np.uint8)
    else:
        combined_edges_rescaled = np.zeros_like(combined_edges, dtype=np.uint8)

    wavelet_edges = cv2.resize(combined_edges_rescaled, (img.shape[1], img.shape[0]))

    stacked_img = np.stack([wavelet_edges] * 3, axis=-1)

    final_img = stacked_img.astype(np.float32) / 255.0

    final_img = np.clip(final_img, 0, 1)

    return final_img

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)
# Combine the two datasets:
from keras.preprocessing.image import DirectoryIterator

class CombinedDirectoryIterator(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pos_gen = DirectoryIterator(directory=train_positive_dir, image_data_generator=train_datagen, target_size=(128, 128), class_mode='binary', shuffle=True)
        neg_gen = DirectoryIterator(directory=train_negative_dir, image_data_generator=train_datagen, target_size=(128, 128), class_mode='binary', shuffle=True)
        self.generators = [pos_gen, neg_gen]

    def __next__(self):
        next_X = []
        next_Y = []
        for gen in self.generators:
            X, Y = next(gen)
            next_X.append(X)
            next_Y.append(Y)
        return np.concatenate(next_X), np.concatenate(next_Y)

train_generator = CombinedDirectoryIterator(
    directory=train_dir,
    image_data_generator=train_datagen,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,                # path to the test data directory
    target_size=(128, 128),  # resize images to this size
    batch_size=32,           # batch size (adjust based on memory constraints)
    class_mode='binary'      # since it's a binary classification problem
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), #, kernel_regularizer=l2(0.0005)),  # Reduced regularization strength
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # kernel_regularizer=l2(0.0005)),  # Reduced regularization strength
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),  # Reduced neurons & regularization strength
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=7,
    validation_data= test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping]
)

model.save('weapon_detection_model.h5')