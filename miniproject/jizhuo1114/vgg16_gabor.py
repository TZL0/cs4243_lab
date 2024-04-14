from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# def custom_preprocessing_Gabor(img):
#     # Convert the image from float [0,1] to uint8 [0,255]
#     img = (img * 255).astype(np.uint8)
#     # Convert to grey scale b4 Gabor
#     if img.shape[-1] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
#     # Apply 3 Gabor filter 0, 45, 90, 135
#     g_kernel0 = cv2.getGaborKernel((21, 21), 8.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#     filtered_img0 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel0)
#     g_kernel45 = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#     filtered_img45 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel45)
#     g_kernel90 = cv2.getGaborKernel((21, 21), 8.0, np.pi/4*3, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#     filtered_img90 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel90)
#     # Stack all 3 filtered img into an img with 4 channel
#     stacked_img = np.stack((filtered_img0, filtered_img45, filtered_img90), axis=-1)
#     # Convert the image back to float [0,1]
#     final_img = stacked_img.astype(np.float32) / 255.0
#     # Ensure values are between 0 and 1
#     final_img = np.clip(final_img, 0, 1)
#     final_img = np.clip(final_img, 0, 1)
#     return final_img

def new_custom_preprocessing(img):
    # Convert the image from float [0,1] to uint8 [0,255]
    img = (img * 255).astype(np.uint8)
    img = cv2.Laplacian(img, cv2.CV_8UC3, ksize=3)
    # Convert the image back to float [0,1]
    final_img = img.astype(np.float32) / 255.0
    # Ensure values are between 0 and 1
    final_img = np.clip(final_img, 0, 1)
    final_img = np.clip(final_img, 0, 1)
    return final_img


# Directory paths
train_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/train'
test_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/test'


# load a image
# img=cv2.imread('/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/train/weap/0200869_220904_carrying_9837_320.png')

# use gabor filter
# img = custom_preprocessing_Gabor(img)

# use new filter
# img = new_custom_preprocessing(img)

# # show the image using plt
# cv2.imshow('image',img)
# cv2.waitKey(0)

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    horizontal_flip=True,
    preprocessing_function=new_custom_preprocessing,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=new_custom_preprocessing
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

model = models.Sequential()
model.add(base_model)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add your own layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Add dropout layer to prevent overfitting
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(lr=1e-4),
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint('vgg16_gabor.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping, checkpoint]
)

model.load_weights('vgg16_gabor.h5')

# confusion matrix:
y_pred = model.predict(test_generator, steps=len(test_generator))
y_pred = np.round(y_pred).flatten()  # Round predictions to get binary classification results
y_true = test_generator.classes
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
