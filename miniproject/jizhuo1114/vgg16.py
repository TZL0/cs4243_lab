from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Directory paths
train_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/train'
test_dir = '/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/test'

# # Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=40,
    horizontal_flip=True,
    fill_mode='nearest'
)
#
test_datagen = ImageDataGenerator(
    rescale=1./255
)
#
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)
#
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
#
# # Load pre-trained VGG16 model + higher level layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
#
# model = models.Sequential()
# model.add(base_model)
#
# # Freeze the layers of the pre-trained model
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Add your own layers
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.8))  # Add dropout layer to prevent overfitting
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # Compile the model
# model.compile(
#     loss='binary_crossentropy',
#     # optimizer=optimizers.Adam(lr=1e-4),
#     optimizer='adam',
#     metrics=['accuracy']
# )

# Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# checkpoint = ModelCheckpoint('vgg16.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=10,
#     validation_data=test_generator,
#     validation_steps=len(test_generator),
#     callbacks=[early_stopping, checkpoint]
# )


### Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Load the best model
best_model = models.load_model('vgg16.h5')
# 2. Predict the classes for the test dataset
test_generator.reset()  # Reset the test_generator index before predicting
predictions = best_model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Get the true classes
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys()) 

# 3. Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# 4. Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# # confusion matrix:
# y_pred = model.predict(test_generator, steps=len(test_generator))
# y_pred = np.round(y_pred).flatten()  # Round predictions to get binary classification results
# y_true = test_generator.classes
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_true, y_pred)
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()
