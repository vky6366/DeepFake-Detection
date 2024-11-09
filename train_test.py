import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,Dense, GlobalAveragePooling2D, TimeDistributed,Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
set_global_policy('mixed_float16')

# Assuming frames are extracted and stored in directories
train_dir = r"D:\DeepFake Detection System\Test_Model\Train_processed"  # This should contain subdirectories for each class, e.g., 'real' and 'fake'
test_dir = r"D:\DeepFake Detection System\Test_Model\Validation"
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Add vertical flip if applicable
    fill_mode='nearest',
    brightness_range=[0.5, 1.5],
    channel_shift_range=50.0,  # Adjust channel values
    featurewise_center=True,
    featurewise_std_normalization=True,
)


# Setting up the validation data generator (usually no augmentation is applied here)
validation_datagen = ImageDataGenerator(
    rescale=1./255  # Only rescale the pixel values for validation data
)

# Define the target size for the images
target_size_dim = (224, 224)  # VGG16 default input size

# Specify where to find the training and validation data, adjust paths as needed
train_generator = train_datagen.flow_from_directory(
    train_dir,       # Specify your training data directory
    target_size=target_size_dim, # The dimensions to which each image will be resized
    batch_size=50,               # Adjust the batch size according to your system's capability
    class_mode='binary'          # For binary classification
)

test_generator = validation_datagen.flow_from_directory(
    test_dir,  # Specify your validation data directory
    target_size=target_size_dim,
    batch_size=50,               # It's okay to have a larger batch size if memory permits, since no backprop is done
    class_mode='binary'
)

# Example to check the output of the generator
x, y = next(train_generator)
print("Batch shape:", x.shape)
print("Labels shape:", y.shape)

for layer in base_model.layers[:-15]:  # Only fine-tune the last 15 layers
    layer.trainable = False

model = Sequential([
    TimeDistributed(base_model, input_shape=(sequence_length, 224, 224, 3)),  # Apply CNN to each frame
    TimeDistributed(GlobalAveragePooling2D()),  # Pooling
    LSTM(128, return_sequences=False),  # Temporal analysis
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: real or fake
])


learning_rate = 1e-5  # Adjust this value based on your needs
optimizer = Adam(learning_rate=learning_rate,clipnorm=1.0)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

# Summarize the model to check the output shapes and parameters including regularization
model.summary()



early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Assuming train_generator and validation_generator are correctly set up
history = model.fit(
    train_generator,  # Assuming you've set up a data generator for training
    epochs=20,
    validation_data=test_generator,  # Validation set for early stopping
    callbacks=[early_stopping, checkpoint],
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Use the model to predict on new data
test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}, Precision: {test_precision}, Recall: {test_recall}, AUC: {test_auc}")