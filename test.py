import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import time

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to run and time model training
def run_benchmark(device, epochs=5):
    with tf.device(device):
        model = create_model()
        start_time = time.time()
        model.fit(train_images, train_labels, batch_size=128, epochs=epochs, verbose=1, validation_data=(test_images, test_labels))
        end_time = time.time()
        print(f"Training time on {device}: {end_time - start_time:.2f} seconds")

# Run benchmarks
print("Benchmarking on CPU...")
run_benchmark('/cpu:0')

if tf.config.list_physical_devices('GPU'):
    print("Benchmarking on GPU...")
    run_benchmark('/gpu:0')
else:
    print("No GPU found, skipping GPU benchmark.")
