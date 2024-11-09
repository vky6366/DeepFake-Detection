import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

# Define your parameters
batch_size_test = 16  # Batch size for testing
sequence_length = 10  # Number of frames in a sequence
input_shape = (224, 224, 3)  # Shape of each frame (image)

# Create dummy data with shape (batch_size, sequence_length, height, width, channels)
dummy_data = tf.random.uniform((batch_size_test, sequence_length, 224, 224, 3))

# Force all layers to run on CPU
with tf.device('/CPU:0'):
    # Load pre-trained Xception model without the top layer, which expects input shape (224, 224, 3)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Build the Sequential model with TimeDistributed for applying CNN on each frame in the sequence
    model = Sequential([
        TimeDistributed(base_model, input_shape=(sequence_length, 224, 224, 3)),  # Apply Xception to each frame
        TimeDistributed(GlobalAveragePooling2D()),  # Pooling for each frame
        SimpleRNN(128, return_sequences=False),  # Replace GRU with SimpleRNN
        Dense(128, activation='relu'),  # Fully connected layer
        Dropout(0.5),  # Dropout for regularization
        BatchNormalization(),
        Dense(1, activation='sigmoid')  # Binary classification output (real vs fake)
    ])

# Forward pass with dummy data to check if the batch size works
try:
    output = model(dummy_data)
    print(f"Batch size {batch_size_test} works fine.")
except tf.errors.ResourceExhaustedError:
    print(f"Batch size {batch_size_test} is too large for your system.")
except Exception as e:
    print(f"Error encountered: {e}")
