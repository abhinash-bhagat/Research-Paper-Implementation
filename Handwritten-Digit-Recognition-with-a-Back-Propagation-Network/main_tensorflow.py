import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [-1, 1]
train_images = train_images / 255.0 * 2 - 1
test_images = test_images / 255.0 * 2 - 1

# Add a channel dimension to the images for compatibility with Conv2D
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build the CNN model
model = models.Sequential([
    # First convolutional layer with 6 filters of 5x5 and ReLU activation
    layers.Conv2D(6, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling

    # Second convolutional layer with 16 filters
    layers.Conv2D(16, kernel_size=5, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the output from convolutional layers
    layers.Flatten(),

    # Fully connected layer with 120 units
    layers.Dense(120, activation='relu'),

    # Fully connected layer with 84 units
    layers.Dense(84, activation='relu'),

    # Output layer with 10 units (for the 10 digit classes)
    layers.Dense(10, activation='softmax')  # softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='sgd',  # Use Stochastic Gradient Descent
              loss='sparse_categorical_crossentropy',  # Cross-entropy loss
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
