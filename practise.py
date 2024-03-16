import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the CSV dataset
df = pd.read_csv('mnist.csv')

# Separate features (X) and labels (y)
X = df.drop('label', axis=1).values
y = df['label'].values

# Preprocess the data
X = X / 255.0  # Scale the pixel values to the range [0, 1]
y = to_categorical(y, num_classes=10)  # Convert labels to one-hot encoded vectors

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, input_shape=(784,), activation='tanh'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# Import the model
model.save('D:\Python\practise\mnist_model.h5')