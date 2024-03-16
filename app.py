from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the pre-trained model
model = load_model('D:\Python\practise\mnist_model.h5')

# Initialize a variable to store the current image data
current_image_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global current_image_data

    # Get image data from the request
    image_data = request.json['image_data']

    # Check if the canvas is cleared
    if not image_data:
        # Clear the stored image data
        current_image_data = None
        return jsonify({'prediction': ''})

    # Decode the base64 image data and convert to grayscale
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')

    # Resize image to 28x28 pixels (MNIST model input size)
    image = image.resize((28, 28))

    # Convert image to numpy array and normalize
    image_array = np.array(image) / 255.0

    # Reshape array to match model input shape
    image_array = image_array.reshape(1, 784)

    # Make prediction
    prediction = np.argmax(model.predict(image_array))

    # Store the current image data
    current_image_data = image_data

    return jsonify({'prediction': str(prediction)})

@app.route('/clear', methods=['POST'])
def clear():
    global current_image_data
    # Clear the stored image data
    current_image_data = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
