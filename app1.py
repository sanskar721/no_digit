import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('D:\Python\practise\mnist_model.h5')

# Function to preprocess the user input image
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Reshape the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit interface
def main():
    st.title('MNIST Digit Recognizer')

    # User input for drawing the digit
    st.write("Please draw a digit between 0 and 9")
    # Create a canvas to draw the digit
    canvas = st.image(
        np.zeros((150, 150)),
        caption="Draw here",
        use_column_width=True,
        channels="L"
    )

    # Predict the digit when the user submits the drawing
    if st.button('Predict'):
        # Get the user input image
        user_input_image = canvas.image_data
        if user_input_image is not None:
            # Convert the image data to PIL Image
            img = Image.fromarray(user_input_image, 'L')
            # Preprocess the image
            preprocessed_img = preprocess_image(img)
            # Predict the digit
            prediction = model.predict(preprocessed_img)
            # Display the prediction result
            st.write('Prediction:', np.argmax(prediction))

if __name__ == '__main__':
    main()
