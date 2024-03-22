import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
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
    canvas_result = st_canvas(
        fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="rgb(0, 0, 0)",  # Black color
        background_color="rgb(255, 255, 255)",  # White background
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Predict the digit when the user submits the drawing
    if st.button('Predict'):
        # Get the user input image
        user_input_image = canvas_result.image_data
        if user_input_image is not None:
            # Convert the image data to PIL Image
            img = Image.fromarray(user_input_image.astype('uint8'), 'RGB')
            # Preprocess the image
            preprocessed_img = preprocess_image(img)
            # Predict the digit
            prediction = model.predict(preprocessed_img)
            # Display the prediction result
            st.write('Prediction:', np.argmax(prediction))

if __name__ == '__main__':
    main()
