#do this after saving the model and installing streamlit

streamlit run "c:/users/riya kumari/desktop/minorproject.py"
#run this on control panel



import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the pre-trained CNN model
model = tf.keras.models.load_model('C:/Users/Rishiraj Saha/Downloads/my_model.h5')  # Replace with the path to your model

# Define class labels
class_labels = ['Potato__Early_blight', 'Potato_healthy', 'Potato__Late_blight']  # Replace with your class labels

# Create a Streamlit app
st.title("Plant vrop disease prediction with CNN")

# Upload an image for classification
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((256, 256))  # Resize to match your model's input size
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
# Make predictions
    predictions = model.predict(img)
    predicted_class = class_labels[predictions.argmax()]

    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"Predicted class: {predicted_class}")
    st.write("Confidence scores:")
    for i, class_label in enumerate(class_labels):
        st.write(f"{class_label}: {predictions[0][i]:.2f}")
