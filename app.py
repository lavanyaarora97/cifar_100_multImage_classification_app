import streamlit as st
import zipfile
import io
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import base64

# Replace 'your_model_file.h5' with the actual filename of your model
model = tf.keras.models.load_model('finetunedCifar100.h5')

def extract_images_from_zip(zip_file, save_directory):
    extracted_files = []

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                with zip_ref.open(file_name) as file:
                    image_data = io.BytesIO(file.read())
                    image = Image.open(image_data)
                    image_path = os.path.join(save_directory, file_name)
                    image.save(image_path)  # Save image to specified directory
                    extracted_files.append(image_path)

    return extracted_files



# Load and preprocess a single image
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')  # Set color_mode to 'rgb'
    
    
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255  # Normalize pixel values to [0, 1]
    return image_array


def load_and_preprocess_images(image_paths, target_size=(32, 32)):
    images = []
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path, target_size=target_size)
        images.append(image)
    return tf.convert_to_tensor(images)


class_name = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def preprocess():
    # Example usage:
    image_dir = 'modelimages'
    image_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, f) for f in image_files]

    images = load_and_preprocess_images(image_paths)
    predictions = model.predict(images)
    pre = [np.argmax(element) for element in predictions]
    confidence_scores = tf.reduce_max(predictions, axis=1)


    confidence_scores = confidence_scores.numpy()

    num_images = len(images)
    num_rows = (num_images + 4) // 5  # Round up to the nearest multiple of 5

    # Create a plot
    fig, axes = plt.subplots(num_rows, 5, figsize=(8, num_rows * 3))

    # Display each image with its predicted class and accuracy
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')
            
            predicted_label = class_name[pre[i]]
            confidence_score = confidence_scores[i]
            ax.set_title(f"Predicted Class: {predicted_label}\nAccuracy: {confidence_score:.2f}", fontsize=7)

    # Adjust layout to avoid overlap and add spacing
    plt.tight_layout(h_pad=1.0, w_pad=4.0)  # Adjust spacing here

    # Display the plot using Streamlit
    st.pyplot(fig)


    plt.figure(figsize=(10, 6))
    sns.countplot(x=pre, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.title('Predicted Class Distribution')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Display the plot using Streamlit
    st.pyplot()


    plt.figure(figsize=(10, 6))
    sns.histplot(pre, bins=20, kde=True, color='skyblue')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    
    # Display the plot using Streamlit
    st.pyplot()

    
    # Calculate the count of unique predicted classes
    unique_classes, class_counts = np.unique(pre, return_counts=True)

    # Calculate the total number of classes
    total_classes = len(class_name)
     # Calculate the percentage of classes found during prediction
    percentage_classes_found = (len(unique_classes) / total_classes) * 100

    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.bar(['Classes Found'], [percentage_classes_found], color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Percentage of Classes Found (%)')
    plt.title('Percentage of Classes Found during Prediction')
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100 for percentage
    
    # Display the plot using Streamlit
    st.pyplot()

    
    with PdfPages('analysissi.pdf') as pdf:
        
        # Add text "CIFAR-100 Image Classification Analysis"
        plt.figure(figsize=(8, 1))
        plt.text(0.5, 0.8, 'CIFAR-100 bulk Image Classification Analysis', ha='center', va='center', fontsize=16)
        plt.text(0.5, 0.8, f'\n\n\nNumber of Images classified: {num_images}', ha='center', va='center', fontsize=16)
        plt.axis('off')  # Turn off axis
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()    # Close the figure to free up memory

        # Predicted Class Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=pre, palette='viridis')
        plt.xticks(rotation=90)
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title('Predicted Class Distribution')
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()    # Close the figure to free up memory
        
        # Prediction Confidence Distribution
        predicted_confidence = np.max(predictions, axis=1)  # Assuming 'predictions' is the output of your model
        plt.figure(figsize=(10, 6))
        sns.histplot(pre, bins=20, kde=True, color='skyblue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Prediction Confidence Distribution')
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()    # Close the figure to free up memory

        # Calculate the count of unique predicted classes
        unique_classes, class_counts = np.unique(pre, return_counts=True)

        # Calculate the total number of classes
        total_classes = len(class_name)

        # Calculate the percentage of classes found during prediction
        percentage_classes_found = (len(unique_classes) / total_classes) * 100

        # Plot the graph
        plt.figure(figsize=(8, 6))
        plt.bar(['Classes Found'], [percentage_classes_found], color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Percentage of Classes Found (%)')
        plt.title('Percentage of Classes Found during Prediction')
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()    # Close the figure to free up memory


      # Button to download the analysis PDF
      # Button to download the analysis PDF
        # Display link to open the PDF in the browser
        def display_pdf(file):
            # Read the PDF file as bytes
            with open(file, "rb") as f:
                pdf_bytes = f.read()

            # Encode the bytes to base64
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

            # Create the HTML embed tag
            pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf">'

            # Display the PDF using markdown
            st.markdown(pdf_display, unsafe_allow_html=True)

        
        display_pdf("analysis.pdf")
        






    ##st.write(pre)

def main():
    st.title('CIFAR 100 BULK IMAGE CLASSIFICATION AND ANALYSIS')
    
    uploaded_file = st.sidebar.file_uploader("Upload a zip file", type=["zip"])

    if uploaded_file is not None:
        st.success('Zip file uploaded successfully!')

        # Create modelimages directory if it doesn't exist
        save_directory = "modelimages"
        os.makedirs(save_directory, exist_ok=True)

        # Extract and save images from the uploaded zip file
        extracted_files = extract_images_from_zip(uploaded_file, save_directory)
        
        preprocess()
        

if __name__ == '__main__':
    main()
