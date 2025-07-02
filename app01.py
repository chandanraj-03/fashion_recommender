import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

# Set page configuration with background image
st.set_page_config(
    page_title="Fashion Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for background image and styling
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-vector/hand-drawn-fashion-shop-pattern-background_23-2150849915.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
            margin-top: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.85);
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stRadio>div {
            flex-direction: row;
            gap: 1rem;
        }
        .stRadio>div>label {
            margin-bottom: 0;
        }
        .recommendation-header {
            color: #FF4B4B;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        .recommendation-image {
            border: 2px solid #FF4B4B;
            border-radius: 8px;
            transition: transform 0.3s;
        }
        .recommendation-image:hover {
            transform: scale(1.05);
        }
        .similarity-bar {
            height: 8px;
            background: #f0f0f0;
            border-radius: 4px;
            margin: 8px 0;
        }
        .similarity-progress {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #FF4B4B, #FF9E9E);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


set_background()


# Load precomputed embeddings and filenames
@st.cache_resource
def load_data():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('filenames.pkl', 'rb'))
        base_dir = os.getcwd()
        filenames = [os.path.join(base_dir, fname) if not os.path.isabs(fname) else fname for fname in filenames]
        return feature_list, filenames
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return None, None


feature_list, filenames = load_data()

if feature_list is None or filenames is None:
    st.stop()


# Load model
@st.cache_resource
def load_model():
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        return tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


model = load_model()
if model is None:
    st.stop()


# Function to save uploaded image
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"File upload error: {str(e)}")
        return None


# Function to extract features from uploaded image
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0).flatten()
        return features / (norm(features) + 1e-10)  # Normalize with small epsilon to avoid division by zero
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None


# Recommendation function with fixed similarity calculation
def recommend(features, feature_list, num_results=5):
    try:
        if len(feature_list) == 0:
            st.error("No features available for recommendation")
            return []

        # Calculate cosine similarities (more accurate for normalized vectors)
        similarities = np.dot(feature_list, features)

        # Get top matches (excluding the query image itself if it's in the dataset)
        top_indices = np.argsort(similarities)[::-1][:num_results + 1]

        # Filter out the query image if it exists in the dataset
        top_indices = [idx for idx in top_indices if not np.allclose(features, feature_list[idx])][:num_results]

        # Convert cosine similarities to percentages (0-100)
        similarity_percentages = [100 * (similarities[idx] + 1) / 2 for idx in
                                  top_indices]  # Scale from [-1,1] to [0,100]

        return list(zip(top_indices, similarity_percentages))
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return []


# Main app content
def main():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    st.title('👗 Fashion Recommender System')
    st.markdown("""
        <p style='font-size: 1.1rem;'>
        Discover similar fashion items based on your input. Upload an image or use your camera to find recommendations!
        </p>
        """, unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio(
        "Choose an input method:",
        ("Upload Image", "Use Camera"),
        horizontal=True,
        key='input_method'
    )

    file_path = None

    # Process based on selected input method
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "📤 Upload an image of a fashion item",
            type=['jpg', 'png', 'jpeg'],
            help="Upload an image of clothing or accessories you like",
            key='file_uploader'
        )
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
    else:
        camera_image = st.camera_input(
            "📷 Take a photo of a fashion item",
            help="Point your camera at an item you'd like to find similar products for",
            key='camera_input'
        )
        if camera_image is not None:
            file_path = save_uploaded_file(camera_image)

    # Process uploaded file or camera image
    if file_path is not None:
        try:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Your Input Image")
                st.image(Image.open(file_path),
                         caption="Uploaded Image",
                         use_container_width=True,
                         output_format='JPEG')

            with st.spinner('Analyzing your image and finding recommendations...'):
                # Extract features
                features = feature_extraction(file_path, model)

                if features is not None and isinstance(features, np.ndarray):
                    with col2:
                        st.success("✅ Image processed successfully!")

                    # Recommendation settings
                    st.subheader("Recommendation Settings")
                    num_results = st.slider(
                        "Number of recommendations",
                        min_value=1,
                        max_value=min(20, len(filenames)),
                        value=5,
                        help="Choose how many similar items you want to see"
                    )

                    show_distances = st.checkbox(
                        "Show similarity scores",
                        value=True,
                        help="Display how similar each recommendation is to your input"
                    )

                    # Get recommendations with similarity scores
                    recommendations = recommend(features, feature_list, num_results)

                    # Display recommended images
                    if recommendations:
                        st.markdown("<h3 class='recommendation-header'> Similar Products You Might Like</h3>",
                                    unsafe_allow_html=True)

                        # Create columns for recommendations
                        cols = st.columns(len(recommendations))

                        for idx, (rec_idx, similarity) in enumerate(recommendations, 1):
                            with cols[idx - 1]:
                                try:
                                    if 0 <= rec_idx < len(filenames):
                                        if os.path.exists(filenames[rec_idx]):
                                            img = Image.open(filenames[rec_idx])
                                            st.image(
                                                img,
                                                caption=f"Product {idx}",
                                                use_container_width=True,
                                                output_format='JPEG',
                                                width=200
                                            )

                                            if show_distances:
                                                # Custom similarity bar with better visualization
                                                st.markdown(
                                                    f"""
                                                    <div style="margin-top: 10px;">
                                                        <div style="display: flex; justify-content: space-between;">
                                                            <span>Similarity:</span>
                                                            <span><strong>{similarity:.1f}%</strong></span>
                                                        </div>
                                                        <div class="similarity-bar">
                                                            <div class="similarity-progress" style="width: {similarity}%"></div>
                                                        </div>
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True
                                                )
                                        else:
                                            st.error("Image file not found")
                                    else:
                                        st.error("Invalid product index")
                                except Exception as e:
                                    st.error(f"Error displaying product: {str(e)}")
                    else:
                        st.warning("⚠️ No recommendations could be generated for this image.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing your image: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()