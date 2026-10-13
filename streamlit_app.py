
import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
import config


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantAI Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f8f5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #14532d, #16a34a);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 42px;
}

.hero p {
    color: #e8f5e9;
    font-size: 18px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e0e8e0;
}

.section-title {
    color: #166534;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD EXISTING TRAINED MODEL
# ==========================================

@st.cache_resource
def load_ai_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {config.MODEL_PATH}"
        )

    if not os.path.exists(config.CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"Class index file not found: {config.CLASS_INDICES_PATH}"
        )

    model = keras.models.load_model(config.MODEL_PATH)

    with open(
        config.CLASS_INDICES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_indices = json.load(file)

    class_indices = {
        int(key): value
        for key, value in class_indices.items()
    }

    return model, class_indices


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(config.IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ==========================================
# FORMAT DISEASE CLASS NAME
# ==========================================

def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🌿 PlantAI Doctor")
st.sidebar.caption("AI-Powered Plant Disease Detection")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Diagnose Plant", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a plant leaf image to obtain an AI-generated "
    "disease prediction and care information."
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🌿 PlantAI Doctor</h1>
        <p>
            Automated Plant Disease Detection and Diagnosis
        </p>
        <p>
            Analyze plant leaf images using your trained
            Convolutional Neural Network (CNN) model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to PlantAI Doctor")

    st.write(
        "This project uses a trained deep learning model to "
        "predict possible plant leaf conditions. It also "
        "provides disease information, symptoms, causes, "
        "treatment guidance and prevention tips."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Technology", "CNN")

    with col2:
        st.metric("Supported Crops", "5")

    with col3:
        st.metric("Image Input", "JPG / PNG")

    st.markdown("---")

    st.subheader("🌱 Supported Crops")

    crop_columns = st.columns(4)

    crops = [
        ("🍅", "Tomato"),
        ("🥔", "Potato"),
        ("🌽", "Corn / Maize"),
        ("🍎", "Apple"),
        ("🍇", "Grape")
    ]

    for index, (emoji, crop) in enumerate(crops):
        with crop_columns[index % 4]:
            st.markdown(f"### {emoji}")
            st.write(crop)

    st.markdown("---")

    st.subheader("How It Works")

    st.write("1. Upload a clear photograph of a plant leaf.")
    st.write("2. The trained CNN model analyzes the image.")
    st.write("3. View the predicted class and confidence score.")
    st.write("4. Review relevant symptoms and care information.")

    if st.button("🔍 Start Plant Diagnosis", type="primary"):
        st.session_state["current_page"] = "Diagnose Plant"
        st.rerun()


# ==========================================
# DIAGNOSIS PAGE
# ==========================================

elif page == "Diagnose Plant":

    st.title("🔬 Plant Disease Diagnosis")

    st.write(
        "Upload a clear image of a tomato, potato, corn, "
        "apple or grape leaf."
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear, well-lit image focused on one leaf."
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            left_column, right_column = st.columns([1, 1])

            with left_column:
                st.subheader("Uploaded Leaf")
                st.image(image, use_container_width=True)

            with right_column:
                st.subheader("Image Information")
                st.write(f"**File:** {uploaded_file.name}")
                st.write(
                    f"**Image dimensions:** "
                    f"{image.width} × {image.height}"
                )
                st.write(f"**Format:** {uploaded_file.type}")

            if st.button(
                "🌿 Predict Disease",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Loading AI model and analyzing the image..."
                ):

                    model, class_indices = load_ai_model()

                    input_image = preprocess_image(image)

                    predictions = model.predict(
                        input_image,
                        verbose=0
                    )[0]

                    if predictions.ndim != 1:
                        raise ValueError(
                            "Unexpected model output dimensions."
                        )

                    if len(predictions) != len(class_indices):
                        raise ValueError(
                            "Model classes do not match class_indices.json."
                        )

                    if not np.all(np.isfinite(predictions)):
                        raise ValueError(
                            "The model returned invalid prediction values."
                        )

                    predicted_index = int(np.argmax(predictions))

                    confidence = (
                        float(predictions[predicted_index]) * 100
                    )

                    raw_class_name = class_indices.get(
                        predicted_index,
                        "Unknown"
                    )

                    top_indices = np.argsort(predictions)[::-1][:3]

                    top_predictions = [
                        {
                            "name": format_class_name(
                                class_indices.get(int(index), "Unknown")
                            ),
                            "confidence": round(
                                float(predictions[index]) * 100, 2
                            )
                        }
                        for index in top_indices
                    ]

                    disease = config.get_disease_details(
                        raw_class_name
                    ).copy()

                    threshold = (
                        getattr(config, "CONFIDENCE_THRESHOLD", 0.60)
                        * 100
                    )

                    is_low_confidence = confidence < threshold

                    # Do not show disease severity as confirmed
                    # when model confidence is below the threshold.
                    if is_low_confidence:
                        disease["severity"] = "Unable to determine"
                        disease["status"] = "Uncertain"
                        disease["pathogen"] = "Not confirmed"

                st.session_state["plant_result"] = {
                    "raw_class_name": raw_class_name,
                    "confidence": round(confidence, 2),
                    "top_predictions": top_predictions,
                    "disease": disease,
                    "is_low_confidence": is_low_confidence,
                    "image": image
                }

        except Exception as error:
            st.error("The prediction could not be completed.")
            st.exception(error)

    # Show saved result after prediction
    result = st.session_state.get("plant_result")

    if result is not None:

        st.markdown("---")
        st.title("📋 Diagnostic Assessment Report")

        disease = result["disease"]
        confidence = result["confidence"]
        low_confidence = result["is_low_confidence"]

        if low_confidence:
            st.warning(
                f"Uncertain prediction: {confidence:.2f}% confidence. "
                "The model's prediction is below the configured "
                f"{threshold if 'threshold' in locals() else getattr(config, 'CONFIDENCE_THRESHOLD', 0.60) * 100:.0f}% "
                "threshold. Try a clearer leaf image and verify "
                "the result with an agricultural specialist."
            )
        else:
            st.info(
                "This is an AI-generated prediction, not a "
                "confirmed diagnosis."
            )

        result_left, result_right = st.columns([1, 2])

        with result_left:
            st.image(
                result["image"],
                caption="Analyzed leaf image",
                use_container_width=True
            )

            st.metric("Model Confidence", f"{confidence:.2f}%")

        with result_right:
            st.subheader(
                format_class_name(result["raw_class_name"])
            )

            st.write(f"**Plant:** {disease.get('plant', 'Unknown')}")
            st.write(f"**Disease:** {disease.get('disease_name', 'Unknown')}")
            st.write(f"**Health status:** {disease.get('status', 'Unknown')}")
            st.write(f"**Severity:** {disease.get('severity', 'Unknown')}")
            st.write(f"**Pathogen:** {disease.get('pathogen', 'Unknown')}")

        st.subheader("📊 Top Predictions")

        chart_data = {
            item["name"]: item["confidence"]
            for item in result["top_predictions"]
        }

        st.bar_chart(chart_data)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview & Cause",
            "Symptoms",
            "Treatment & Care",
            "Prevention"
        ])

        with tab1:
            st.subheader("Disease Overview")
            st.write(disease.get("description", "No description available."))

            st.subheader("Cause")
            st.write(disease.get("cause", "No cause information available."))

        with tab2:
            st.subheader("Symptoms to Check")

            for symptom in disease.get("symptoms", []):
                st.markdown(f"- {symptom}")

        with tab3:
            st.subheader("Recommended Care")
            st.write(
                disease.get(
                    "recommendation",
                    "Consult a local agricultural specialist."
                )
            )

            st.caption(
                "Treatment recommendations should be checked "
                "against local agricultural guidance and product labels."
            )

        with tab4:
            st.subheader("Prevention Tips")

            for tip in disease.get("prevention", []):
                st.markdown(f"- {tip}")

        if st.button("🗑️ Clear Current Result"):
            st.session_state.pop("plant_result", None)
            st.rerun()


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About PlantAI Doctor")

    st.write(
        "PlantAI Doctor is a plant disease detection project "
        "that uses a trained CNN image classification model."
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - **Python** — application programming
    - **TensorFlow / Keras** — deep learning model
    - **Streamlit** — web interface
    - **Pillow** — image processing
    - **NumPy** — numerical operations
    """)

    st.subheader("Model Information")

    st.write(f"Input image size: {config.IMAGE_SIZE}")

    st.write(
        "Configured confidence threshold: "
        f"{config.CONFIDENCE_THRESHOLD * 100:.0f}%"
    )

    st.warning(
        "The model is intended for educational and research use. "
        "Its real-world accuracy depends on its training data, "
        "validation results and image quality."
    )
