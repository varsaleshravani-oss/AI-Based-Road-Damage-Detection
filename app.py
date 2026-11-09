import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# Application title
st.title("🚧 AI-Based Road Damage Detection System")
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }

    h1 {
        color: #16324f;
        text-align: center;
    }

    .stButton > button {
        background-color: #16324f;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #245b85;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.write(
    "Upload a road image to detect road damage using our trained YOLO11 model."
)

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("best_v2.pt")

model = load_model()

st.success("Road damage detection model loaded successfully!")

# Image upload
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)
# Detect road damage
# Detect road damage
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Road Image")
    st.image(image, use_container_width=True)
    confidence = st.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)
    if st.button("Detect Road Damage"):

        with st.spinner("Detecting road damage..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640
            )

            result = results[0]

            # Display annotated image
            annotated_image = result.plot()

            st.subheader("Detection Results")

            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

            # Count detected objects
            total_damages = len(result.boxes)

            st.metric(
                label="Total Road Damages Detected",
                value=total_damages
            )

            if total_damages > 0:

                st.success("Road damage detected successfully!")

                # Prepare detection details
                detection_data = []

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    detection_data.append({
                        "Damage Class": class_name,
                        "Confidence": f"{confidence:.2%}"
                    })

                # Display table
                st.subheader("Damage Detection Details")

                st.table(detection_data)

                # Download annotated image
                output_image = Image.fromarray(
                    annotated_image[:, :, ::-1]
                )

                image_buffer = BytesIO()
                output_image.save(image_buffer, format="JPEG")

                st.download_button(
                    label="Download Detected Image",
                    data=image_buffer.getvalue(),
                    file_name="road_damage_detection.jpg",
                    mime="image/jpeg"
                )

            else:
               
                st.warning(
                    "No road damage detected above the selected confidence threshold."
                  )
                st.info(
                          "Try lowering the confidence threshold or uploading another image."
                   )
                st.markdown("---")

st.subheader("About This Project")

st.write("""
This project uses a YOLO11 object detection model
to identify road damage from uploaded images.

It detects different types of road damage and displays
bounding boxes, confidence scores, and downloadable results.


""")
st.write("**Dataset:** RDD2022 India")
st.write("**Model:** YOLO11n")
st.write("**Framework:** Ultralytics")