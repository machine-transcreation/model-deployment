import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

st.title("Point Plotter with Image Upload")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_width, img_height = image.size
else:
    img_width, img_height = 400, 400

# Radio buttons for point selection
point_color = st.radio("Select point color:", ("Red", "Green"))

# Canvas for plotting points
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.8)" if point_color == "Red" else "rgba(0, 255, 0, 0.8)",
    stroke_width=1,
    stroke_color="rgba(0, 0, 0, 0)",
    background_color="#eee",
    background_image=image if uploaded_file else None,
    update_streamlit=True,
    height=img_height,
    width=img_width,
    drawing_mode="point",
    point_display_radius=10,
    key="canvas",
)

# Initialize session state for storing points
if 'red_points' not in st.session_state:
    st.session_state.red_points = []
if 'green_points' not in st.session_state:
    st.session_state.green_points = []

# Update points based on canvas result
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    if len(objects) > 0:
        last_object = objects[-1]
        if point_color == "Red":
            st.session_state.red_points.append((last_object["left"], last_object["top"]))
        else:
            st.session_state.green_points.append((last_object["left"], last_object["top"]))

# Create Mask button
if st.button("Create Mask"):
    st.write("Red points coordinates:")
    for point in st.session_state.red_points:
        st.write(point)
    
    st.write("Green points coordinates:")
    for point in st.session_state.green_points:
        st.write(point)

# Clear points button
if st.button("Clear Points"):
    st.session_state.red_points = []
    st.session_state.green_points = []
    st.experimental_rerun()