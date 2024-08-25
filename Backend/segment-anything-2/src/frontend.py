import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

@st.cache_resource
def build_predictor():
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    return predictor

predictor = build_predictor()

def main():
    st.title("SAM2: Point-Based")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((512, 512))
        
        width, height = image.size
        
        st.write("Draw on the image:")
        
        draw_mode = st.radio("Select the point type:", ("Green +", "Red -"))
        if draw_mode == "Green +":
            stroke_color = "#00FF00"
        else:
            stroke_color = "#FF0000"

        if "green_points" not in st.session_state:
            st.session_state.green_points = []
        if "red_points" not in st.session_state:
            st.session_state.red_points = []

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  
            stroke_width=3,
            stroke_color=stroke_color,
            background_image=image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="point",
            point_display_radius=5,
            key="canvas",
        )

        if canvas_result.json_data is not None:
            current_green_points = []
            current_red_points = []

            for obj in canvas_result.json_data["objects"]:
                point = (obj["left"], obj["top"])
                if obj["stroke"] == "#00FF00":
                    current_green_points.append(point)
                elif obj["stroke"] == "#FF0000":  
                    current_red_points.append(point)

            st.session_state.green_points = current_green_points
            st.session_state.red_points = current_red_points

        if st.button("Create Mask"):
            sam_points = []
            sam_labels = []

            for point in st.session_state.green_points:
                sam_points.append(point)
                sam_labels.append(1)

            for point in st.session_state.red_points:
                sam_points.append(point)
                sam_labels.append(0)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(np.asarray(image))
                masks, _, _ = predictor.predict(point_coords = np.array(sam_points), point_labels = np.array(sam_labels), multimask_output = False)
                print(masks)

if __name__ == "__main__":
    main()
