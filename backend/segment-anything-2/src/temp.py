import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import base64
from io import BytesIO
import os 
from dotenv import load_dotenv
import requests
import json

@st.cache_data
def load_runpod_info():
    load_dotenv("../../.env")
    
    url = os.getenv("SAM2_ENDPOINT")
    key = os.getenv("RUNPOD_KEY")

    return url, key

def scale_point(point, from_size, to_size):
    return (
        int(point[0] * to_size[0] / from_size[0]),
        int(point[1] * to_size[1] / from_size[1])
    )

def create_colored_mask_image(mask, R, G, B, A):
    R, G, B, A = map(lambda x: max(0, min(255, x)), [R, G, B, A])
    
    if mask.ndim > 2:
        mask = mask[0]

    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    rgba_image[mask == 1] = [R, G, B, A]
    image = Image.fromarray(rgba_image, 'RGBA')
    
    return image

def image_to_base64(image: Image) -> str:
    buffer = BytesIO()
    
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return image_base64

def load_image_from_base64(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image

endpoint_url, key = load_runpod_info()

def SAM2(image: Image, points: np.array, labels: np.array, rgba: tuple):
    image_b64 = image_to_base64(image)

    points = points.tolist()
    labels = labels.tolist()

    url = endpoint_url

    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }

    input_dict = {
        "input": {
            "image": image_b64,
            "points": points,
            "labels": labels,
            "R": 50,
            "G": 50,
            "B": 50,
            "A": 255
        }
    }

    payload = json.dumps(input_dict)

    response = requests.post(
        url = url,
        headers = headers,
        data = payload
    )

    mask_b64 = json.loads(response.text)["output"]["mask"]

    return load_image_from_base64(mask_b64)

def process_image(prefix):
    if f"{prefix}_image" in st.session_state:
        original_image = st.session_state[f"{prefix}_original_image"]
        resized_image = st.session_state[f"{prefix}_image"]
        width, height = 256, 256
        
        st.write(f"Draw points on the {prefix} image:")
        
        draw_mode = st.radio(f"Select the point type for {prefix}:", ("Green +", "Red -"), key=f"{prefix}_draw_mode")
        stroke_color = "#00FF00" if draw_mode == "Green +" else "#FF0000"

        if f"{prefix}_green_points" not in st.session_state:
            st.session_state[f"{prefix}_green_points"] = []
        if f"{prefix}_red_points" not in st.session_state:
            st.session_state[f"{prefix}_red_points"] = []

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  
            stroke_width=3,
            stroke_color=stroke_color,
            background_image=resized_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="point",
            key=f"{prefix}_canvas",
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

            st.session_state[f"{prefix}_green_points"] = current_green_points
            st.session_state[f"{prefix}_red_points"] = current_red_points

        if st.button(f"Create Mask for {prefix.capitalize()} Image"):
            create_mask(prefix)

        if f"{prefix}_mask" in st.session_state:
            mask_overlay = create_mask_overlay(resized_image, st.session_state[f"{prefix}_mask"])
            st.image(mask_overlay, caption=f"{prefix.capitalize()} Image with Transparent Mask")
            
            st.session_state[f"{prefix}_mask_created"] = True

def create_mask(prefix):
    original_image = st.session_state[f"{prefix}_original_image"]
    resized_image = st.session_state[f"{prefix}_image"]
    
    original_size = original_image.size
    new_size = (512, 512)  
    display_size = resized_image.size  

    green_points = [scale_point(p, display_size, new_size) for p in st.session_state[f"{prefix}_green_points"]]
    red_points = [scale_point(p, display_size, new_size) for p in st.session_state[f"{prefix}_red_points"]]

    sam_points = green_points + red_points
    sam_labels = [1] * len(green_points) + [0] * len(red_points)

    sam2_input_image = original_image.resize(new_size)

    with st.spinner(f"Generating mask for {prefix} image"):
        mask = SAM2(image=sam2_input_image, points=np.array(sam_points), labels=np.array(sam_labels), rgba=(50, 50, 50, 255))
    
    st.session_state[f"{prefix}_mask"] = mask.resize(display_size)

def create_mask_overlay(image, mask, opacity=0.5):
    mask_np = np.array(mask.convert('L'))
    mask_np = cv2.resize(mask_np, (256, 256))

    mask_color = [128, 128, 128, int(255 * opacity)]
    mask_overlay = np.zeros((256, 256, 4), dtype=np.uint8)
    mask_overlay[mask_np > 0] = mask_color

    return Image.fromarray(mask_overlay, 'RGBA')

def process_image_drawing(prefix, column, show_base_mask=True, show_ref_mask=False):
    if f"{prefix}_canvas_data" not in st.session_state:
        st.session_state[f"{prefix}_canvas_data"] = {
            "version": "4.4.0",
            "objects": []
        }
    image = st.session_state[f"{prefix}_image"]
    mask = st.session_state.get(f"{prefix}_mask")
    
    width, height = 256, 256
    
    with column:
        st.write(f"Draw on the {prefix} image:")
        
        drawing_mode = st.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform"),
            key = f'{prefix}_select_box'
        )

        stroke_width = st.slider("Stroke width:", 1, 150, 75, key=f"{prefix}_stroke_width")
        stroke_color = st.color_picker("Stroke color:", "#646464", key=f"{prefix}_stroke_color")

        initial_drawing = {"version": "4.4.0", "objects": []}

        if prefix == 'base':
            if show_base_mask and mask is not None:
                mask_overlay = create_mask_overlay(image, mask)
                mask_b64 = image_to_base64(mask_overlay)
                initial_drawing["objects"].append(create_image_object(width, height, mask_b64))

            if show_ref_mask and "reference_updated_mask" in st.session_state:
                ref_mask_overlay = create_mask_overlay(image, st.session_state["reference_updated_mask"])
                ref_mask_b64 = image_to_base64(ref_mask_overlay)
                initial_drawing["objects"].append(create_image_object(width, height, ref_mask_b64))

        elif prefix == 'reference' and mask is not None:
            mask_overlay = create_mask_overlay(image, mask)
            mask_b64 = image_to_base64(mask_overlay)
            initial_drawing["objects"].append(create_image_object(width, height, mask_b64))

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key=f"{prefix}_canvas",
            initial_drawing=initial_drawing
        )
        
        if canvas_result.image_data is not None:
            st.session_state[f"{prefix}_updated_mask"] = Image.fromarray(canvas_result.image_data[:,:,3], mode='L')
            
def create_image_object(width, height, image_b64):
    return {
        "type": "image",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": 0,
        "top": 0,
        "width": width,
        "height": height,
        "fill": "rgb(0,0,0)",
        "stroke": None,
        "strokeWidth": 0,
        "strokeDashArray": None,
        "strokeLineCap": "butt",
        "strokeDashOffset": 0,
        "strokeLineJoin": "miter",
        "strokeUniform": False,
        "strokeMiterLimit": 4,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "shadow": None,
        "visible": True,
        "backgroundColor": "",
        "fillRule": "nonzero",
        "paintFirst": "fill",
        "globalCompositeOperation": "source-over",
        "skewX": 0,
        "skewY": 0,
        "cropX": 0,
        "cropY": 0,
        "src": f"data:image/png;base64,{image_b64}",
        "crossOrigin": None,
        "filters": []
    }

def fetch_and_resize_image(source, max_size=512):
    if source.startswith('http'):
        response = requests.get(source)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = load_image_from_base64(source).convert("RGB")
    
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
    return image

def main():
    streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

            html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            }
            </style>
            """
    st.markdown(streamlit_style, unsafe_allow_html=True)

    st.title("SAM2: Point-Based and Draw")

    query_params = st.query_params
    base_url = query_params.get("base_url", "")
    reference_url = query_params.get("reference_url", "")
    email = query_params.get("email", "")

    if not base_url or not reference_url:
        st.error("No image URL provided in query parameters.")
        return

    # Initialize session state variables
    if 'base_original_image' not in st.session_state:
        st.session_state.base_original_image = fetch_and_resize_image(base_url)
        st.session_state.base_image = st.session_state.base_original_image.resize((256, 256))
    
    if 'reference_original_image' not in st.session_state:
        st.session_state.reference_original_image = fetch_and_resize_image(reference_url)
        st.session_state.reference_image = st.session_state.reference_original_image.resize((256, 256))

    if 'base_mask_created' not in st.session_state:
        st.session_state.base_mask_created = False
    
    if 'reference_mask_created' not in st.session_state:
        st.session_state.reference_mask_created = False

    page = st.sidebar.radio("Select Mode", ["Point", "Draw"])

    if page == "Point":
        st.subheader("Reference Image")
        process_image("reference")
        
        st.subheader("Base Image")
        process_image("base")

        if st.session_state.reference_mask_created and st.session_state.base_mask_created:
            st.success("Both masks have been created. You can now switch to the Draw mode.")

    elif page == "Draw":
        if not st.session_state.base_mask_created or not st.session_state.reference_mask_created:
            st.warning("Please create masks for both images in the Point mode first.")
        else:
            col1, col2 = st.columns(2)
            show_base_mask = st.checkbox("Show Base Mask", value=True, key="show_base_mask")
            show_ref_mask = st.checkbox("Show Reference Mask on Base", value=False, key="show_ref_mask")

            with col1:
                st.subheader("Base Image")
                process_image_drawing("base", col1, show_base_mask, show_ref_mask)

            with col2:
                st.subheader("Reference Image")
                process_image_drawing("reference", col2)

            if show_ref_mask:
                st.session_state["base_canvas_data"] = st.session_state.get("reference_canvas_data", {"version": "4.4.0", "objects": []})

            if st.button("Submit Mask Data"):
                if "base_updated_mask" in st.session_state and "reference_updated_mask" in st.session_state:
                    base_mask = np.array(st.session_state["base_updated_mask"])
                    ref_mask = np.array(st.session_state["reference_updated_mask"])

                    buffered_base = BytesIO()
                    Image.fromarray(base_mask).save(buffered_base, format="PNG")
                    base64_base = base64.b64encode(buffered_base.getvalue()).decode()

                    buffered_ref = BytesIO()
                    Image.fromarray(ref_mask).save(buffered_ref, format="PNG")
                    base64_ref = base64.b64encode(buffered_ref.getvalue()).decode()

                    response = requests.post(
                        os.getenv('BACKEND_URL'),
                        json={
                            "base_image": base_url,
                            "reference_image": reference_url,
                            "base_mask": base64_base,
                            "reference_mask": base64_ref,
                            "email": email
                        }
                    )

                    if response.status_code == 200:
                        st.success("Mask data submitted successfully!")
                    else:
                        st.error(f"Failed to submit mask data. Status code: {response.status_code}")
                else:
                    st.error("Please create masks for both images before submitting.")

if __name__ == "__main__":
    main()