import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt
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

    image_b64 = image_to_base64(image.resize((256, 256)))

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


def overlay(image, mask, borders=True):
    image_np = np.array(image)
    mask_np = np.array(mask.convert('L'))  

    h, w = image_np.shape[:2]
    mask_np = cv2.resize(mask_np, (w, h))

    color = np.array([50 / 255, 50 / 255, 50 / 255, 1])  

    mask_image = mask_np.reshape(h, w, 1) / 255.0 * color.reshape(1, 1, -1)

    if borders:
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(mask_image, alpha=0.6)
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    pil_image = Image.open(buf)
    buf.seek(0) 
    plt.close()

    return pil_image

def process_image(uploaded_file, prefix):
    if uploaded_file is not None:
        orig_image = Image.open(uploaded_file)
        orig_width, orig_height = orig_image.size
        
        image = orig_image.resize((256, 256))
        
        st.session_state[f"{prefix}_image"] = image

        width, height = image.size
        
        st.write(f"Draw on the {prefix} image:")
        
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
            background_image=image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="point",
            point_display_radius=5,
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

def create_mask(prefix):
    image = st.session_state[f"{prefix}_image"]
    sam_points = []
    sam_labels = []

    for point in st.session_state[f"{prefix}_green_points"]:
        sam_points.append(point)
        sam_labels.append(1)

    for point in st.session_state[f"{prefix}_red_points"]:
        sam_points.append(point)
        sam_labels.append(0)

    with st.spinner(f"Generating mask for {prefix} image"):
        mask = SAM2(image=image, points=np.array(sam_points), labels=np.array(sam_labels), rgba=(50, 50, 50, 255))
    
    st.image(overlay(image, mask), caption=f"{prefix.capitalize()} Image with Mask")

    st.session_state[f"{prefix}_mask"] = mask

def show_base_canvas(canvas_img_data):
    ref_mask_image = Image.fromarray(canvas_img_data)

    initial_drawing = {
        "version": "4.4.0",
        "objects": [
            {
                "type": "image",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": 0,
                "top": 0,
                "width": 256,
                "height": 256,
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
                "src": f"data:image/png;base64,{image_to_base64(ref_mask_image)}",
                "crossOrigin": None,
                "filters": []
            }
        ]
    }

    st.session_state["ref_mask_inital_drawing"] = initial_drawing
    st.session_state["show_base_canvas"] = True
    

def process_image_drawing(uploaded_file, prefix, column = None):
    if uploaded_file is not None:
        orig_image = Image.open(uploaded_file)
                
        image = orig_image.resize((256, 256))
        
        st.session_state[f"{prefix}_image"] = image

        width, height = image.size
        
        st.write(f"Draw on the {prefix} image:")
        
        drawing_mode = st.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform"),
            key = f'{prefix}_select_box'
        ) if (prefix == "ref_draw") else "transform"

        stroke_width = st.slider("Stroke width:", 1, 150, 75, key=f"{prefix}_stroke_width")
        stroke_color = st.color_picker("Stroke color:", "#B5B5B5", key=f"{prefix}_stroke_color")

        if (prefix == "ref_draw"):
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",  
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=image,
                update_streamlit=True,
                height=height,
                width=width,
                point_display_radius=5,
                key=f"{prefix}_canvas",
                drawing_mode = drawing_mode
            )
            
            st.button("Clone Mask", key = "ref_mask_btn", on_click = show_base_canvas(canvas_result.image_data))


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

    mode = st.sidebar.selectbox("Select Mode", ["Point", "Draw"])

    st.title("SAM2: Point-Based")

    if mode == "Point":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Base Image")
            base_file = st.file_uploader("Choose base image...", type=["jpg", "jpeg", "png"], key="base_image_uploader")
            if base_file is not None:
                if "base_image" not in st.session_state:
                    st.session_state["base_image"] = Image.open(base_file).resize((256, 256))
                process_image(base_file, "base")

        with col2:
            st.subheader("Reference Image")
            ref_file = st.file_uploader("Choose reference image...", type=["jpg", "jpeg", "png"], key="ref_image_uploader")
            if ref_file is not None:
                if "ref_image" not in st.session_state:
                    st.session_state["ref_image"] = Image.open(ref_file).resize((256, 256))

    
    if mode == "Draw":

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Base Image")
            base_file = st.file_uploader("Choose base image...", type=["jpg", "jpeg", "png"], key="base_image_uploader")
            if base_file is not None:
                if "base_image" not in st.session_state:
                    st.session_state["base_image"] = Image.open(base_file).resize((256, 256))

                if ("show_base_canvas" not in st.session_state):
                    st.session_state["show_base_canvas"] = False

                if (st.session_state["show_base_canvas"]):
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0, 0)",  
                        stroke_width=1,
                        stroke_color="#ffffff",
                        background_image=st.session_state["base_image"],
                        update_streamlit=True,
                        height=256,
                        width=256,
                        point_display_radius=5,
                        key=f"base_draw_canvas",
                        drawing_mode = "transform",
                        initial_drawing = st.session_state["ref_mask_inital_drawing"]
                    )
                

        with col2:
            st.subheader("Reference Image")
            ref_file = st.file_uploader("Choose reference image...", type=["jpg", "jpeg", "png"], key="ref_image_uploader")
            if ref_file is not None:
                if "ref_image" not in st.session_state:
                    st.session_state["ref_image"] = Image.open(ref_file).resize((256, 256))
                process_image_drawing(ref_file, "ref_draw", column = col1)

                

        # image, mask = st.session_state.image, st.session_state.mask

        # if st.session_state.image == None:
        #     st.write("Please upload an Image")  

        # if st.session_state.image is not None and st.session_state.mask is not None:
        #     drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon"), key="ref_drawing_mode")
        #     stroke_width = st.slider("Stroke width:", 1, 150, 3, key="ref_stroke_width")
        #     stroke_color = st.color_picker("Stroke color:", "#B5B5B5", key="ref_stroke_color")

        #     image_np = np.array(image)
        #     mask_np = np.array(mask.convert('RGBA'))
            
        #     if mask_np.shape[2] == 3:
        #         mask_np = np.concatenate([mask_np, np.full((*mask_np.shape[:2], 1), 220, dtype=np.uint8)], axis=2)
            
        #     mask_np[..., 3] = np.where(np.any(mask_np[..., :3] > 0, axis=2), 200, 0)
            
        #     combined = Image.alpha_composite(image.convert('RGBA'), Image.fromarray(mask_np))
            
        #     combined_b64 = base64.b64encode(cv2.imencode('.png', np.array(combined))[1]).decode()

        #     initial_drawing = {
        #         "version": "4.4.0",
        #         "objects": [
        #             {
        #                 "type": "image",
        #                 "version": "4.4.0",
        #                 "originX": "left",
        #                 "originY": "top",
        #                 "left": 0,
        #                 "top": 0,
        #                 "width": st.session_state["image"].width,
        #                 "height": st.session_state["image"].height,
        #                 "fill": "rgb(0,0,0)",
        #                 "stroke": None,
        #                 "strokeWidth": 0,
        #                 "strokeDashArray": None,
        #                 "strokeLineCap": "butt",
        #                 "strokeDashOffset": 0,
        #                 "strokeLineJoin": "miter",
        #                 "strokeUniform": False,
        #                 "strokeMiterLimit": 4,
        #                 "scaleX": 1,
        #                 "scaleY": 1,
        #                 "angle": 0,
        #                 "flipX": False,
        #                 "flipY": False,
        #                 "opacity": 1,
        #                 "shadow": None,
        #                 "visible": True,
        #                 "backgroundColor": "",
        #                 "fillRule": "nonzero",
        #                 "paintFirst": "fill",
        #                 "globalCompositeOperation": "source-over",
        #                 "skewX": 0,
        #                 "skewY": 0,
        #                 "cropX": 0,
        #                 "cropY": 0,
        #                 "src": f"data:image/png;base64,{combined_b64}",
        #                 "crossOrigin": None,
        #                 "filters": []
        #             }
        #         ]
        #     }

        #     main_canvas, reference_canvas = st.columns(2)

            
        #     edit_mask = st_canvas(
        #         fill_color="rgba(181, 181, 181, 0.8)",
        #         stroke_width=stroke_width,
        #         stroke_color=f"{stroke_color}80",
        #         background_image=None,  
        #         height=image.height,
        #         width=image.width,
        #         drawing_mode=drawing_mode,
        #     )



            ### `edit_mask` is the final mask 
           

if __name__ == "__main__":
    main()