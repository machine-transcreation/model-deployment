import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import cv2
import base64
from io import BytesIO

def overlay_mask(base_image: Image, mask_image: Image):
    base_image = base_image.convert("RGBA")
    
    mask_image = mask_image.convert("L")  
    
    mask_image = mask_image.resize(base_image.size)
    
    gray_overlay = Image.new("RGBA", base_image.size, color=(int(30/255 * 255), int(144/255 * 255), int(255/255 * 255), int(0.6 * 255)))
    
    overlay = Image.composite(gray_overlay, base_image, mask_image)
    
    return overlay

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

def SAM2(image: Image, points: np.array, labels: np.array, predictor: SAM2ImagePredictor, rgba: tuple):
    orig_width, orig_height = image.size
    predictor.set_image(image.resize((512, 512)))

    masks, confidence, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
    )

    mask = create_colored_mask_image(mask = masks, R = rgba[0], G = rgba[1], B = rgba[2], A = rgba[3]).resize((orig_width, orig_height))
    
    return mask

@st.cache_resource
def build_predictor():
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    return predictor

predictor = build_predictor()

def mask_to_pillow(image, mask, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(mask_image, alpha=0.6)
    plt.axis('off')
    
    # Convert Matplotlib plot to PIL image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    pil_image = Image.open(buf)
    buf.seek(0)  # Reset the buffer's current position to 0 before returning the image
    plt.close()  # Close the figure after saving to buffer
    
    return pil_image

def main():

    mode = st.sidebar.selectbox("Select Mode", ["Make Mask", "Draw/Edit Mask"])

    st.title("SAM2: Point-Based")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if "image" not in st.session_state:
        st.session_state.image = None
    if "mask" not in st.session_state:
        st.session_state.mask = None
    
    if mode == "Make Mask":

        if uploaded_file is not None:
            
            orig_image = Image.open(uploaded_file)
            
            image = orig_image.resize((512, 512))
            
            st.session_state.image = image

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

                mask = SAM2(image = image, points = np.array(sam_points), labels = np.array(sam_labels), predictor = predictor, rgba = (50, 50, 50, 50))

                st.image(overlay_mask(orig_image, mask), caption = "mask")

                st.session_state.mask = mask
    
    if mode == "Draw/Edit Mask":

        image, mask = st.session_state.image, st.session_state.mask

        if st.session_state.image == None:
            st.write("Please upload and Image")  

        if st.session_state.image is not None and st.session_state.mask is None:
            
            drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon"), key="ref_drawing_mode")
            stroke_width = st.slider("Stroke width:", 1, 150, 3, key="ref_stroke_width")
            stroke_color = st.color_picker("Stroke color:", "#B5B5B5", key="ref_stroke_color")
            
            edit_mask = st_canvas(
                fill_color="rgba(181, 181, 181, 0.8)",
                stroke_width=stroke_width,
                stroke_color=f"{stroke_color}80",
                background_image=image,
                height=image.height,
                width=image.width,
                drawing_mode=drawing_mode,
            )

        if st.session_state.image is not None and st.session_state.mask is not None:

            drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon"), key="ref_drawing_mode")
            stroke_width = st.slider("Stroke width:", 1, 150, 3, key="ref_stroke_width")
            stroke_color = st.color_picker("Stroke color:", "#B5B5B5", key="ref_stroke_color")

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
                        "width": st.session_state["image"].width,
                        "height": st.session_state["image"].height,
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
                        "src": f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', np.array(mask))[1]).decode()}",
                        "crossOrigin": None,
                        "filters": []
                    }
                ]
            }
            
            edit_mask = st_canvas(
                fill_color="rgba(181, 181, 181, 0.8)",
                stroke_width=stroke_width,
                stroke_color=f"{stroke_color}80",
                background_image=image,
                height=image.height,
                width=image.width,
                drawing_mode=drawing_mode,
                initial_drawing=initial_drawing
            )

            ### `edit_mask` is the final mask 
           

if __name__ == "__main__":
    main()
