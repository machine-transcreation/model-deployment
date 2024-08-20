import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def main():
    st.title("Image Plotter: Green + and Red - Points")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
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
            st.write("Green + Points:", st.session_state.green_points)
            st.write("Red - Points:", st.session_state.red_points)

if __name__ == "__main__":
    main()
