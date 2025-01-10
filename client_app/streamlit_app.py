import streamlit as st
from PIL import Image
import cv2
from distinctipy import distinctipy
import json
import io
import requests
import numpy as np

# Function to handle image upload and API request
def upload_image_to_api(image):
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # External API endpoint
    url = "http://localhost:7000/upload"

    # Prepare headers and files for the request
    #headers = {'Authorization': 'Bearer your-api-key'}  # If your API requires authentication
    files = {'file': ('image.png', img_byte_arr, 'image/png')}

    # Send a POST request to the external API with the image file
    response = requests.post(url, files=files)

    return response.json()

# Define the Streamlit app
def main():
    st.set_page_config(page_title="Yolov5-SKU110", page_icon="💡")
    st.title("🔪Yolov5-SKU110K")
    st.write("Object Detection and Group products in Dense Environments using Yolov5 on SKU110K dataset")
    # Allow the user to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Button to post the image to the API
    if st.button("Submit Image to API"):
        data = upload_image_to_api(image)
        st.write(data)  # Display the API response

        #### --- draw BB for the objects----------------------------
        ids = json.loads(data['cluster_ids'])
        #print(ids)
        bb_list=json.loads(data['BB_list'])
        #print(bb_list)
        image = np.array(image)

        colors = distinctipy.get_colors(max(ids)+1)
        colors = [distinctipy.get_rgb256(color) for color in colors]

        for bbox, id in zip(bb_list, ids):
            xmin, ymin, xmax, ymax = bbox
            # Draw the rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[id], 4)
            # Put the ID text
            #print(colors[id])
            cv2.putText(image, str(id), (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[id], 4)
        #--------------------------------------------------------

        # Convert back the image from NumPy array to PIL Image for display
        image_with_bboxes = Image.fromarray(image)

        # Show the image with bounding boxes in Streamlit
        st.image(image_with_bboxes, caption="Detected Objects", use_column_width=True)


if __name__ == "__main__":
    main()