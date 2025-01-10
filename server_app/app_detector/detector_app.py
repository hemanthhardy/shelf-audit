from PIL import Image
import torch
import numpy as np
from flask import Flask, request, jsonify
import io
# Import your custom YOLOv5 model
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Define the path to your saved YOLOv5 model
model_path = "weights/best.pt"

# Set the device to GPU if available
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path, device=device)

# Detection thresholds
conf_thres = 0.5  # Confidence threshold
iou_thres = 0.4   # IoU threshold

app = Flask(__name__)

@app.route('/detect_products', methods=['POST'])
def detect_products():
    # Check if the file is in the request
    if 'file' not in request.files:
        return jsonify({'status': 'error','error': 'No file part in the request'}), 400

    file = request.files['file']

    # Check if the user selected a file
    if file.filename == '':
        return jsonify({'status': 'error','error': 'No file selected'}), 400

    # try:
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({'status': 'error', 'error': 'Empty image data'}), 400
    image_orig = Image.open(io.BytesIO(img_bytes))
    #shape = image.size  # Original image size
    scale_h = image_orig.size[1]/640
    scale_w = image_orig.size[0]/640
    print("Scaling Factor : ", scale_h,scale_w)

    image = image_orig.resize((640, 640))  # Resize to 640x640 for YOLOv5 input
    img = np.array(image)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and reshape
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize image
    img = img.unsqueeze(0)  # Add batch dimension

    # Run the YOLOv5 model on the image
    pred = model(img)[0]
    # print(pred)
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    # Convert to numpy and process output
    pred = [x.detach().cpu().numpy() for x in pred]
    pred = [x.astype(int) for x in pred]

    # Post-process the output and prepare data
    boxes = []
    confidences = []
    class_ids = []
    for det in pred:
        if det is not None and len(det):
            # Scale the bounding box coordinates to the original image size
            #det[:, :4] = det[:, :4] / 640 * image_orig.size[0]
            #print(det)
            det[:, 0] = det[:, 0] * scale_w  # x_min
            det[:, 1] = det[:, 1] * scale_h  # y_min
            det[:, 2] = det[:, 2] * scale_w  # x_max
            det[:, 3] = det[:, 3] * scale_h  # y_max

            for *xyxy, conf, cls in det:
                boxes.append([int(coord) for coord in xyxy])

    # Return the results in a JSON response
    # result = {
    #     'status': 'success',
    #     'detections': []
    # }

    # for i in range(len(boxes)):
    #     result['detections'].append({
    #         'bbox': boxes[i]
    #     })
    result = {
        'status': 'success',
        'detections': boxes
    }

    return jsonify(result), 200

    # except Exception as e:
    #     return jsonify({'status': 'error','error': 'An error occurred during detection', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(port=7001, debug=True)
