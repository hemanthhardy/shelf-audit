from flask import Flask, request, jsonify
from FE_clustring import cluster_detections
from PIL import Image
import io
import json

app = Flask(__name__)

@app.route('/group_products', methods=['POST'])
def group_products():
    # Check if the file is in the request

    bb_list = request.form.get('detections')
    bb_list = json.loads(bb_list)
    if bb_list is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    print(bb_list)

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

    image = Image.open(io.BytesIO(img_bytes))

    # Get clustered unique ids
    print("Calling cluster_detections")

    product_ids = cluster_detections(image,bb_list)

    result = {
            'status': 'success',
            #'detections': bb_list,
            'product_ids': json.dumps(list(product_ids))
            }
    return jsonify(result), 200

    # except Exception as e:
    #     return jsonify({'status': 'error','error': 'An error occurred during Grouping', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(port=7002, debug=True)