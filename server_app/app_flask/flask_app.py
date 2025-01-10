from flask import Flask, request, jsonify
import requests
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the file is in the request
    if 'file' not in request.files:
        return jsonify({'status': 'error','error': 'No file part in the request'}), 400
 
    file = request.files['file']
    # Read the file into a byte stream
    img_bytes = file.read()
    
    # Check if the user selected a file
    if file.filename == '':
        return jsonify({'status': 'error','error': 'No file selected'}), 400
    try:
        # Secure the filename
        secure_name = secure_filename(file.filename)
        print("Received file name:", secure_name)

        # Detect API URL
        detect_api_url = "http://detector_app_c1:7001/detect_products"  # Replace with your detection API

        # Forward the file to the detection API
        response = requests.post(
            detect_api_url,
            files={'file': (secure_name, img_bytes, file.content_type)}  #files={'file': (secure_name, file.stream, file.content_type)}
        )

        # Handle response from the detection API
        if response.status_code == 200:
            print("Detection completed!")
            #return jsonify({'status': 'success', 'api_response': response.json()}), 200  # Stops till the detection process
            detections = response.json()["detections"]

            grouper_api_url = "http://grouper_app_c1:7002/group_products"
            # Forward the file to the detection API
            #print(detections)
            bb_list = json.dumps(detections)
            #headers = {'Content-Type': 'application/json'}

            #data={'detections':detections}
            response = requests.post(
                grouper_api_url,
                files={'file': (secure_name, img_bytes, file.content_type)},
                data={'detections':bb_list},
            )

            # Handle response from the detection API
            if response.status_code == 200:
                print("Grouping completed!")
                return jsonify({'status': 'success', 'BB_list':bb_list,'cluster_ids': response.json()["product_ids"]}), 200
            else:
                print("Grouping process failed with status:", response.status_code)
                return jsonify({'status': 'error','error': 'Failed Grouping the products', 'details': response.text}), response.status_code

        else:
            print("Detection failed with status:", response.status_code)
            return jsonify({'status': 'error','error': 'Failed to process the image', 'details': response.text}), response.status_code

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'status': 'error','error': 'An error occurred while sending the request', 'details': str(e)}), 500
    
    #return jsonify({'status': 'success', 'flask_api_response': detections}), 200

if __name__ == '__main__':
    app.run("0.0.0.0",port=7000,debug=True)