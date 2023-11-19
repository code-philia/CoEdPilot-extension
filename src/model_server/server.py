from flask import Flask, request, make_response
from discriminator.interface import predict as disc_predict
from locator.interface import predict as loc_predict
from generator.interface import predict as gen_predict
import time
import json

app = Flask(__name__)

def make_plain_text_response(result):
    response = make_response(result, 200)
    response.mimetype = "text/plain"
    response.charset = "utf-8"
    return response

@app.route('/discriminator', methods=['POST'])

def run_discriminator():
    print(">>> Running discriminator")
    json_str = request.data.decode('utf-8')
    input_json = json.loads(json_str)

    print(f">>> Discriminator infering: \n${json.dumps(input_json, indent=4)}")
    start_time = time.time()
    result = disc_predict(input_json)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    print(f">>> Discriminator sending output: \n${json.dumps(result, indent=4)}")
    print(f"Inference time: {inference_time:.3f} seconds")   
    return make_plain_text_response(result)

@app.route('/range', methods=['POST'])
def run_range():
    print(">>> Running locator")
    json_str = request.data.decode('utf-8')
    input_json = json.loads(json_str)
    
    print(f">>> Locator infering: \n${json.dumps(input_json, indent=4)}")
    start_time = time.time()
    result = loc_predict(input_json)

    end_time = time.time()
    inference_time = end_time - start_time

    print(f">>> Locator sending output: \n${json.dumps(result, indent=4)}")
    print(f"Inference time: {inference_time:.3f} seconds")
    return make_plain_text_response(result)

@app.route('/content', methods=['POST'])
def run_content():
    print(">>> Running editor")
    json_str = request.data.decode('utf-8')
    input_json = json.loads(json_str)

    print(f">>> Editor infering: \n${json.dumps(input_json, indent=4)}")
    start_time = time.time()
    result = gen_predict(input_json)

    end_time = time.time()
    inference_time = end_time - start_time

    print(f">>> Editor sending output: \n${json.dumps(result, indent=4)}")
    print(f"Inference time: {inference_time:.3f} seconds")
    return make_plain_text_response(result)

if __name__ == '__main__':
    app.run(host='localhost', port=5012, debug=True)
