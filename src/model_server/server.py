from flask import Flask, request, make_response
from discriminator.interface import predict as disc_predict
from locator.interface import predict as loc_predict
from generator.interface import predict as gen_predict
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

    # print(f">>> Discriminator inferencing: \n${json.dumps(input_json, indent=4)}")
    result = disc_predict(input_json)

    # print(f">>> Discriminator sending output: \n${json.dumps(result, indent=4)}")
    print(f">>> Discriminator sending output")
    return make_plain_text_response(result)

@app.route('/range', methods=['POST'])
def run_range():
    print(">>> Running locator")
    json_str = request.data.decode('utf-8')
    input_json = json.loads(json_str)
    
    # print(f">>> Locator inferencing: \n${json.dumps(input_json, indent=4)}")
    result = loc_predict(input_json)

    # print(f">>> Locator sending output: \n${json.dumps(result, indent=4)}")
    print(f">>> Locator sending output")
    return make_plain_text_response(result)

@app.route('/content', methods=['POST'])
def run_content():
    print(">>> Running generator")
    json_str = request.data.decode('utf-8')
    input_json = json.loads(json_str)

    # print(f">>> Editor inferencing: \n${json.dumps(input_json, indent=4)}")
    result = gen_predict(input_json)
    
    # print(f">>> Editor sending output: \n${json.dumps(result, indent=4)}")
    print(f">>> Generator sending output")
    return make_plain_text_response(result)

if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)
