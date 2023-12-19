import os
from flask import Flask, config, request, make_response
from waitress import serve
from discriminator.interface import predict as disc_predict
from locator.interface import predict as loc_predict
from generator.interface import predict as gen_predict
import json
import configparser

app = Flask(__name__)

DEBUG = False
SUPPORTED_LANGUAGES = ["go", "python", "java", "typescript", "javascript"]

print(">>> Modules loaded. Server ready.")

def make_plain_text_response(result):
    response = make_response(result, 200)
    response.mimetype = "text/plain"
    response.charset = "utf-8"
    return response

def make_400_response(err_msg):
    response = make_response(err_msg, 400)
    response.mimetype = "text/plain"
    response.charset = "utf-8"
    return response

def run_predict(predict_name, predict_func):
    print(f">>> Running {predict_name}")
    json_str = request.data.decode('utf-8')
    input_json = json.loads(json_str)

    language = input_json["language"]
    if language not in SUPPORTED_LANGUAGES:
        return make_400_response(f"Not supporting language {language} yet.")
    
    if DEBUG:
        print(f">>> {predict_name} inferencing: \n${json.dumps(input_json, indent=4)}")
    result = predict_func(input_json, language)

    if DEBUG:
        print(f">>> {predict_name} output: \n${json.dumps(result, indent=4)}")
    print(f">>> {predict_name} sending output")
    return make_plain_text_response(result)

@app.route('/discriminator', methods=['POST'])
def run_discriminator():
    return run_predict('discriminator', disc_predict)

@app.route('/range', methods=['POST'])
def run_range():
    return run_predict('locator', loc_predict)

@app.route('/content', methods=['POST'])
def run_content():
    return run_predict('generator', gen_predict)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5001, debug=True)
    config = configparser.ConfigParser()
    config.read(f'{os.path.dirname(__file__)}/server.ini')
    serve(app, host=config['DEFAULT']['ListenHost'], port=config['DEFAULT']['ListenPort'])
