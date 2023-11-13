from flask import Flask, request, make_response
import discriminator_model, range_model, content_model

d_tokenizer = None
d_model = None

r_tokenizer = None
r_model = None
r_device = None

c_tokenizer = None
c_model = None
c_device = None

app = Flask(__name__)

def make_plain_text_response(result):
    response = make_response(result, 200)
    response.mimetype = "text/plain"
    response.charset = "utf-8"
    return response

@app.route('/discriminator', methods=['POST'])
def run_discriminator():
    print(">>> Running discriminator")
    global d_tokenizer, d_model
    input_json = request.data.decode('utf-8')
    
    if d_tokenizer == None or d_model == None:
        print(">>> Discriminator not initialized. Initializing")
        d_tokenizer = discriminator_model.load_tokenizer()
        d_model = discriminator_model.load_model()

    print(">>> Discriminator inferencing")
    result = discriminator_model.predict(input_json, d_tokenizer, d_model)

    print(">>> Discriminator sending output")
    return make_plain_text_response(result)

@app.route('/range', methods=['POST'])
def run_range():
    print(">>> Running locator")
    global r_tokenizer, r_model, r_device
    input_json = request.data.decode('utf-8')
    
    if r_tokenizer == None or r_model == None or r_device == None:
        print(">>> Locator not initialized. Initializing")
        r_tokenizer, r_model, r_device = range_model.load_model()
    
    print(">>> Locator inferencing")
    result = range_model.predict(input_json, r_tokenizer, r_model, r_device)

    print(">>> Locator sending output")
    return make_plain_text_response(result)

@app.route('/content', methods=['POST'])
def run_content():
    print(">>> Running editor")
    global c_tokenizer, c_model, c_device
    input_json = request.data.decode('utf-8')
    
    if c_tokenizer == None or c_model == None or c_device == None:
        print(">>> Editor not initialized. Initializing")
        c_tokenizer, c_model, c_device = content_model.load_model()

    print(">>> Editor inferencing")
    result = content_model.predict(input_json, c_tokenizer, c_model, c_device)
    
    print(f">>> Editor sending output: {result}")
    return make_plain_text_response(result)

if __name__ == '__main__':
    app.run(debug=True)

    # port = app.config['SERVER_PORT']
    # print(f'PORT:{port}')