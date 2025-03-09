import os
import json
import warnings
import configparser

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
from discriminator.interface import predict as disc_predict
from locator.interface import predict as loc_predict
from generator.interface import predict as gen_predict
from transformers import logging

logging.set_verbosity_error()
app = FastAPI()

DEBUG = False
SUPPORTED_LANGUAGES = ["go", "python", "java", "typescript", "javascript"]

print(">>> Modules loaded. Server ready.")


async def make_plain_text_response(result):
    if isinstance(result, dict):
        return JSONResponse(content=result, status_code=200)
    return PlainTextResponse(content=result, status_code=200)


async def make_400_response(err_msg):
    return PlainTextResponse(content=err_msg, status_code=400)


async def run_predict(predict_name, predict_func, request: Request, multi_lang=False):
    print(f">>> Running {predict_name}")
    json_str = await request.body()
    input_json = json.loads(json_str.decode('utf-8'))

    language = input_json["language"]
    if language not in SUPPORTED_LANGUAGES:
        return await make_400_response(f"Not supporting language {language} yet.")

    if DEBUG:
        print(f">>> {predict_name} inferencing: \n{json.dumps(input_json, indent=4)}")
    if multi_lang:
        result = await predict_func(input_json)
    else:
        result = await predict_func(input_json, language)

    if DEBUG:
        print(f">>> {predict_name} output: \n{json.dumps(result, indent=4)}")
    print(f">>> {predict_name} sending output")
    return await make_plain_text_response(result)


@app.post('/discriminator')
async def run_discriminator(request: Request):
    return await run_predict('discriminator', disc_predict, request)


@app.post('/range')
async def run_range(request: Request):
    return await run_predict('locator', loc_predict, request, multi_lang=True)


@app.post('/content')
async def run_content(request: Request):
    return await run_predict('generator', gen_predict, request, multi_lang=True)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(f'{os.path.dirname(__file__)}/server.ini')
    uvicorn.run(app, host=config['DEFAULT']['ListenHost'], port=int(config['DEFAULT']['ListenPort']))