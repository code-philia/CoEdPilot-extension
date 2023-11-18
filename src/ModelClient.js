const path = require('path');
const axios = require('axios').default;
const { spawn } = require('child_process');
const { AxiosError } = require('axios');
const fs = require('fs');

const srcDir = __dirname;
const PyInterpreter = "C:/Program Files/Python310/python.exe";
const pyServerPath = path.join(srcDir, 'model_server', "server.py");

// const regPortInfo = /PORT:[0-9]+/;

class ModelServerProcess{
    constructor() {
        this.ip = 'localhost';
        this.port = '5004';
        this.setup();
    }

    setup() {
        console.log(`[ModelServer] Initializing process from "${pyServerPath}"`)
        this.process = spawn(PyInterpreter, [pyServerPath]);
        this.process.stdout.setEncoding('utf-8');
        console.log("[ModelServer] Process initialized.")

        this.process.stdout.on('data', (data) => {
            const output = data.toString();
            console.log(`[ModelServer] ${output}`);
            // if (regPortInfo.test(output)) {
            //     this.port = output.slice(5);
            //     console.log(`[ModelServer] Port number set to ${this.port}`) // Not Implemented Yet
            // }
        });

        this.process.stderr.on('data', (data) => {
            console.log(data.toString());
        })
    }

    toURL(path) {
        return `http://${this.ip}:${this.port}/${path}`;
    }

    async sendPostRequest(urlPath, jsonObject) {
        console.log(`[ModelServer] Sending to ${this.toURL(urlPath)}`)
        console.log(`[ModelServer] Sending request:`);
        console.log(jsonObject);
        const response = await axios.post(this.toURL(urlPath), jsonObject, {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 100000
        });
        if (response.statusText === 'OK') {
            console.log(`[ModelServer] Received response:`);
            console.log(response.data);
            // DEBUGGING
            // fs.writeFileSync(
            //     path.join(srcDir, '../mock/backend_response.json'),
            //     JSON.stringify(response.data), { flag: 'a' }
            // );
            return response.data;
        } else {
            throw new AxiosError(JSON.stringify(response));
        }
    }
}

const res_jsons = JSON.parse(fs.readFileSync(path.join(srcDir, '../mock/mock_json_res.json'), { encoding:'utf-8' }));

class MockBackend {
    static disc_res_json = res_jsons['disc'];
    static loc_res_json = res_jsons['loc'];
    static gen_res_json = res_jsons['gen'];

    static async delayed_res(json_obj) {
        await new Promise(resolve => {
            setTimeout(resolve, 1000);
        })
        return JSON.parse(JSON.stringify(json_obj)); // shallow clone necessary here
    }

    static async disc_res() {
        return await this.delayed_res(this.disc_res_json);
    }

    static async loc_res() {
        return await this.delayed_res(this.loc_res_json);
    }

    static async gen_res() {
        return await this.delayed_res(this.gen_res_json);
    }
}

const modelServerProcess = new ModelServerProcess();

async function basic_query(suffix, json_obj) {
    fs.writeFileSync('../backend_request.json', JSON.stringify(json_obj), {flag: 'a'});
    return await modelServerProcess.sendPostRequest(suffix, json_obj);
}

async function query_discriminator(json_obj) {
    return await basic_query("discriminator", json_obj);
    // return await MockBackend.disc_res();
}

async function query_locator(json_obj) {
    return await basic_query("range", json_obj);
    // return await MockBackend.loc_res();
}

async function query_generator(json_obj) {
    return await basic_query("content", json_obj);
    // return await MockBackend.gen_res();
}

module.exports = {
    query_discriminator,
    query_locator,
    query_generator
}