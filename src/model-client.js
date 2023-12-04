import path from 'path';
import axios from 'axios';
import { spawn } from 'child_process';
import fs from 'fs';

const srcDir = __dirname;
const PyInterpreter = "C:/Program Files/Python310/python.exe";
const pyServerPath = path.join(srcDir, 'model_server', "server.py");

// const regPortInfo = /PORT:[0-9]+/;

class ModelServerProcess{
    constructor() {
        this.ip = 'localhost';
        this.port = '5001';
        // this.setup();
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
            throw new axios.AxiosError(JSON.stringify(response));
        }
    }
}

// const res_jsons = JSON.parse(fs.readFileSync(path.join(srcDir, '../mock/mock_json_res.json'), { encoding:'utf-8' }));

// class MockBackend {
//     static async delayedResponse(res_type) {
//         await new Promise(resolve => {
//             setTimeout(resolve, 1000);
//         })
//         return JSON.parse(JSON.stringify(res_jsons[res_type])); // deep clone necessary here
//     }
// }

const modelServerProcess = new ModelServerProcess();

async function basicQuery(suffix, json_obj) {
    // fs.writeFileSync('../backend_request.json', JSON.stringify(json_obj), {flag: 'a'});
    return await modelServerProcess.sendPostRequest(suffix, json_obj);
}

async function queryDiscriminator(json_obj) {
    return await basicQuery("discriminator", json_obj);
    // return await MockBackend.delayedResponse('disc');
}

async function queryLocator(json_obj) {
    return await basicQuery("range", json_obj);
    // return await MockBackend.delayedResponse('loc');
}

async function queryGenerator(json_obj) {
    return await basicQuery("content", json_obj);
    // return await MockBackend.delayedResponse('gen');
}

export {
    queryDiscriminator,
    queryLocator,
    queryGenerator
};