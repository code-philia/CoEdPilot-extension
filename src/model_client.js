const path = require('path');
const axios = require('axios').default;
const { spawn } = require('child_process');
const { AxiosError } = require('axios');

// 考虑解耦到配置文件，即 VSCode 的 configuration
const extensionDirectory = __dirname;
const PyInterpreter = "C:/Program Files/Python310/python.exe";
const pyServerPath = path.join(extensionDirectory, 'model_server', "server.py");

const regPortInfo = /PORT:[0-9]+/;

class ModelServerProcess{
    constructor() {
        this.ip = 'localhost';
        this.port = '5000';
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
            if (regPortInfo.test(output)) {
                this.port = output.slice(5);
                console.log(`[ModelServer] Port number set to ${this.port}`) // 此处在后端还未实现
            }
        });

        this.process.stderr.on('data', (data) => {
            console.log(data.toString());
        })
    }

    toURL(path) {
        return `http://${this.ip}:${this.port}/${path}`;
    }

    async sendPostRequest(path, json_str) {
        console.log(`[ModelServer] Sending to ${this.toURL(path)}`)
        // console.log(`[ModelServer] Sending request:`);
        // console.log(json_str);
        const response = await axios.post(this.toURL(path), json_str, {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 100000
        });
        if (response.statusText === 'OK') {
            console.log(`[ModelServer] Received response:`);
            console.log(response);
            return response.data;
        } else {
            throw new AxiosError(JSON.stringify(response));
        }
    }
}

// 管理模型子进程
const modelServerProcess = new ModelServerProcess();

async function basic_query(suffix, json_str) {
    return await modelServerProcess.sendPostRequest(suffix, json_str);
}

async function query_discriminator(json_str) {
    return await basic_query("discriminator", json_str);
}

async function query_locator(json_str) {
    return await basic_query("range", json_str);
}

async function query_editor(json_str) {
    return await basic_query("content", json_str);
}

module.exports = {
    query_discriminator,
    query_locator,
    query_editor
}