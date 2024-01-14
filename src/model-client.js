import axios from 'axios';
import fs from 'fs';
import vscode from 'vscode';
import { BaseComponent } from './base-component';

/**
 * 
 * @param {string} proxyUrl 
 * @returns 
 */
function parseProxyUrl(proxyUrl) {
    const regex = /^(http[s]?:\/\/)?(?:[^:@/]*:?[^:@/]*@)?([^:/?#]+)(:(\d+))?$/;
    const match = proxyUrl.match(regex);

    if (match) {
        return { protocol: match[1], host: match[2], port: match[4] ? parseInt(match[4], 10) : null };
    } else {
        return null;
    }
}

let server_mock = false;

class ModelServerProcess extends BaseComponent{
    constructor() {
        super();
        this.apiUrl = this.getAPIUrl();
        this.proxy = undefined;

        const vscodeProxyConfig = vscode.workspace.getConfiguration('http').get('proxy');
        if (vscodeProxyConfig?.trim()) {
            const parseResult = parseProxyUrl(vscodeProxyConfig);
            if (parseResult) {
                let port = null;
                if (parseResult.port) {
                    port = parseResult.port;
                } else if (parseResult.protocol) {
                    if (parseResult.protocol.includes("https")) {
                        port = 443;
                    } else {
                        port = 80
                    }
                }
                if (port) {
                    console.log(`Setting up proxy at ${vscodeProxyConfig}`);
                    this.proxy = {
                        host: parseResult.host,
                        port: port 
                    }
                }
            }
        }

        this.register(
            vscode.workspace.onDidChangeConfiguration((e) => {
                if (e.affectsConfiguration("coEdPilot.queryURL")) {
                    this.apiUrl = this.getAPIUrl();
                }
            })
        )
    }

    getAPIUrl() {
        return vscode.workspace.getConfiguration("coEdPilot").get("queryURL");
    }

    toURL(path) {
        return (new URL(path, this.apiUrl)).href ;
    }

    async sendPostRequest(urlPath, jsonObject) {
        console.log(`[ModelServer] Sending to ${this.toURL(urlPath)}`)
        console.log(`[ModelServer] Sending request:`);
        console.log(jsonObject);
        const response = await axios.post(this.toURL(urlPath), jsonObject, {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 200000,
            proxy: this.proxy
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

class MockBackend {
    static async delayedResponse(res_type, json_obj) {
        await new Promise(resolve => {
            setTimeout(resolve, 1000);
        })
        switch (res_type) {
            case "disc":
                return { "data": json_obj.files.map(file_info => file_info[0]).slice(0, 3) };
            case "loc":
                return {
                    "data": [
                        {
                            "targetFilePath": json_obj.files[0][0],
                            "editType": "add",
                            "lineBreak": "\n",
                            "atLines": [0]
                        },
                        {
                            "targetFilePath": json_obj.files[1][0],
                            "editType": "replace",
                            "lineBreak": "\n",
                            "atLines": [2]
                        }
                    ]
                };
            case "gen":
                return {
                    "data": {
                        "editType": json_obj.editType,
                        "replacement":
                            [
                                "1231233312",
                                "4546666666\n4545445",
                                "77788888",
                                "9999999999"
                            ]
                    }
                };
        }
    }
}

export const modelServerProcess = new ModelServerProcess();

async function basicQuery(suffix, json_obj) {
    // fs.writeFileSync('../backend_request.json', JSON.stringify(json_obj), {flag: 'a'});
    return await modelServerProcess.sendPostRequest(suffix, json_obj);
}

async function queryDiscriminator(json_obj) {
    if (server_mock)
        return await MockBackend.delayedResponse('disc', json_obj);
    return await basicQuery("discriminator", json_obj);
}

async function queryLocator(json_obj) {
    if (server_mock)
        return await MockBackend.delayedResponse('loc', json_obj);
    return await basicQuery("range", json_obj);
}

async function queryGenerator(json_obj) {
    if (server_mock)
        return await MockBackend.delayedResponse('gen', json_obj);
    return await basicQuery("content", json_obj);
}

export {
    queryDiscriminator,
    queryLocator,
    queryGenerator
};