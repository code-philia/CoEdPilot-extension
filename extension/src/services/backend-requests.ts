import axios from 'axios';
import vscode from 'vscode';
import { DisposableComponent } from '../utils/base-component';

function parseProxyUrl(proxyUrl: string) {
    const regex = /^(http[s]?:\/\/)?(?:[^:@/]*:?[^:@/]*@)?([^:/?#]+)(:(\d+))?$/;
    const match = proxyUrl.match(regex);

    if (match) {
        const [_0, protocol, host, _3, portStr] = match;
        let port: number | undefined;
        if (portStr) {
            port = parseInt(match[4], 10);
        } else if (protocol) {
            port = protocol.includes("https") ? 443 : 80;
        }

        if (port) {
            return { host, port };
        }
    }
    return null;
}

let server_mock = false;

class ModelServerProcess extends DisposableComponent {
    apiUrl: string;
    proxy: {host: string, port: number} | undefined;

    constructor() {
        super();
        this.apiUrl = this.getApiUrl();
        this.proxy = undefined;

        const vscodeProxyConfigValue = vscode.workspace.getConfiguration('http').get('proxy');
        const vscodeProxyConfig = typeof (vscodeProxyConfigValue) === 'string' ? vscodeProxyConfigValue : undefined;

        if (vscodeProxyConfig?.trim()) {
            const parseResult = parseProxyUrl(vscodeProxyConfig);
            if (parseResult) {
                this.proxy = parseResult;
            }
        }

        this.register(
            vscode.workspace.onDidChangeConfiguration((e) => {
                if (e.affectsConfiguration("coEdPilot.queryURL")) {
                    this.apiUrl = this.getApiUrl();
                }
            })
        );
    }

    getApiUrl() {
        const apiUrlConfigValue = vscode.workspace.getConfiguration("coEdPilot").get("queryURL");
        const apiUrl = typeof(apiUrlConfigValue) === 'string' ? apiUrlConfigValue : "http://localhost:5000";
        return apiUrl;
    }

    toURL(path: string) {
        return (new URL(path, this.apiUrl)).href ;
    }

    async request(path: string, data: object) {
        const response = await axios.post(this.toURL(path), data, {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 200000,
            proxy: this.proxy
        });
        if (response.statusText === 'OK') {
            return response.data;
        } else {
            throw new axios.AxiosError(JSON.stringify(response));
        }
    }
}

// const res_jsons = JSON.parse(fs.readFileSync(path.join(srcDir, '../mock/mock_json_res.json'), { encoding:'utf-8' }));

// FIXME atLines must be continuous now, or it will just be the range between the first and the last line number, which is not a good representation
// FIXME reading files take too long time
const mockDemo1Locator = [
    [{
        "targetFilePath": 'util/chunk/chunk.go',
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [90]
    }],
    [{
        "targetFilePath": 'util/chunk/chunk.go',
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [101]
    },
    {
        "targetFilePath": 'util/chunk/row.go',
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [213]
    }],
    []
];
const mockDemo1Generator = [
    {
        "editType": "replace",
        "replacement": ['	newChk.requiredRows = maxChunkSize']
    },
    {
        "editType": "replace",
        "replacement": ['	return renewWithCapacity(chk, newCap, maxChunkSize)']
    },
    {
        "editType": "replace",
        "replacement": ['	newChk := renewWithCapacity(r.c, 1, 1)']
    },
    undefined
];
const mockDemo2Locator = [
    [{
        "targetFilePath": "modules/sd_samplers_kdiffusion.py",
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [285]
    },
    {
        "targetFilePath": "modules/sd_samplers_kdiffusion.py",
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [287]
    },
    {
        "targetFilePath": "modules/sd_samplers_kdiffusion.py",
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [289]
    },
    {
        "targetFilePath": "modules/sd_samplers_kdiffusion.py",
        "editType": "replace",
        "lineBreak": "\n",
        "atLines": [291]
    }],
    []
];
const mockDemo2Generator = [
    {
        "editType": "replace",
        "replacement": ["        if 'sigma_max' in parameters:"]
    },
    {
        "editType": "replace",
        "replacement": ["        if 'n' in parameters:"]
    },
    {
        "editType": "replace",
        "replacement": ["        if 'sigma_sched' in parameters:"]
    },
    {
        "editType": "replace",
        "replacement": ["        if 'sigmas' in parameters:"]
    },
    undefined
];


function copyObj(obj: any) {
    return JSON.parse(JSON.stringify(obj));
}

class MockBackend {
    static counter = {
        loc: -1,
        gen: -1,
        'rename-loc': -1
    };
    
    static async delayedResponse(res_type: string, json_obj: any) {
        if (res_type === 'loc' || res_type === 'gen' || res_type === 'rename-loc') {
            this.counter[res_type] += 1;
        } 

        await new Promise(resolve => {
            setTimeout(resolve, 1000);
        });
        switch (res_type) {
            case "disc":
                return { "data": json_obj.files.map((file_info: any[]) => file_info[0]).slice(0, 3) };
            case "loc":
                return {
                    // "data": [
                    //     {
                    //         "targetFilePath": json_obj.files[0][0],
                    //         "editType": "add",
                    //         "lineBreak": "\n",
                    //         "atLines": [0]
                    //     },
                    //     {
                    //         "targetFilePath": json_obj.files[1][0],
                    //         "editType": "replace",
                    //         "lineBreak": "\n",
                    //         "atLines": [2]
                    //     }
                    // ]
                    "data": mockDemo2Locator[this.counter['loc'] % mockDemo2Locator.length].map(x => copyObj(x))
                };
            case "gen":
                return {
                    // "data": {
                    //     "editType": json_obj.editType,
                    //     "replacement":
                    //         [
                    //             "1231233312",
                    //             "4546666666\n4545445",
                    //             "77788888",
                    //             "9999999999"
                    //         ]
                    // }
                    "data": copyObj(mockDemo2Generator[this.counter['gen'] % mockDemo2Generator.length])
                };
            case "rename-loc":
                return {
                    "type": "rename",
                    // "data": [
                    //     {
                    //         "file": "a.py",
                    //         "line": 2,
                    //         "beforeText": "def ad(a, b):",
                    //         "afterText": "def add(a, b):"
                    //     },
                    // ]
                    
                    // the first rename that user has already performed
                    "data": this.counter['rename-loc'] === 0 ? [
                        {
                            "file": "modules/sd_samplers_kdiffusion.py",
                            "line": 279,
                            "beforeText": "        extra_params_kwargs = self.initialize(p)",
                            "afterText": "        init_extra_params_kwargs = self.initialize(p)"
                        },
                    ] : []
                };
        }
    }
}

export const modelServerProcess = new ModelServerProcess();

async function basicQuery(suffix: string, json_obj: any) {
    // fs.writeFileSync('../backend_request.json', JSON.stringify(json_obj), {flag: 'a'});
    return await modelServerProcess.request(suffix, json_obj);
}

async function postRequestToDiscriminator(json_obj: any) {
    if (server_mock)
        return await MockBackend.delayedResponse('disc', json_obj);
    return await basicQuery("discriminator", json_obj);
}

async function postRequestToLocator(json_obj: any) {
    if (server_mock)
        return await MockBackend.delayedResponse('rename-loc', json_obj);
        // return await MockBackend.delayedResponse('loc', json_obj);
    return await basicQuery("range", json_obj);
}

async function postRequestToNavEditInvoker(json_obj: any) {
    return await basicQuery("navedit/invoker", json_obj);
}

async function postRequestToNavEditLocator(json_obj: any) {
    return await basicQuery("navedit/locator", json_obj);
}

async function postRequestToGenerator(json_obj: any) {
    if (server_mock)
        return await MockBackend.delayedResponse('gen', json_obj);
    return await basicQuery("content", json_obj);
}

export {
    postRequestToDiscriminator,
    postRequestToLocator,
    postRequestToGenerator,
    postRequestToNavEditInvoker,
    postRequestToNavEditLocator
};