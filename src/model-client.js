import axios from "axios";
import fs from "fs";
import vscode from "vscode";
import { BaseComponent } from "./base-component";

// const regPortInfo = /PORT:[0-9]+/;

class ModelServerProcess extends BaseComponent{
    constructor() {
        super();
        this.apiUrl = this.getAPIUrl();

        this.register(
            vscode.workspace.onDidChangeConfiguration((e) => {
                if (e.affectsConfiguration("coEdPilot.queryURL")) {
                    this.apiUrl = this.getAPIUrl();
                }
            })
        );
    }

    getAPIUrl() {
        return vscode.workspace.getConfiguration("coEdPilot").get("queryURL");
    }

    toURL(path) {
        return (new URL(path, this.apiUrl)).href ;
    }

    async sendPostRequest(urlPath, jsonObject) {
        console.log("[ModelServer] Sending to ${this.toURL(urlPath)}");
        console.log("[ModelServer] Sending request:");
        console.log(jsonObject);
        const response = await axios.post(this.toURL(urlPath), jsonObject, {
            headers: {
                "Content-Type": "application/json",
            },
            timeout: 100000
        });
        if (response.statusText === "OK") {
            console.log("[ModelServer] Received response:");
            console.log(response.data);
            // DEBUGGING
            // fs.writeFileSync(
            //     path.join(srcDir, "../mock/backend_response.json"),
            //     JSON.stringify(response.data), { flag: "a" }
            // );
            return response.data;
        } else {
            throw new axios.AxiosError(JSON.stringify(response));
        }
    }
}

// const res_jsons = JSON.parse(fs.readFileSync(path.join(srcDir, "../mock/mock_json_res.json"), { encoding:"utf-8" }));

// function copyObj(obj) {
//     return JSON.parse(JSON.stringify(obj));
// }
// class MockBackend {
//     static resultCache = undefined;

//     static async delayedResponse(res_type, req) {
//         await new Promise(resolve => {
//             setTimeout(resolve, 1000);
//         });
//         if (this.resultCache === undefined) {
//             this.resultCache = await this.getLocatorMockResponse()
//         }
//         if (res_type === 'disc') {
//             return copyObj(this.resultCache.discriminatorResponse);
//         } else if (res_type === 'loc') {
//             return copyObj(this.resultCache.locatorResponse);
//         } else {
//             return undefined;
//         }
//     }

//     static async getLocatorMockResponse() {
//         const workspaceFolders = vscode.workspace.workspaceFolders;
//         if (!workspaceFolders) {
//             return;
//         }
    
//         const rootPath = workspaceFolders[0].uri.fsPath;
//         const files = await vscode.workspace.findFiles('**/*');
    
//         let targetFilePath = null;
//         let fileContent = null;
    
//         for (const file of files) {
//             const filePath = file.fsPath;
//             try {
//                 const content = fs.readFileSync(filePath, 'utf-8');
//                 targetFilePath = filePath;
//                 fileContent = content.split('\n');
//                 break;
//             } catch (err) {
//                 // Not a UTF-8 decodable file, continue to the next file
//                 continue;
//             }
//         }
    
//         if (!targetFilePath || !fileContent) {
//             return;
//         }

//         const relativeTargetFilePath = vscode.workspace.asRelativePath(targetFilePath);

//         const discriminatorResponse = {
//             data: [relativeTargetFilePath, fileContent]
//         };
    
//         const selectedLines = [];
//         const lineCount = fileContent.length;
//         const numLinesToSelect = Math.floor(Math.random() * 3) + 3; // Randomly select 3-5 lines
    
//         for (let i = 0; i < numLinesToSelect; i++) {
//             const lineIndex = Math.floor(Math.random() * lineCount);
//             // prevent duplicated lines
//             if (selectedLines.some(line => line.atLines.includes(lineIndex))) {
//                 continue;
//             }

//             const editType = Math.random() > 0.5 ? 'add' : 'replace';
//             selectedLines.push({
//                 targetFilePath: relativeTargetFilePath,
//                 toBeReplaced: fileContent[lineIndex],
//                 editType,
//                 lineBreak: '\n',
//                 atLines: [lineIndex],
//                 confidence: Math.random()
//             });
//         }
    
//         const locatorResponse = {
//             data: selectedLines
//         };
    
//         console.log(locatorResponse);
//         return {
//             discriminatorResponse,
//             locatorResponse
//         };
//     }
// }

export const modelServerProcess = new ModelServerProcess();

async function basicQuery(suffix, json_obj) {
    // fs.writeFileSync("../backend_request.json", JSON.stringify(json_obj), {flag: "a"});
    return await modelServerProcess.sendPostRequest(suffix, json_obj);
}

async function queryDiscriminator(json_obj) {
    return await basicQuery("discriminator", json_obj);
    // return await MockBackend.delayedResponse("disc");
}

async function queryLocator(json_obj) {
    return await basicQuery("range", json_obj);
    // return await MockBackend.delayedResponse("loc");
}

async function queryGenerator(json_obj) {
    return await basicQuery("content", json_obj);
    // return await MockBackend.delayedResponse("gen");
}

export {
    queryDiscriminator,
    queryLocator,
    queryGenerator
};