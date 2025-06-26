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
        console.log(`[ModelServer] Sending to ${this.toURL(urlPath)}`);
        console.log("[ModelServer] Sending request:");
        console.log(jsonObject);
        // TODO: @yuhuan, fix the jsonObject becomes too long(?) problem
        const response = await axios.post(this.toURL(urlPath), jsonObject, {
            headers: {
                "Content-Type": "application/json",
            },
            timeout: 0
        });
        if (response.statusText === "OK") {
            console.log("[ModelServer] Received response:");
            console.log(response.data);
            return response.data;
        } else {
            throw new axios.AxiosError(JSON.stringify(response));
        }
    }
}

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