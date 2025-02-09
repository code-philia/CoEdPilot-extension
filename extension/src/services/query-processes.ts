import * as vscode from 'vscode';
import { createRenameRefactor, globalQueryContext } from '../global-result-context';
import { toRelPath, getActiveFilePath, toAbsPath, getLineInfoInDocument } from '../utils/file-utils';
import { postRequestToDiscriminator, postRequestToLocator, postRequestToGenerator, modelServerProcess, postRequestToNavEditInvoker, postRequestToNavEditLocator } from './backend-requests';
import { statusBarItem } from '../ui/progress-indicator';
import { BackendApiEditLocation, Edit, EditType, SimpleEdit } from '../utils/base-types';
import { BackendApiEditGenerationJsonType } from './json-validator';

// cancellable, request process
type RequestStatus = {
    status: 'not-started' | 'started' | 'cancelled' | 'succeeded' | 'failed';
    result?: any;
}
type RequestOptions = {
    cancelPath?: string;
    cancelData?: object;
}

type indicatedResult<T> = {
    desc: string,
    success: boolean,
    data: T
};
function isIndicatedResult<T>(value: any, typeCheck: (value: any) => value is T): value is indicatedResult<T> {
    return value
        && typeof value.desc === 'string'
        && typeof value.success === 'boolean'
        && typeCheck(value.data);
}
function indicatedPromise<T>(desc: string, promise: Promise<T>): Promise<indicatedResult<T>> {
    return promise.then(
        (result) => ({ desc: desc, success: true, data: result }),
        (error) => ({ desc: desc, success: false, data: error })
    );
}

type Condition<T> = {
    promise: Promise<T>;
    allowedStatus?: boolean;
};
function expectCondition<T>(promise: Promise<any> | undefined, allowedStatus?: boolean): Condition<T> | undefined {
    if (promise === undefined) {
        return undefined;
    }

    const condition: Condition<T> = { promise };
    if (allowedStatus !== undefined) {
        condition.allowedStatus = allowedStatus;
    }
    return condition;
}
function wrapCondition<T>(condition: Condition<T>) {
    const acceptResolve = condition.allowedStatus !== false;
    const accpetReject = condition.allowedStatus !== true;

    return new Promise((res: (value: T) => void, rej: (value: T | Error) => void) => {
        condition.promise.then(
            (result) => acceptResolve ? res(result) : rej(result),
            (error) => accpetReject ? res(error) : rej(error)
        );
    });
}
class ConditionFilter<T> {
    private conditionStack: Condition<T>[][] = [];
    private defaultResult: T | undefined = undefined;

    wait(condition: Condition<T> | undefined) {
        const newLayer: Condition<T>[] = [];
        if (condition !== undefined) {
            newLayer.push(condition);
        }
        this.conditionStack.push(newLayer);

        return this;
    }

    or(condition: Condition<T> | undefined) {
        if (this.conditionStack.length === 0) {
            throw new RangeError('No previous condition to perform `or`');
        }

        const lastLayer = this.conditionStack.at(-1);
        if (lastLayer && condition !== undefined) {
            lastLayer.push(condition);
        }

        return this;
    }

    else(defaultResult: T) {
        this.defaultResult = defaultResult;
        return this;
    }

    // resolve a result, not throwing error but just pass it when a promise is required to fail
    async result(): Promise<T | Error | undefined> {
        for (const layer of this.conditionStack) {
            try {
                const expectedResultOrError = await Promise.any(layer.map(condition => wrapCondition(condition)));
                return expectedResultOrError;
            } catch { }
        }
        return this.defaultResult;
    }
}

class BackendRequest {
    private path: string;
    private requestData: object;
    private options: RequestOptions = {};

    private processes: { [key: string]: Promise<undefined | object>; } = {};
    private resolvedStatus: RequestStatus;

    constructor (path: string, requestData: object, options?: RequestOptions) {
        this.path = path;
        this.requestData = requestData;
        
        if (options) {
            this.updateOptions(options);
        }
        
        // set default values
        this.resolvedStatus = { status: 'not-started' };
    }

    private cancelPath() {
        return (`/cancel${this.path}`);
    }

    private defaultFailedResult(error?: any): RequestStatus {
        return { status: 'failed', result: error };
    }

    private startProcess(key: string, promise: Promise<undefined | object>) {
        this.processes[key] = promise;
    }
    
    private expectProcess(desc: string, allowedStatus?: boolean): Condition<RequestStatus> | undefined {
        return expectCondition(
            indicatedPromise(desc, this.processes[desc]),
            allowedStatus
        );
    }

    private async getRequestOrCancelled(): Promise<RequestStatus> {
        // if it is cancelled or failed, the result or error will immediately be returned
        try {
            const result = await new ConditionFilter<RequestStatus>()
                .wait(this.expectProcess('request', true)).or(this.expectProcess('cancel', true))
                .wait(this.expectProcess('cancel', false))
                .wait(this.expectProcess('request', false))
                .else({ status: 'failed' })
                .result();
            if (result === undefined) {
                return this.defaultFailedResult();
            } else if (result instanceof Error) {
                return this.defaultFailedResult(result);
            } else {
                return result;
            }
        } catch (error) {
            return this.defaultFailedResult(error);
        }
    }

    private async getRequestOrCancelledResult(): Promise<RequestStatus> {
        if (this.status === 'started') {
            this.resolvedStatus = await this.getRequestOrCancelled();
        }
        return this.resolvedStatus;
    }

    private tryStartRequest() {
        if (this.status === 'not-started') {
            const promise = modelServerProcess.request(this.path, this.requestData);
            this.startProcess('request', promise);
            this.resolvedStatus = { status: 'started' };
            return true;
        }
        return false;
    }

    private tryCancelRequest() {
        if (this.status === 'started' && !('cancel' in this.processes)) {
            const promise = modelServerProcess.request(this.cancelPath(), this.options.cancelData || {});
            this.startProcess('cancel', promise);
            return true;
        }
        return false;
    }

    get status() {
        return this.resolvedStatus.status;
    }

    updateOptions(options: RequestOptions) {
        this.options = { ...this.options, ...options };
    }

    async getResponse(): Promise<RequestStatus> {
        this.tryStartRequest();
        return await this.getRequestOrCancelledResult();
    }
    
    async cancel() {
        this.tryCancelRequest();
        return await this.getRequestOrCancelled();
    }
}

async function requestAndUpdateLocation(
    rootPath: string, 
    files: [string, string][],
    prevEdits: SimpleEdit[],
    commitMessage: string, 
    language: string
) {
    /* 
        Discriminator:
        input:
        {
            "rootPath":         str, rootPath,
            "files":            list, [[filePath, fileContent], ...],
            "targetFilePath":   str, filePath
        }
        output:
        {
            "data": list, [filePath, ...]
        }
	
        Locator:
        input:
        {
            "files":            list, [[filePath, fileContent], ...],
            "targetFilePath":   str, filePath,
            "commitMessage":    str, edit description,
            "prevEdits":        list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}
        }
        output:
        {
            "data": 
            [ 
                { 
                    "targetFilePath":   str, filePath,
                    "toBeReplaced":     str, the content to be replaced, 
                    "editType":         str, the type of edit, add or remove,
                    "lineBreak":        str, '\n', '\r' or '\r\n',
                    "atLines":           number, line number (beginning from 1) of the location
                }, ...
            ]
        }
     */
    const activeFileAbsPath = getActiveFilePath();
    if (!activeFileAbsPath) {
        return;
    }
    
    const activeFilePath = toRelPath(
        rootPath,
        activeFileAbsPath
    );

    // convert all paths to relative paths
    for (const file_info of files) {
        file_info[0] = toRelPath(
            rootPath,
            file_info[0]
        );
    }

    // Send to the discriminator model for analysis
    const disc_input = {
        rootPath: rootPath,
        files: files,
        targetFilePath: activeFilePath,
        commitMessage: commitMessage,
        prevEdits: prevEdits,
        language: language
    };
    const discriminatorOutput = await postRequestToDiscriminator(disc_input);

    // Send the selected files to the locator model for location prediction
    const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename));

    const loc_input = {
        files: filteredFiles,
        targetFilePath: activeFilePath,
        commitMessage: commitMessage,
        prevEdits: prevEdits,
        language: language
    };
    statusBarItem.setStatusQuerying("locator");
    
    const locatorOutput = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Analyzing...' }, async () => {
        return await postRequestToLocator(loc_input);
    });
    // const locatorOutput = await postRequestToLocator(loc_input);

    // TODO add strict format check for each "valid type" of locatorOutput
    if (locatorOutput?.type === 'rename' && locatorOutput?.data?.length) {
        const refactorInfo = locatorOutput.data[0];
        const renameRefactor = await createRenameRefactor(
            refactorInfo.file,
            refactorInfo.line,
            refactorInfo.beforeText,
            refactorInfo.afterText
        );
        if (renameRefactor) {
            globalQueryContext.updateRefactor(renameRefactor);
        }
        return renameRefactor;
    } else {
        // convert all paths back to absolute paths
        let rawLocations = locatorOutput.data;
        for (const loc of rawLocations) {
            loc.targetFilePath = toAbsPath(rootPath, loc.targetFilePath);
            loc.lineInfo = await getLineInfoInDocument(loc.targetFilePath, loc.atLines[0]);
        }
        // TODO add failure processing if there are no locations in response
        globalQueryContext.updateLocations(rawLocations);
        return rawLocations;
    }
}

async function requestAndUpdateLocationByNavEdit(
    rootPath: string, 
    files: [string, string][],
    prevEdits: Edit[],
    commitMessage: string, 
    language: string
) {
    /* 
        Locator:
        input:
        {
            "files":            list, [[filePath, fileContent], ...],
            "targetFilePath":   str, filePath,
            "commitMessage":    str, edit description,
            "prevEdits":        list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}
        }
        output:
        {
            "data": 
            [ 
                { 
                    "targetFilePath":   str, filePath,
                    "toBeReplaced":     str, the content to be replaced, 
                    "editType":         str, the type of edit, add or remove,
                    "lineBreak":        str, '\n', '\r' or '\r\n',
                    "atLines":           number, line number (beginning from 1) of the location
                }, ...
            ]
        }
     */
    const activeFileAbsPath = getActiveFilePath();
    if (!activeFileAbsPath) {
        return;
    }
    
    const activeFilePath = toRelPath(
        rootPath,
        activeFileAbsPath
    );

    // Send to the discriminator model for analysis
    const editedFilePaths = new Set(prevEdits.map((edit) => edit.path));
    const invokerInput = {
        files: files.filter(([path, _]) => editedFilePaths.has(path)),
        prevEdits: prevEdits,
        language: language
    };
    statusBarItem.setStatusQuerying("locator");
    
    const invokerOutput = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Analyzing...' }, async () => {
        return await postRequestToNavEditInvoker(invokerInput);
    });

    /*
        {
            type: "rename" | "def&ref" | "clone" | "normal",
            info: {
                originalFile: "/relative/path/to/file",
                originalLine: 123,
                identifier?: "the-identifier-of-rename-def-ref-clone"
            }
        }
    */

    // TODO add strict format check for each "valid type" of locatorOutput
    if (invokerOutput?.type === 'rename' && invokerOutput?.data?.length) {
        const refactorInfo = invokerOutput.info;
        const renameRefactor = await createRenameRefactor(
            refactorInfo.file,
            refactorInfo.line,
            refactorInfo.beforeText,
            refactorInfo.afterText
        );
        if (renameRefactor) {
            globalQueryContext.updateRefactor(renameRefactor);
        }
        return renameRefactor;
    } else if (invokerOutput?.type === 'normal') {
        const locatorInput = {
            files: files,
            commitMsg: commitMessage,
            prevEdits: prevEdits,
            language: language
        };

        const locatorOutput = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Analyzing...' }, async () => {
            return await postRequestToNavEditLocator(locatorInput);
        });

        const rawLocations = locatorOutput;
        const convertedLocations: BackendApiEditLocation[] = [];
        for (const path in rawLocations) {
            const info = rawLocations[path];
            const labels: [string, number, number][] = [];
            info.inline_preds.forEach((label: string, index: number) => {
                if (label === '<keep>')
                    return;
                if (labels.length === 0 || labels.at(-1)?.[0] !== label) {
                    labels.push([label, index, 0]);
                }
                const lastLabel = labels.at(-1);
                if (lastLabel) {
                    lastLabel[2] += 1;
                }
            });

            for (const [label, start, lines] of labels) {
                const _label = label.slice(1, -1);

                convertedLocations.push({
                    targetFilePath: path,
                    // FIXME strip <delete> to delete should not use this way
                    editType: _label === 'delete' ? 'remove' :
                        _label === 'add' ? 'add' : 'replace',
                    lineBreak: '\n',
                    atLines: Array(lines).fill(0).map((_, i) => start + i),
                    lineInfo: await getLineInfoInDocument(path, start)
                });
            }
        }
        globalQueryContext.updateLocations(convertedLocations);
        return rawLocations;   
    }
}


async function requestAndUpdateEdit(
    fileContent: string,
    editType: EditType,
    atLines: number[],
    prevEdits: SimpleEdit[],
    commitMessage: string,
    language: string
) {
    /* 	
        Generator:
        input:
        { 
            "targetFileContent":    string
            "commitMessage":        string, edit description,
            "editType":             string, edit type,
            "prevEdits":            list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
            "atLines":               list, of edit line indices
            "language":             string, the language used (to select a language-specific model)
        }
        output:
        {
            "data": 
            { 
                "editType":         string, 'remove', 'add'
                "replacement":      list of strings, replacement content   
            }
        } 
    */       
    const input = {
        targetFileContent: fileContent,
        commitMessage: commitMessage,
        editType: editType,
        prevEdits: prevEdits,
        atLines: atLines,
        language: language
    };

    if (editType === "add") { // the model was designed to generate addition at next line, so move one line backward
        atLines = atLines.map((l) => l > 0 ? l - 1 : 0);
    }

    const output = await postRequestToGenerator(input);

    const result = output.data;
    // new BackendApiEditGenerationJsonType().assert(result);

    return result;
}

function isInt(value: any): asserts value is number {
    if (!(typeof value === 'number' && Number.isInteger(value))) {
        throw new Error('Value is not an integer');
    }
}
isInt(1);


export {
    requestAndUpdateLocation,
    requestAndUpdateEdit,
    requestAndUpdateLocationByNavEdit
};
