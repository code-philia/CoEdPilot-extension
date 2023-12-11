import vscode from "vscode";
import { BaseComponent } from "./base-component";
import { registerCommand } from "./base-component";
import os from "os";

export const supportedLanguages = [
    "go",
    "python",
    "typescript",
    "javascript",
    "java"
]

export function isLanguageSupported() {
    return supportedLanguages.includes(editorState.language);
}

class EditLock {
    constructor() {
        this.isLocked = false;
    }

    tryWithLock(callback) {
        if (this.isLocked) return undefined;
        this.isLocked = true;
        try {
            return callback();
        } catch (err) {
            console.error(`Error occured when running in edit lock: \n${err.stack}`);
            // throw err;
        } finally {
            this.isLocked = false;
        }
    }

    async tryWithLockAsync(asyncCallback) {
        if (this.isLocked) return undefined;
        this.isLocked = true;
        try {
            return await asyncCallback();
        } catch (err) {
            console.error(`Error occured when running in edit lock (async): \n${err.stack}`);
            // throw err;
        } finally {
            this.isLocked = false;
        }
    }
}

export const globalEditLock = new EditLock();

class QueryState extends BaseComponent {
    constructor() {
        super();
        // request parameters
        this.commitMessage = "";

        // response parameters
        this.locations = [];
        this.locatedFilePaths = [];
        this._onDidChangeLocations = new vscode.EventEmitter();
        this.onDidChangeLocations = this._onDidChangeLocations.event;

        this.register(
            registerCommand('editPilot.inputMessage', this.inputCommitMessage, this),
            this._onDidChangeLocations
        );
    }

    async updateLocations(locations) {
        this.locations = locations;
        if (this.locations.length) {
            this.locatedFilePaths = [...new Set(locations.map((loc) => loc.targetFilePath))];
        }
        this._onDidChangeLocations.fire(this);
    }

    async clearLocations() {
        this.updateLocations([]);
    }

    async requireCommitMessage() {
        if (this.commitMessage) {
            return this.commitMessage;
        }

        return await this.inputCommitMessage();
    }

    async inputCommitMessage() {
        const userInput = await vscode.window.showInputBox({
            prompt: 'Enter a description of edits you want to make.',
            placeHolder: 'Add a feature...',
            ignoreFocusOut: true,
            value: queryState.commitMessage,
            title: "✍️ Edit Description"
        });
        
        if (userInput) {
            this.commitMessage = userInput;
        }
        return userInput;   // returns undefined if canceled
    }
}

export const queryState = new QueryState();

class EditorState extends BaseComponent {
    constructor() {
        super();
        this.prevCursorAtLine = 0;
        this.currCursorAtLine = 0;
        this.prevSnapshot = undefined;
        this.currSnapshot = undefined;
        this.prevEdits = [];
        this.inDiffEditor = false;
        this.language= "unknown";
    }
}

export const editorState = new EditorState();

export const supportedOSTypes = ['Windows_NT', 'Darwin', 'Linux'];
export const osType = os.type();

if (!supportedOSTypes.includes(osType)) {
    throw RangeError(`Operating system (node detected: ${osType}) is not supported yet.`);
}

export const defaultLineBreaks = {
    'Windows_NT': '\r\n',
    'Darwin': '\r',
    'Linux': '\n'
};
export const defaultLineBreak = defaultLineBreaks[osType] ?? '\n';

