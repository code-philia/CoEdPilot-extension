import vscode from 'vscode';
import { toRelPath, getActiveFilePath, toAbsPath, getLineInfoInDocument, getRootPath } from './file';
import { queryDiscriminator, queryLocator, queryGenerator } from './model-client';
import { BaseComponent } from './base-component';
import { registerCommand } from './extension-register';

class QueryState extends BaseComponent {
    constructor() {
        super();
        // request parameters
        this.commitMessage = "";

        // response parameters
        this.locations = [];
        this.locatedFilePaths = [];
        this._onDidQuery = new vscode.EventEmitter();
        this.onDidQuery = this._onDidQuery.event;
        
        this.register(
            registerCommand('editPilot.inputMessage', this.inputCommitMessage, this),
            this._onDidQuery
        );
    }

    async updateLocations(locations) {
        this.locations = locations;
        if (this.locations.length) {
            this.locatedFilePaths = [...new Set(locations.map((loc) => loc.targetFilePath))];
        }
        for (const loc of this.locations) {
            loc.lineInfo = await getLineInfoInDocument(loc.targetFilePath, loc.atLines[0]);
        }
        this._onDidQuery.fire(this);
    }

    async clearLocations() {
        this.updateLocations([]);
    }
   
    async requireCommitMessage(msg) {
        if (!msg) {
            msg = await this.inputCommitMessage();
        }
        
        this.commitMessage = msg;
        return msg;
    }

    async inputCommitMessage() {
        console.log('==> Edit description input box is displayed')
        const userInput = await vscode.window.showInputBox({
            prompt: 'Enter commit message of description of edits you would make.',
            placeHolder: 'add a feature',
        }) ?? "";
        console.log('==> Edit description:', userInput);
        this.commitMessage = userInput;

        return userInput;
    }
}

const queryState = new QueryState();

// ------------ Extension States -------------
async function queryLocationFromModel(rootPath, files, prevEdits, commitMessage) {
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
            "data": list, [[filePath, fileContent], ...]
        }
	
        Locator:
        input:
        {
            "files":            list, [[filePath, fileContent], ...],
            "targetFilePath":   str, filePath,
            "commitMessage":    str, commit message,
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
    const activeFilePath = toRelPath(
        rootPath,
        getActiveFilePath()
    );

    // convert all paths to relative paths
    for (const file_info of files) {
        file_info[0] = toRelPath(
            rootPath,
            file_info[0]
        );
    }

    commitMessage = await queryState.requireCommitMessage(commitMessage);

    // Send to the discriminator model for analysis
    const disc_input = {
        rootPath: rootPath,
        files: files,
        targetFilePath: activeFilePath,
        commitMessage: commitMessage,
        prevEdits: prevEdits
    };
    console.log('==> Sending to discriminator model');
    const discriminatorOutput = await queryDiscriminator(disc_input);
    console.log('==> Discriminator model returned successfully');
    console.log('==> Files to be analyzed:');
    discriminatorOutput.data.forEach(file => {
        console.log('\t*' + file);
    });
    console.log('==> Total no. of files:', files.length);
    console.log('==> No. of files to be analyzed:', discriminatorOutput.data.length);

    // Send the selected files to the locator model for location prediction
    const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename) || filename == activeFilePath);

    console.log("==> Filtered files:")
    console.log(filteredFiles)

    const loc_input = {
        files: filteredFiles,
        targetFilePath: activeFilePath,
        commitMessage: commitMessage,
        prevEdits: prevEdits
    };
    console.log('==> Sending to edit locator model');
    const locatorOutput = await queryLocator(loc_input);
    console.log('==> Edit locator model returned successfully');

    // convert all paths back to absolute paths
    let rawLocations = locatorOutput.data;
    for (const loc of rawLocations) {
        loc.targetFilePath = toAbsPath(rootPath, loc.targetFilePath);
    }
    queryState.updateLocations(rawLocations);
    return rawLocations;
}

async function queryEditFromModel(fileContent, editType, atLines, prevEdits, commitMessage) {
    /* 	
        Generator:
        input:
        { 
            "targetFileContent":    string
            "commitMessage":        string, commit message,
            "editType":             string, edit type,
            "prevEdits":            list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
            "atLines":               list, of edit line indices
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
        atLines: atLines
    };

    commitMessage = await queryState.requireCommitMessage(commitMessage);

    const output = await queryGenerator(input);
    let edits = output.data;
    console.log('==> Edit generator model returned successfully');
    return edits; // Return newmodification
}

export {
    queryLocationFromModel,
    queryEditFromModel,
    queryState
};
