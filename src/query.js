import vscode from 'vscode';
import { toRelPath, getActiveFilePath, toAbsPath, getLineInfoInDocument, getRootPath } from './file';
import { queryDiscriminator, queryLocator, queryGenerator } from './model-client';
import { BaseComponent } from './base-component';

class QueryState {
    constructor() {
        this.commitMessage = "";
        this.locations = [];
        this.locatedFilePaths = [];
        this._onDidQuery = new vscode.EventEmitter();
        this.onDidQuery = this._onDidQuery.event;
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

    dispose() {
        this._onDidQuery.dispose();
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
    if (discriminatorOutput.data.length == 0) {
        console.log('==> No files will be analyzed');
        return;
    }
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

    const output = await queryGenerator(input);
    let edits = output.data;
    console.log('==> Edit generator model returned successfully');
    return edits; // Return newmodification
}

class CommitMessageInput extends BaseComponent{
    constructor() {
        super();
        this.inputBox = vscode.window.createInputBox();
        this.inputBox.prompt = 'Enter edit description';
        this.inputBox.ignoreFocusOut = true; // The input box will not be hidden after losing focus
        
        this.register(
            this.showInputBox,
            this.acceptInputBox
        );
    }

    showInputBox() {
        console.log('==> Edit description input box is displayed')
        this.inputBox.show();
    }

    acceptInputBox() {
        const userInput = this.inputBox.value;
        console.log('==> Edit description:', userInput);
        queryState.commitMessage = userInput;
        this.inputBox.hide();
    }
}

export {
    queryLocationFromModel,
    queryEditFromModel,
    queryState,
    CommitMessageInput
};
