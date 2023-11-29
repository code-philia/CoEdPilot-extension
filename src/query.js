const vscode = require('vscode');
const { toRelPath, getActiveFilePath, toAbsPath } = require('./file');
const { queryDiscriminator, queryLocator, queryGenerator } = require('./model-client');
const { BaseComponent } = require('./base-component');

class QueryState {
    constructor() {
        this.commitMessage = "";
        this.locations = [];
        this.locatedFilePaths = [];
        this._onDidQuery = new vscode.EventEmitter();
        this.onDidQuery = this._onDidQuery.event;
    }

    updateLocations(locations) {
        this.locations = locations;
        if (this.locations.length) {
            this.locatedFilePaths = [...new Set(locations.map((loc) => loc.targetFilePath))];
        }
        this._onDidQuery.fire(this);
    }

    clearLocations() {
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
            "rootPath": str, rootPath,
            "files": list, [[filePath, fileContent], ...],
            "targetFilePath": str, filePath
        }
        output:
        {
            "data": list, [[filePath, fileContent], ...]
        }
	
        Locator:
        input:
        {
            "files": list, [[filePath, fileContent], ...],
            "targetFilePath": str, filePath,
            "commitMessage": str, commit message,
            "prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}
        }
        output:
        {
            "data": 
            [ 
                { 
                    "targetFilePath": str, filePath,
                    "beforeEdit", str, the content before edit for previous edit,
                    "afterEdit", str, the content after edit for previous edit, 
                    "toBeReplaced": str, the content to be replaced, 
                    "startPos": int, start position of the word,
                    "endPos": int, end position of the word,
                    "editType": str, the type of edit, add or remove,
                    "lineBreak": str, '\n', '\r' or '\r\n'
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


async function queryEditFromModel(rootPath, files, location, commitMessage) {
    /* 	
        Generator:
        input:
        { 
            "files": list, [[filePath, fileContent], ...],
            "targetFilePath": string filePath,
            "commitMessage": string, commit message,
            "editType": string, edit type,
            "prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
            "startPos": int, start position,
            "endPos": int, end position,
            "atLine": list, of edit line indices
        }
        output:
        {
            "data": 
            { 
                "targetFilePath": string, filePath of target file,
                "editType": string, 'remove', 'add'
                "startPos": int, start position,
                "endPos": int, end position,
                "replacement": list of strings, replacement content   
            }
        } 
    */

    for (const file_info of files) {
        file_info[0] = toRelPath(
            rootPath,
            file_info[0]
        );
    }

    const input = {
        files: files,
        targetFilePath: location.targetFilePath,
        commitMessage: commitMessage,
        editType: location.editType,
        prevEdits: location.prevEdits,
        startPos: location.startPos,
        endPos: location.endPos,
        atLine: location.atLine
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

module.exports = {
    queryLocationFromModel,
    queryEditFromModel,
    queryState,
    CommitMessageInput
}
