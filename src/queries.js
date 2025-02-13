import { queryState } from "./global-context";
import { toRelPath, getActiveFilePath, toAbsPath, getLineInfoInDocument } from "./file";
import { queryDiscriminator, queryLocator, queryGenerator } from "./model-client";
import { statusBarItem } from "./status-bar";

// ------------ Extension States -------------
async function queryLocationFromModel(rootPath, files, prevEdits, commitMessage, language) {
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
                    "lineBreak":        str, "\n", "\r" or "\r\n",
                    "atLines":          number, line number (beginning from 1) of the location,
                    "confidence":       number, float point confidence within [0, 1]
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
        prevEdits: prevEdits,
        language: language
    };
    const discriminatorOutput = await queryDiscriminator(disc_input);
    console.log("==> Discriminated files to be analyzed:");
    discriminatorOutput.data.forEach(file => {
        console.log("\t*" + file);
    });
    console.log("==> Total no. of files:", files.length);
    console.log("==> No. of files to be analyzed:", discriminatorOutput.data.length);

    // Send the selected files to the locator model for location prediction
    const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename) || filename == activeFilePath);

    console.log("==> Filtered files:");
    console.log(filteredFiles);

    const loc_input = {
        files: filteredFiles,
        targetFilePath: activeFilePath,
        commitMessage: commitMessage,
        prevEdits: prevEdits,
        language: language
    };
    statusBarItem.setStatusQuerying("locator");
    const locatorOutput = await queryLocator(loc_input);

    // convert all paths back to absolute paths
    let rawLocations = locatorOutput.data;
    for (const loc of rawLocations) {
        loc.targetFilePath = toAbsPath(rootPath, loc.targetFilePath);
        loc.lineInfo = await getLineInfoInDocument(loc.targetFilePath, loc.atLines[0]);
    }
    queryState.updateLocations(rawLocations);
    return rawLocations;
}

async function queryEditFromModel(fileContent, editType, atLines, prevEdits, commitMessage, language) {
    /* 	
        Generator:
        input:
        { 
            "targetFileContent":    string
            "commitMessage":        string, edit description,
            "editType":             string, edit type,
            "prevEdits":            list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
            "atLines":               list, of edit line indices
        }
        output:
        {
            "data": 
            { 
                "editType":         string, "remove", "add"
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

    const output = await queryGenerator(input);
    let edits = output.data;
    return edits; // Return newmodification
}

export {
    queryLocationFromModel,
    queryEditFromModel,
    queryState
};
