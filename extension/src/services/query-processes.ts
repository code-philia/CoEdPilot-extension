import { globalQueryContext } from '../global-result-context';
import { toRelPath, getActiveFilePath, toAbsPath, getLineInfoInDocument } from '../utils/file-utils';
import { postRequestToDiscriminator, postRequestToLocator, postRequestToGenerator } from './backend-requests';
import { statusBarItem } from '../ui/progress-indicator';
import { EditType, SimpleEdit } from '../utils/base-types';

async function startLocationQueryProcess(
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

    // console.log('==> Discriminated files to be analyzed:');
    // discriminatorOutput.data.forEach((file: string) => {
    //     console.log('\t*' + file);
    // });
    // console.log('==> Total no. of files:', files.length);
    // console.log('==> No. of files to be analyzed:', discriminatorOutput.data.length);

    // Send the selected files to the locator model for location prediction
    // const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename) || filename == activeFilePath);
    const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename));

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
    const locatorOutput = await postRequestToLocator(loc_input);

    // convert all paths back to absolute paths
    let rawLocations = locatorOutput.data;
    for (const loc of rawLocations) {
        loc.targetFilePath = toAbsPath(rootPath, loc.targetFilePath);
        loc.lineInfo = await getLineInfoInDocument(loc.targetFilePath, loc.atLines[0]);
    }
    globalQueryContext.updateLocations(rawLocations);
    return rawLocations;
}

async function startEditQueryProcess(
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
    let edits = output.data;
    return edits; // Return newmodification
}

export {
    startLocationQueryProcess,
    startEditQueryProcess
};
