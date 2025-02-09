import vscode from 'vscode';
import fs from 'fs';
import path from 'path';
import { glob } from 'glob';
import { osType } from '../global-result-context';
import { SimpleEdit } from './base-types';
import { globalEditorState } from '../global-workspace-context';

type GlobPatterns = {
    exclude: string[],
    permExclude: string[], // permanently exclude a directory, unable to re-include
    include: string []
};

let gitignorePatterns: GlobPatterns = {
    'exclude': [],
    'permExclude': [],
    'include': []
};


function parseIgnoreLinesToPatterns(lines: string[]) {
    // Like Git, we only glob files in the patterns
    let patterns: GlobPatterns = {
        'exclude': [],
        'permExclude': [],
        'include': []
    };
    let exp = "";
    let prefix = "";
    let suffix = "";
    function addPattern(type: 'exclude' | 'permExclude' | 'include') {
        patterns[type].push(prefix + exp + suffix);
    } 

    for (const line of lines) {
        exp = line.trim();
        if (!exp || exp.startsWith('#')) continue;
        
        let isReverse = false;
        if (exp.startsWith('!')) {
            exp = exp.slice(1).trim();
            isReverse = true;
        }

        prefix = exp.indexOf('/') !== exp.length ? '/' : '/**/';   // check if relative path

        if (exp.endsWith('*')) {            // the same matching as glob
            suffix = '';
            isReverse ? addPattern('include') : addPattern('exclude');
        } else if (exp.endsWith('/')) {     // matches directory only
            suffix = '**';
            isReverse ? addPattern('include') : addPattern('permExclude');      // permanently exclude
        } else {                            // matches all files
            suffix = '/**';
            isReverse ? addPattern('include') : addPattern('permExclude');      // permanently exclude if it's a directory
            suffix = '';
            isReverse ? addPattern('include') : addPattern('exclude');
        }
    }

    patterns["exclude"] = patterns["exclude"].concat(patterns["permExclude"]);
    return patterns;
} 

try {
    const workspacePath = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
    if (workspacePath) {
        const gitignoreText = fs.readFileSync(path.join(
            workspacePath,
            '.gitignore'
        ), 'utf-8');
        const gitignoreLines = gitignoreText.match(/[^\r\n]+/g);
        if (gitignoreLines) {
            gitignorePatterns = parseIgnoreLinesToPatterns(gitignoreLines);
        }
    }
} catch (err) {
    console.log(`Neglecting .gitignore rules: a problem occurs: ${err}`);
}

// BASIC FUNCTIONS

function toDriveLetterLowerCasePath(filePath: string) {
    return fs.realpathSync.native(filePath);
}

// Convert any-style path to POSIX-style path
function toPosixPath(filePath: string) {
    return osType === 'Windows_NT' ?
        filePath.replace(/\\/g, '/')
        : filePath;
}

function toAbsPath(rootPath: string, filePath: string) {
    return toPosixPath(path.join(rootPath, filePath));
}

function toRelPath(rootPath: string, filePath: string) {
    return toPosixPath(path.relative(rootPath, filePath));
}

function getOpenedFilePaths(): Set<string> {
    const openedPaths: Set<string> = new Set();
    for (const tabGroup of vscode.window.tabGroups.all) {
        for (const tab of tabGroup.tabs) {
            if (tab.input instanceof vscode.TabInputText) {
                openedPaths.add(tab.input.uri.fsPath);
            }
        }
    }
    return openedPaths;
}

function getActiveFilePath() {
    const filePath = vscode.window.activeTextEditor?.document?.uri.fsPath;
    return filePath === undefined ? undefined : toPosixPath(filePath);
}

async function getLineInfoInDocument(path: string, lineNo: number) {
    const doc = await vscode.workspace.openTextDocument(path);
    const textLine = doc.lineAt(lineNo);
    return {
        range: textLine.range,
        text: textLine.text
    };
}

function getRootPath() {
    const workspacePath = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
    if (!workspacePath) {
        throw new Error('No workspace folder is opened');
    }
    return toPosixPath(workspacePath);
}

async function readGlobFiles(useSnapshot = true) {
    const rootPath = getRootPath();

    // Use glob to exclude certain files and return a list of all valid files
    
    const absolutePathList = await globFiles(rootPath);

    const fileGetter = useSnapshot
        ? async (filePath: string) => {
            const openedPaths = getOpenedFilePaths();
            return await getStagedFile(openedPaths, toDriveLetterLowerCasePath(filePath));
        }
        : async (filePath: string) => fs.readFileSync(filePath, 'utf-8');
    

    async function readFileFromPathList(filePathList: string[], contentList: [string, string][]) {
        for (const filePath of filePathList) {
            try {
                const stat = fs.statSync(filePath);
                if (stat.isFile()) {
                    const fileContent = await fileGetter(filePath);  // Skip files that cannot be correctly decoded
                    contentList.push([filePath, fileContent]);
                }
            } catch (error) {
                console.log(`Ignoring file: some error occurs when reading file ${filePath}`);
            }
        }
    }

    const fileList: [string, string][] = [];
    await readFileFromPathList(absolutePathList, fileList);
    // Replace directly when reading files, instead of replacing later
    // if (useSnapshot) {
    //     replaceCurrentSnapshot(fileList);
    // }

    return fileList;
}

// Exact match is used here! Ensure the file paths and opened paths are in the same format
async function getStagedFile(openedPaths: Set<string>, filePath: string) {
    return openedPaths.has(filePath)
        ? (await vscode.workspace.openTextDocument(vscode.Uri.file(filePath))).getText()
        : fs.readFileSync(filePath, 'utf-8');
}

function liveFilesGetter() {
    return async (openedPaths: Set<string>, filePath: string) => 
        openedPaths.has(filePath)
            ? (await vscode.workspace.openTextDocument(vscode.Uri.file(filePath))).getText()
            : fs.readFileSync(filePath, 'utf-8');
}

// ABOUT EDIT

function detectEdit(prev: string, curr: string): SimpleEdit {
    // Split the strings into lists of strings by line
    const prevSnapshotStrList = prev.match(/(.*?(?:\r\n|\n|\r))|(.+$)/g) ?? []; // Keep the line break at the end of each line
    const currSnapshotStrList = curr.match(/(.*?(?:\r\n|\n|\r))|(.+$)/g) ?? []; // Keep the line break at the end of each line

    // Find the line number where the difference starts from the beginning
    let start = 0;
    while (start < prevSnapshotStrList.length && start < currSnapshotStrList.length && prevSnapshotStrList[start] === currSnapshotStrList[start]) {
        start++;
    }

    // Find the line number where the difference starts from the end
    let end = 0;
    while (end < prevSnapshotStrList.length - start && end < currSnapshotStrList.length - start && prevSnapshotStrList[prevSnapshotStrList.length - 1 - end] === currSnapshotStrList[currSnapshotStrList.length - 1 - end]) {
        end++;
    }

    // Combine the remaining lines into strings
    const beforeEdit = prevSnapshotStrList.slice(start, prevSnapshotStrList.length - end).join('');
    const afterEdit = currSnapshotStrList.slice(start, currSnapshotStrList.length - end).join('');

    // Find context 
    // const codeAbove = prevSnapshotStrList.slice(Math.max(0, start - 3), start).join('');
    // const codeBelow = prevSnapshotStrList.slice(prevSnapshotStrList.length - end, Math.min(prevSnapshotStrList.length, prevSnapshotStrList.length - end + 3)).join('');

    // Return the result
    return {
        beforeEdit: beforeEdit.trim(),
        afterEdit: afterEdit.trim(),
        // codeAbove: codeAbove.trim(),
        // codeBelow: codeBelow.trim()
    };
}

function pushEdit(item: SimpleEdit) {
    globalEditorState.prevEdits.push(item);

    if (globalEditorState.prevEdits.length > 3) {
        globalEditorState.prevEdits.shift(); // FIFO pop the earliest element
    }
}

// function getLocationAtRange(edits: NativeEditLocation[], document: vscode.TextDocument, range: vscode.Range) {
//     const filePath = toPosixPath(document.uri.fsPath);
//     const startPos = document.offsetAt(range.start);
//     const endPos = document.offsetAt(range.end);
//     return edits.find((mod: NativeEditLocation) => {
//         if (filePath == mod.targetFilePath && mod.startPos <= startPos && endPos <= mod.endPos) {
//             let highlightedRange = new vscode.Range(document.positionAt(mod.startPos), document.positionAt(mod.endPos));
//             const currentToBeReplaced = document.getText(highlightedRange).trim();
//             if (currentToBeReplaced == mod.toBeReplaced.trim()) {
//                 return true;
//             } else {
//                 return false;
//             }
//         }
//     })
// }

//	Try updating prevEdits
// 	If there's new edits return prevEdits
//  Else return null
function updatePrevEdits(lineNo: number) {
    const line = lineNo;
    globalEditorState.currCursorAtLine = line + 1; // VScode API starts counting lines from 0, while our line numbers start from 1, note the +- 1
    console.log(`==> Cursor position: Line ${globalEditorState.prevCursorAtLine} -> ${globalEditorState.currCursorAtLine}`);
    globalEditorState.currSnapshot = vscode.window.activeTextEditor?.document.getText(); // Read the current text in the editor
    if (globalEditorState.prevCursorAtLine !== globalEditorState.currCursorAtLine && globalEditorState.prevCursorAtLine !== 0) { // When the pointer changes position and is not at the first position in the editor
        if (!(globalEditorState.prevSnapshot && globalEditorState.currSnapshot)) {
            return false;
        }
        let edition = detectEdit(globalEditorState.prevSnapshot, globalEditorState.currSnapshot); // Detect changes compared to the previous snapshot

        if (edition.beforeEdit !== edition.afterEdit) {
            // Add the modification to prevEdit
            pushEdit(edition);
            console.log('==> Before edit:\n', edition.beforeEdit);
            console.log('==> After edit:\n', edition.afterEdit);
            globalEditorState.prevSnapshot = globalEditorState.currSnapshot;
            return true;
        }
        return false;
    }
    globalEditorState.prevCursorAtLine = globalEditorState.currCursorAtLine; // Update the line number where the mouse pointer is located
    return false;
}

function getPrevEdits() {
    return globalEditorState.prevEdits;
}

// glob files with specific patterns
async function globFiles(rootPath: string, globPatterns: string[] = []) {
    // Built-in glob patterns
    const globPatternStr = (globPatterns instanceof Array && globPatterns.length > 0)
        ? globPatterns
        : '/**/*';

    const pathList = await glob(globPatternStr, {
        root: rootPath,
        windowsPathsNoEscape: true,
        ignore: gitignorePatterns["exclude"],
        nodir: true
    });
    const reincludedPathList = await glob(gitignorePatterns["include"], {
        root: rootPath,
        windowsPathsNoEscape: true,
        ignore: gitignorePatterns["permExclude"],
        nodir: true
    });
    return pathList.concat(reincludedPathList);
}

function replaceCurrentSnapshot(fileList: [string, string | undefined][]) {
    const activeFilePath = getActiveFilePath();
    const currentFile = fileList.find((file) => file[0] === activeFilePath);
    if (currentFile) {
        currentFile[1] = globalEditorState.currSnapshot; // Use the unsaved content as the actual file content
    }
}

export {
    toPosixPath,
    toAbsPath,
    toRelPath,
    getActiveFilePath,
    getLineInfoInDocument,
    detectEdit,
    pushEdit,
    updatePrevEdits,
    getPrevEdits,
    globFiles,
    replaceCurrentSnapshot,
    getRootPath,
    readGlobFiles,
    getOpenedFilePaths,
    getStagedFile,
    liveFilesGetter
};