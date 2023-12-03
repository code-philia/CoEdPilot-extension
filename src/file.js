import vscode from 'vscode';
import { diffTrimmedLines } from 'diff';
import fs from 'fs';
import path from 'path';
import glob from 'glob';
import os from 'os';
import { BaseComponent } from './base-component';

const prevEditNum = 3;

class FileState extends BaseComponent{
    constructor() {
        super();
        this.prevCursorAtLine = 0;
        this.currCursorAtLine = 0;
        this.prevSnapshot = undefined;
        this.currSnapshot = undefined;
        this.prevEdits = [];
        this.inDiffEditor = false;
    }
}

const fileState = new FileState();

const osType = os.type();
const supportedOSTypes = ['Windows_NT', 'Darwin', 'Linux'];

if (!supportedOSTypes.includes(osType)) {
    throw RangeError(`Operating system (node detected: ${osType}) is not supported yet.`);
}

const defaultLineBreaks = {
    'Windows_NT': '\r\n',
    'Darwin': '\r',
    'Linux': '\n'
};
const defaultLineBreak = defaultLineBreaks[osType] ?? '\n';

class EditDetector {
    constructor() {
        this.editLimit = 10;
        this.textBaseSnapshots = new Map();

        /**
         * Edit list, in which all edits are based on `textBaseSnapshots`, is like:
         * [
         * 		{
         * 			"path": string, the file path,
         * 			"s": int, starting line,
         * 			"rmLine": int, number of removed lines, 
         * 			"rmText": string or null, removed text, could be null
         * 			"addLine": int, number of added lines,
         * 			"addText": string or null, number of added text, could be null
         * 		},
         * 		...
         * ]
         */
        this.editList = [];
    }

    hasSnapshot(path) {
        return this.textBaseSnapshots.has(path);
    }

    updateSnapshot(path, text) {
        this.textBaseSnapshots[path] = text;
    }

    updateAllSnapshotsFromDocument() {
        for (const [path,] of this.textBaseSnapshots) {
            try {
                const text = fs.readFileSync(path, 'utf-8');	// won't throw error on non-utf-8 character here
                this.updateEdits(path, text);
            } catch (err) {
                console.log('Cannot update snapshot on ${path}$')
            }
        }
    }

    updateEdits(path, text) {
        if (this.textBaseSnapshots.has(path)) {
            return;
        }

        // Compare old `editList` with new diff on a document
        // All new diffs should be added to edit list, but merge some of them to the old ones
        // Merge "-" (removed) diff into an overlapped old edit
        // Merge "+" (added) diff into an old edit only if its precedented "-" hunk (a zero-line "-" hunk if there's no) wraps the old edit's "-" hunk
        // By default, there could only be zero or one "+" hunk following a "-" hunk
        const newDiffs = diffTrimmedLines(
            this.textBaseSnapshots[path],
            text
        )
        const oldEditsWithIdx = this.editList.
            filter((edit) => edit.path === path)
            .map((edit, idx) => {
                return {
                    idx: idx,
                    edit: edit
                }
            });		// keep index order (aka. time order)

        oldEditsWithIdx.sort((edit1, edit2) => edit1.edit.s - edit2.edit.s);	// sort in starting line order

        const oldAdjustedEditsWithIdx = [];
        const newEdits = [];
        let lastLine = 1;
        let oldEditIdx = 0;

        function mergeDiff(rmDiff, addDiff) {
            const fromLine = lastLine;
            const toLine = lastLine + (rmDiff.count ?? 0);

            // construct new edit
            const newEdit = {
                "path": path,
                "s": fromLine,
                "rmLine": rmDiff.count ?? 0,
                "rmText": rmDiff.value ?? null,
                "addLine": addDiff.count ?? 0,
                "addText": addDiff.count ?? null,
            };

            // skip all old edits between this diff and the last diff
            while (
                oldEditIdx < oldEditsWithIdx.length &&
                oldEditsWithIdx[oldEditIdx].edit.s +
                oldEditsWithIdx[oldEditIdx].edit.rmLine <= lastLine
            ) {
                ++oldEditIdx;
                oldAdjustedEditsWithIdx.push(oldEditsWithIdx[oldEditIdx]);
            }

            // if current edit is overlapped with this diff
            if (oldEditsWithIdx[oldEditIdx].edit.s < toLine) {
                // replace the all the overlapped old edits with the new edit
                const fromIdx = oldEditIdx;
                while (
                    oldEditIdx < oldEditsWithIdx.length &&
                    oldEditsWithIdx[oldEditIdx].edit.s +
                    oldEditsWithIdx[oldEditIdx].edit.rmLine <= toLine
                ) {
                    ++oldEditIdx;
                }
                // use the maximum index of the overlapped old edits	---------->  Is it necessary?
                const minIdx = Math.max.apply(
                    null,
                    oldEditsWithIdx.slice(fromIdx, oldEditIdx).map((edit) => edit.idx)
                );
                oldAdjustedEditsWithIdx.push({
                    idx: minIdx,
                    edit: newEdit
                })
            } else {
                // simply add the edit as a new edit
                newEdits.push(newEdit);
            }
        }

        for (let i = 0; i < newDiffs.length; ++i) {
            const diff = newDiffs[i];

            if (diff.removed) {
                // unite the following "+" (added) diff
                if (i + 1 < newDiffs.length && newDiffs[i + 1].added) {
                    mergeDiff(diff, newDiffs[i + 1]);
                    ++i;
                } else {
                    mergeDiff(diff, null);
                }
            } else if (diff.added) {
                // deal with a "+" diff not following a "-" diff
                mergeDiff(null, diff);
            }

            lastLine += diff.count;
        }
    }

    // Shift editList if out of capacity
    // For every overflown edit, apply it and update the document snapshots on which the edits base
    shiftEdits() {
        // filter all removed edits
        const numRemovedEdits = this.editList.length - this.editLimit;
        if (numRemovedEdits <= 0) {
            return;
        }
        const removedEdits = this.editList.slice(
            0,
            numRemovedEdits
        );

        // for each file involved in the removed edits
        // rebase other edits in file
        const affectedPaths = new Set(
            removedEdits.map((edit) => edit.path)
        );
        for (const filePath of affectedPaths) {
            const editsOnPath = this.editList
                .filter((edit) => edit.path === filePath)
                .sort((edit1, edit2) => edit1.s - edit2.s);

            let offsetLines = 0;
            for (let edit of editsOnPath) {
                if (edit in removedEdits) {
                    offsetLines = offsetLines - edit.rmLine + edit.addLine;
                } else {
                    edit.s += offsetLines;
                }
            }
        }

        this.editList.splice(0, numRemovedEdits);
    }

    /**
     * Return edit list in such format:
     * [
     * 		{
     * 			"beforeEdit": string, the deleted hunk, could be null;
     * 			"afterEdit": string, the added hunk, could be null;
     * 		},
     * 		...
     * ]
     */
    getEditListText() {
        return this.editList.map((edit) => ({
            "beforeEdit": edit.rmText ?? "",
            "afterEdit": edit.addText ?? ""
        }))
    }
}

const globalEditDetector = EditDetector;

// BASIC FUNCTIONS

// Convert any-style path to POSIX-style path
function toPosixPath(filePath) {
    return osType == 'Windows_NT' ?
        filePath.replace(/\\/g, '/')
        : filePath;
}

function toAbsPath(rootPath, filePath) {
    return toPosixPath(path.join(rootPath, filePath));
}

function toRelPath(rootPath, filePath) {
    return toPosixPath(path.relative(rootPath, filePath));
}

function getActiveFilePath() {
    const filePath = vscode.window.activeTextEditor?.document?.fileName;
    return filePath ?? toPosixPath(filePath);
}

async function getLineInfoInDocument(path, lineNo) {
    const doc = await vscode.workspace.openTextDocument(path);
    const textLine = doc.lineAt(lineNo);
    return {
        range: textLine.range,
        text: textLine.text
    }
}

function getRootPath() {
    return toPosixPath(vscode.workspace.workspaceFolders[0].uri.fsPath);
}

async function getFiles(useSnapshot = true) {
    const rootPath = getRootPath();
    const fileList = [];

    // Use glob to exclude certain files and return a list of all valid files
    const filePathList = globFiles(rootPath, []);

    async function readFileFromPathList(filePathList, contentList) {
        for (const filePath of filePathList) {
            try {
                const stat = fs.statSync(filePath);
                if (stat.isFile()) {
                    const fileContent = await fs.promises.readFile(filePath, 'utf-8');  // Skip files that cannot be correctly decoded
                    contentList.push([filePath, fileContent]);
                }
            } catch (error) {
                console.log("Some error occurs when reading file");
            }
        }
    }

    await readFileFromPathList(filePathList, fileList);
    // Replace directly when reading files, instead of replacing later
    if (useSnapshot) {
        replaceCurrentSnapshot(fileList);
    }

    return fileList;
}

// ABOUT EDIT

function detectEdit(prev, curr) {
    // Split the strings into lists of strings by line
    const prevSnapshotStrList = prev.match(/(.*?(?:\r\n|\n|\r|$))/g).slice(0, -1); // Keep the line break at the end of each line
    const currSnapshotStrList = curr.match(/(.*?(?:\r\n|\n|\r|$))/g).slice(0, -1); // Keep the line break at the end of each line

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

    // Return the result
    return {
        beforeEdit: beforeEdit.trim(),
        afterEdit: afterEdit.trim()
    };
}

function pushEdit(item) {
    fileState.prevEdits.push(item);

    if (fileState.prevEdits.length > prevEditNum) {
        fileState.prevEdits.shift(); // FIFO pop the earliest element
    }
}

function getLocationAtRange(edits, document, range) {
    const filePath = toPosixPath(document.uri.fsPath);
    const startPos = document.offsetAt(range.start);
    const endPos = document.offsetAt(range.end);
    return edits.find((mod) => {
        if (filePath == mod.targetFilePath && mod.startPos <= startPos && endPos <= mod.endPos) {
            let highlightedRange = new vscode.Range(document.positionAt(mod.startPos), document.positionAt(mod.endPos));
            const currentToBeReplaced = document.getText(highlightedRange).trim();
            if (currentToBeReplaced == mod.toBeReplaced.trim()) {
                return true;
            } else {
                return false;
            }
        }
    })
}

//	Try updating prevEdits
// 	If there's new edits return prevEdits
//  Else return null
function updatePrevEdits(lineNo) {
    const line = lineNo;
    fileState.currCursorAtLine = line + 1; // VScode API starts counting lines from 0, while our line numbers start from 1, note the +- 1
    console.log(`==> Cursor position: Line ${fileState.prevCursorAtLine} -> ${fileState.currCursorAtLine}`);
    fileState.currSnapshot = vscode.window.activeTextEditor.document.getText(); // Read the current text in the editor
    if (fileState.prevCursorAtLine != fileState.currCursorAtLine && fileState.prevCursorAtLine != 0) { // When the pointer changes position and is not at the first position in the editor
        let edition = detectEdit(fileState.prevSnapshot, fileState.currSnapshot); // Detect changes compared to the previous snapshot

        if (edition.beforeEdit != edition.afterEdit) {
            // Add the modification to prevEdit
            pushEdit(edition);
            console.log('==> Before edit:\n', edition.beforeEdit);
            console.log('==> After edit:\n', edition.afterEdit);
            fileState.prevSnapshot = fileState.currSnapshot;
            return true;
        }
        return false;
    }
    fileState.prevCursorAtLine = fileState.currCursorAtLine; // Update the line number where the mouse pointer is located
    return false;
}

function getPrevEdits() {
    return fileState.prevEdits;
}

// glob files with specific patterns
function globFiles(rootPath, globPatterns) {
    // Built-in glob patterns
    const defaultGlobPatterns = [
        '!.git/**'
    ];
    const allPatterns = defaultGlobPatterns.concat(globPatterns);

    // Concatenate glob patterns
    function concatRoot(pattern) {
        const concat_path = pattern[0] == '!' ? '!' + path.join(rootPath, pattern.slice(1)) : path.join(rootPath, pattern);
        return concat_path;
    }
    const allPatternsWithRoot = allPatterns.map(concatRoot).concat([path.join(rootPath, '**')]);

    const pathList = glob.sync('{' + allPatternsWithRoot.join(',') + '}', { windowsPathsNoEscape: true });
    return pathList;
}

function replaceCurrentSnapshot(fileList) {
    const currentFile = fileList.find((file) => file[0] === getActiveFilePath);
    if (currentFile) {
        currentFile[1] = fileState.currSnapshot; // Use the unsaved content as the actual file content
    }
}

function initFileState(editor) {
    if (!editor) return;
    fileState.prevCursorAtLine = 0;
    fileState.currCursorAtLine = 0;
    fileState.prevSnapshot = editor.document.getText();
    fileState.currSnapshot = editor.document.getText();
    fileState.inDiffEditor = (vscode.window.tabGroups.activeTabGroup.activeTab.input instanceof vscode.TabInputTextDiff);
    console.log('==> Active File:', getActiveFilePath());
    console.log('==> Global variables initialized');
    // highlightModifications(modifications, editor);
}

class FileStateMonitor extends BaseComponent{
    constructor() {
        super();
        this.register(
            vscode.window.onDidChangeActiveTextEditor(initFileState),
            vscode.window.onDidChangeTextEditorSelection((event) => {
                updatePrevEdits(event.selections[0].active.line);
            })
        );
    }

    
}

export {
    globalEditDetector,
    toPosixPath,
    toAbsPath,
    toRelPath,
    getActiveFilePath,
    getLineInfoInDocument,
    detectEdit,
    pushEdit,
    getLocationAtRange,
    updatePrevEdits,
    getPrevEdits,
    globFiles,
    replaceCurrentSnapshot,
    getRootPath,
    getFiles,
    fileState,
    initFileState,
    FileStateMonitor,
    defaultLineBreak
};