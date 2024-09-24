import vscode from 'vscode';
import { diffLines, Change } from 'diff';
import fs from 'fs';
import path from 'path';
import { glob } from 'glob';
import { DisposableComponent } from './base-component';
import { editorState, isActiveEditorLanguageSupported, osType } from '../global-context';
import { statusBarItem } from '../ui/progress-indicator';
import { Edit, SimpleEdit } from './base-types';

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

        prefix = exp.indexOf('/') != exp.length ? '/' : '/**/';   // check if relative path

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

class EditDetector {
    editLimit: number;
    textBaseSnapshots: Map<any, any>;
    editList: Edit[];
    
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
         * 			"addText": string or null, added text, could be null,
         *          "codeAbove": string, content above the edited text, up to 3 lines
         *          "codeBelow": string, content below the edited text, up to 3 lines
         * 		},
         * 		...
         * ]
         */
        this.editList = [];
    }

    clearEditsAndSnapshots() {
        this.textBaseSnapshots = new Map();
        this.editList = [];
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            initFileState(activeEditor);
        }
    }

    hasSnapshot(path: string) {
        return this.textBaseSnapshots.has(path);
    }

    addSnapshot(path: string, text: string) {
        if (!this.hasSnapshot(path)) {
            this.textBaseSnapshots.set(path, text);
        }
    }

    async updateAllSnapshotsFromDocument(getDocument: (path: string) => Promise<string>) {
        // need to fetch text from opened editors
        for (const [path,] of this.textBaseSnapshots) {
            try {
                const text = await getDocument(path);
                this.updateEdits(path, text);
            } catch (err) {
                console.log(`Using saved version: cannot update snapshot on ${path}`)
            }
        }
        this.shiftEdits(undefined);
    }

    updateEdits(path: string, text: string) {
        // Compare old `editList` with new diff on a document
        // All new diffs should be added to edit list, but merge the overlapped/adjoined to the old ones of them 
        // Merge "-" (removed) diff into an overlapped/adjoined old edit
        // Merge "+" (added) diff into an old edit only if its precedented "-" hunk (a zero-line "-" hunk if there's no) wraps the old edit's "-" hunk
        // By default, there could only be zero or one "+" hunk following a "-" hunk
        const newDiffs = diffLines(
            this.textBaseSnapshots.get(path),
            text
        )
        const oldEditsWithIdx: { idx: number, edit: Edit }[] = [];
        const oldEditIndices = new Set();
        this.editList.forEach((edit, idx) => {
            if (edit.path === path) {
                oldEditsWithIdx.push({
                    idx: idx,
                    edit: edit
                });
                oldEditIndices.add(idx);
            }
        })
        
        oldEditsWithIdx.sort((edit1, edit2) => edit1.edit.s - edit2.edit.s);	// sort in starting line order

        const oldAdjustedEditsWithIdx = new Map();
        const newEdits: Edit[] = [];
        let lastLine = 1;
        let oldEditIdx = 0;

        function mergeDiff(rmDiff?: Change, addDiff?: Change) {
            const fromLine = lastLine;
            const toLine = lastLine + (rmDiff?.count ?? 0);

            // construct new edit
            const newEdit = {
                "path": path,
                "s": fromLine,
                "rmLine": rmDiff?.count ?? 0,
                "rmText": rmDiff?.value ?? null,
                "addLine": addDiff?.count ?? 0,
                "addText": addDiff?.value ?? null,
                "codeAbove": "",
                "codeBelow": ""
            };

            // Find context
            const lines = text.split('\n');
            const startAbove = Math.max(0, fromLine - 4);
            const endAbove = fromLine - 1;
            const startBelow = toLine;
            const endBelow = Math.min(lines.length, toLine + 3);

            // Get the lines above and below
            newEdit.codeAbove = lines.slice(startAbove, endAbove).join('\n');
            newEdit.codeBelow = lines.slice(startBelow, endBelow).join('\n');

            // skip all old edits between this diff and the last diff
            while (
                oldEditIdx < oldEditsWithIdx.length &&
				oldEditsWithIdx[oldEditIdx].edit.s + oldEditsWithIdx[oldEditIdx].edit.rmLine < fromLine
            ) {
                ++oldEditIdx;
                // oldAdjustedEditsWithIdx.push(oldEditsWithIdx[oldEditIdx]);
            }

            // if current edit is overlapped/adjoined with this diff
			if (
				oldEditIdx < oldEditsWithIdx.length &&
				oldEditsWithIdx[oldEditIdx].edit.s <= toLine
			) {
                // replace the all the overlapped/adjoined old edits with the new edit
                const fromIdx = oldEditIdx;
                while (
                    oldEditIdx < oldEditsWithIdx.length &&
                    oldEditsWithIdx[oldEditIdx].edit.s <= toLine
                ) {
                    ++oldEditIdx;
                }
                // use the maximum index of the overlapped/adjoined old edits	---------->  Is it necessary?
                const minIdx = Math.max.apply(
                    null,
                    oldEditsWithIdx.slice(fromIdx, oldEditIdx).map((edit) => edit.idx)
                );
                oldAdjustedEditsWithIdx.set(minIdx, newEdit);
				// // skip the edit
				// ++oldEditIdx;
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
                    mergeDiff(diff, undefined);
                }
            } else if (diff.added) {
                // deal with a "+" diff not following a "-" diff
                mergeDiff(undefined, diff);
            }

			if (!(diff.added)) {
				lastLine += diff.count ?? 0;
			}
        }

        const oldAdjustedEdits: Edit[] = [];
		this.editList.forEach((edit, idx) => {
			if (oldEditIndices.has(idx)) {
				if (oldAdjustedEditsWithIdx.has(idx)) {
					oldAdjustedEdits.push(oldAdjustedEditsWithIdx.get(idx));
				}
			} else {
				oldAdjustedEdits.push(edit);
			}
		});

		this.editList = oldAdjustedEdits.concat(newEdits);
    }

    // Shift editList if out of capacity
    // For every overflown edit, apply it and update the document snapshots on which the edits base
    shiftEdits(numShifted?: number) {
        // filter all removed edits
        const numRemovedEdits = numShifted ?? this.editList.length - this.editLimit;
        if (numRemovedEdits <= 0) {
            return;
        }
        const removedEdits = new Set(this.editList.slice(
            0,
            numRemovedEdits
        ));
		
		function performEdits(doc: string, edits: Edit[]) {
            const lines = doc.match(/[^\r\n]*(\r?\n|\r\n|$)/g);
            if (!lines) return;

			const addedLines = Array(lines.length).fill("");
			for (const edit of edits) {
				const s = edit.s - 1;  // zero-based starting line
				for (let i = s; i < s + edit.rmLine; ++i) {
					lines[i] = "";
				}
				addedLines[s] = edit.addText ?? "";
			}
			return lines
                .map((x, i) => addedLines[i] + x)
                .join("");
		}
		
		// for each file involved in the removed edits
        const affectedPaths = new Set(
			[...removedEdits].map((edit) => edit.path)
			);
		for (const filePath of affectedPaths) {
			const doc = this.textBaseSnapshots.get(filePath);
			const editsOnPath = this.editList
				.filter((edit) => edit.path === filePath)
				.sort((edit1, edit2) => edit1.s - edit2.s);
				
			// execute removed edits
			const removedEditsOnPath = editsOnPath.filter((edit) => removedEdits.has(edit));
			this.textBaseSnapshots.set(filePath, performEdits(doc, removedEditsOnPath));
			
			// rebase other edits in file
			let offsetLines = 0;
			for (let edit of editsOnPath) {
				if (removedEdits.has(edit)) {
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
    async getEditList() {
        return this.editList.map((edit) => ({
			"beforeEdit": edit.rmText?.trim() ?? "",
            "afterEdit": edit.addText?.trim() ?? "",
            "codeAbove": edit.codeAbove.trim(),
            "codeBelow": edit.codeBelow.trim()
        }))
    }

    async getUpdatedEditList() {
        const liveGetter = liveFilesGetter();
        const openedDocuments = getOpenedFilePaths();
        const docGetter = async (filePath: string) => {
            return await liveGetter(openedDocuments, filePath);
        }
        await globalEditDetector.updateAllSnapshotsFromDocument(docGetter);
        return await globalEditDetector.getEditList();
    }
}

const globalEditDetector = new EditDetector();

// BASIC FUNCTIONS

function toDriveLetterLowerCasePath(filePath: string) {
    return fs.realpathSync.native(filePath);
}

// Convert any-style path to POSIX-style path
function toPosixPath(filePath: string) {
    return osType == 'Windows_NT' ?
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
    }
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
    const filePathList = await globFiles(rootPath);
    const fileGetter = useSnapshot
        ? async (filePath: string) => {
            const liveGetter = liveFilesGetter();
            const openedPaths = getOpenedFilePaths();
            return await liveGetter(openedPaths, toDriveLetterLowerCasePath(filePath))
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
    await readFileFromPathList(filePathList, fileList);
    // Replace directly when reading files, instead of replacing later
    // if (useSnapshot) {
    //     replaceCurrentSnapshot(fileList);
    // }

    return fileList;
}

// Exact match is used here! Ensure the file paths and opened paths are in the same format
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
    const codeAbove = prevSnapshotStrList.slice(Math.max(0, start - 3), start).join('');
    const codeBelow = prevSnapshotStrList.slice(prevSnapshotStrList.length - end, Math.min(prevSnapshotStrList.length, prevSnapshotStrList.length - end + 3)).join('');

    // Return the result
    return {
        beforeEdit: beforeEdit.trim(),
        afterEdit: afterEdit.trim(),
        codeAbove: codeAbove.trim(),
        codeBelow: codeBelow.trim()
    };
}

function pushEdit(item: SimpleEdit) {
    editorState.prevEdits.push(item);

    if (editorState.prevEdits.length > 3) {
        editorState.prevEdits.shift(); // FIFO pop the earliest element
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
    editorState.currCursorAtLine = line + 1; // VScode API starts counting lines from 0, while our line numbers start from 1, note the +- 1
    console.log(`==> Cursor position: Line ${editorState.prevCursorAtLine} -> ${editorState.currCursorAtLine}`);
    editorState.currSnapshot = vscode.window.activeTextEditor?.document.getText(); // Read the current text in the editor
    if (editorState.prevCursorAtLine != editorState.currCursorAtLine && editorState.prevCursorAtLine != 0) { // When the pointer changes position and is not at the first position in the editor
        if (!(editorState.prevSnapshot && editorState.currSnapshot)) {
            return false;
        }
        let edition = detectEdit(editorState.prevSnapshot, editorState.currSnapshot); // Detect changes compared to the previous snapshot

        if (edition.beforeEdit != edition.afterEdit) {
            // Add the modification to prevEdit
            pushEdit(edition);
            console.log('==> Before edit:\n', edition.beforeEdit);
            console.log('==> After edit:\n', edition.afterEdit);
            editorState.prevSnapshot = editorState.currSnapshot;
            return true;
        }
        return false;
    }
    editorState.prevCursorAtLine = editorState.currCursorAtLine; // Update the line number where the mouse pointer is located
    return false;
}

function getPrevEdits() {
    return editorState.prevEdits;
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
        currentFile[1] = editorState.currSnapshot; // Use the unsaved content as the actual file content
    }
}

function initFileState(editor: vscode.TextEditor | undefined) {
    if (!editor) return;
    editorState.inDiffEditor = (vscode.window.tabGroups.activeTabGroup.activeTab?.input instanceof vscode.TabInputTextDiff);
    editorState.language = vscode.window.activeTextEditor?.document?.languageId.toLowerCase() ?? "unknown";
    statusBarItem.setStatusDefault(true);

    // update file snapshot
    const currUri = editor?.document?.uri;
    const currPath = currUri?.fsPath;
    if (currUri && currUri.scheme === "file" && currPath && !(globalEditDetector.hasSnapshot(currPath))) {
        globalEditDetector.addSnapshot(currPath, editor.document.getText());
    }

    let isEditDiff = false;
    if (editorState.inDiffEditor) {
        const input = vscode.window.tabGroups.activeTabGroup?.activeTab?.input;
        isEditDiff = (input instanceof vscode.TabInputTextDiff)
            && input.original.scheme === 'temp'
            && input.modified.scheme === 'file';
    }

    if (vscode.workspace.getConfiguration("coEdPilot").get("predictLocationOnEditAcception") && editorState.toPredictLocation) {
        vscode.commands.executeCommand("coEdPilot.predictLocations");
        editorState.toPredictLocation = false;
    }
    vscode.commands.executeCommand('setContext', 'coEdPilot:isEditDiff', isEditDiff);
    vscode.commands.executeCommand('setContext', 'coEdPilot:isLanguageSupported', isActiveEditorLanguageSupported());
}

class FileStateMonitor extends DisposableComponent{
    constructor() {
        super();
        this.register(
            vscode.window.onDidChangeActiveTextEditor(initFileState)
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
    updatePrevEdits,
    getPrevEdits,
    globFiles,
    replaceCurrentSnapshot,
    getRootPath,
    readGlobFiles,
    editorState,
    initFileState,
    FileStateMonitor,
    EditDetector,
    getOpenedFilePaths,
    liveFilesGetter
};