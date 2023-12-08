import vscode from 'vscode';
import { diffLines } from 'diff';
import fs from 'fs';
import path from 'path';
import { glob } from 'glob';
import { BaseComponent } from './base-component';
import { editorState, osType } from './global-context';
import { statusBarItem } from './status-bar';

let gitignorePatterns = {
    'exclude': [],
    'permExclude': [], // permanently exclude a directory, unable to re-include
    'include': []
};

function parseIgnoreLinesToPatterns(lines) {
    // Like Git, we only glob files in the patterns
    let patterns = {
        'exclude': [],
        'permExclude': [], // permanently exclude a directory, unable to re-include
        'include': []
    };

    let exp = "";
    let prefix = "";
    let suffix = "";
    function addPattern(type) {
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
    const gitignoreText = fs.readFileSync(path.join(
        vscode.workspace.workspaceFolders[0].uri.fsPath,
        '.gitignore'
    ), 'utf-8');
    const gitignoreLines = gitignoreText.match(/[^\r\n]+/g);
    gitignorePatterns = parseIgnoreLinesToPatterns(gitignoreLines);
} catch (err) {
    console.log(`Neglecting .gitignore rules: a problem occurs: ${err}`);
}

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

    addSnapshot(path, text) {
        if (!this.hasSnapshot(path)) {
            this.textBaseSnapshots.set(path, text);
        }
    }

    async updateAllSnapshotsFromDocument(getDocument) {
        // need to fetch text from opened editors
        for (const [path,] of this.textBaseSnapshots) {
            try {
                const text = await getDocument(path);
                this.updateEdits(path, text);
            } catch (err) {
                console.log(`Using saved version: cannot update snapshot on ${path}`)
            }
        }
        this.shiftEdits();
    }

    updateEdits(path, text) {
        // Compare old `editList` with new diff on a document
        // All new diffs should be added to edit list, but merge the overlapped/adjoined to the old ones of them 
        // Merge "-" (removed) diff into an overlapped/adjoined old edit
        // Merge "+" (added) diff into an old edit only if its precedented "-" hunk (a zero-line "-" hunk if there's no) wraps the old edit's "-" hunk
        // By default, there could only be zero or one "+" hunk following a "-" hunk
        const newDiffs = diffLines(
            this.textBaseSnapshots.get(path),
            text
        )
        const oldEditsWithIdx = [];
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
        const newEdits = [];
        let lastLine = 1;
        let oldEditIdx = 0;

        function mergeDiff(rmDiff, addDiff) {
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
            };

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
                    mergeDiff(diff, null);
                }
            } else if (diff.added) {
                // deal with a "+" diff not following a "-" diff
                mergeDiff(null, diff);
            }

			if (!(diff.added)) {
				lastLine += diff.count;
			}
        }

        const oldAdjustedEdits = [];
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
    shiftEdits(numShifted) {
        // filter all removed edits
        const numRemovedEdits = numShifted ?? this.editList.length - this.editLimit;
        if (numRemovedEdits <= 0) {
            return;
        }
        const removedEdits = new Set(this.editList.slice(
            0,
            numRemovedEdits
        ));
		
		function performEdits(doc, edits) {
			const lines = doc.match(/[^\r\n]*(\r?\n|\r\n|$)/g);
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
            "afterEdit": edit.addText?.trim() ?? ""
        }))
    }

    async getUpdatedEditList() {
        await globalEditDetector.updateAllSnapshotsFromDocument(liveFilesGetter());
        return await globalEditDetector.getEditList();
    }
}

const globalEditDetector = new EditDetector();

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

function getOpenedFilePaths() {
    const openedPaths = new Set();
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

async function getGlobFiles(useSnapshot = true) {
    const rootPath = getRootPath();
    const fileList = [];

    // Use glob to exclude certain files and return a list of all valid files
    const filePathList = globFiles(rootPath);
    const fileGetter = useSnapshot
        ? liveFilesGetter()
        : async (filePath) => fs.readFileSync(filePath, 'utf-8');

    async function readFileFromPathList(filePathList, contentList) {
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

    await readFileFromPathList(filePathList, fileList);
    // Replace directly when reading files, instead of replacing later
    // if (useSnapshot) {
    //     replaceCurrentSnapshot(fileList);
    // }

    return fileList;
}

function liveFilesGetter() {
    const openedPaths = getOpenedFilePaths();
    return async (filePath) => 
        openedPaths.has(filePath)
            ? (await vscode.workspace.openTextDocument(vscode.Uri.file(filePath))).getText()
            : fs.readFileSync(filePath, 'utf-8');
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
    editorState.prevEdits.push(item);

    if (editorState.prevEdits.length > 3) {
        editorState.prevEdits.shift(); // FIFO pop the earliest element
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
    editorState.currCursorAtLine = line + 1; // VScode API starts counting lines from 0, while our line numbers start from 1, note the +- 1
    console.log(`==> Cursor position: Line ${editorState.prevCursorAtLine} -> ${editorState.currCursorAtLine}`);
    editorState.currSnapshot = vscode.window.activeTextEditor.document.getText(); // Read the current text in the editor
    if (editorState.prevCursorAtLine != editorState.currCursorAtLine && editorState.prevCursorAtLine != 0) { // When the pointer changes position and is not at the first position in the editor
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
function globFiles(rootPath, globPatterns = []) {
    // Built-in glob patterns
    const globPatternStr = (globPatterns instanceof Array && globPatterns.length > 0)
        ? globPatterns
        : '/**/*';

    const pathList = glob.sync(globPatternStr, {
        root: rootPath,
        windowsPathsNoEscape: true,
        ignore: gitignorePatterns["exclude"],
        nodir: true
    });
    const reincludedPathList = glob.sync(gitignorePatterns["include"], {
        root: rootPath,
        windowsPathsNoEscape: true,
        ignore: gitignorePatterns["permExclude"],
        nodir: true
    });
    return pathList.concat(reincludedPathList);
}

function replaceCurrentSnapshot(fileList) {
    const currentFile = fileList.find((file) => file[0] === getActiveFilePath);
    if (currentFile) {
        currentFile[1] = editorState.currSnapshot; // Use the unsaved content as the actual file content
    }
}

function initFileState(editor) {
    if (!editor) return;
    editorState.inDiffEditor = (vscode.window.tabGroups.activeTabGroup.activeTab.input instanceof vscode.TabInputTextDiff);
    editorState.language = vscode.window.activeTextEditor?.document?.languageId.toLowerCase();
    statusBarItem.setStatusDefault(true);

    // update file snapshot
    const currUri = editor?.document?.uri;
    const currPath = currUri?.fsPath;
    if (currUri && currUri.scheme === "file" && currPath && !(globalEditDetector.hasSnapshot(currPath))) {
        globalEditDetector.addSnapshot(currPath, editor.document.getText());
    }

    console.log('==> Active File:', getActiveFilePath());
    console.log('==> Global variables initialized');
}

class FileStateMonitor extends BaseComponent{
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
    getLocationAtRange,
    updatePrevEdits,
    getPrevEdits,
    globFiles,
    replaceCurrentSnapshot,
    getRootPath,
    getGlobFiles,
    editorState,
    initFileState,
    FileStateMonitor,
    EditDetector,
    getOpenedFilePaths,
    liveFilesGetter
};