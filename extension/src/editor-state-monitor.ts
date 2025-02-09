import * as vscode from 'vscode';
import { globalEditorState } from './global-workspace-context';
import { statusBarItem } from './ui/progress-indicator';
import { Change, diffLines } from 'diff';
import { Edit } from "./utils/base-types";
import { DisposableComponent } from "./utils/base-component";
import { getOpenedFilePaths, getStagedFile } from './utils/file-utils';
import { globalQueryContext } from './global-result-context';

// TODO add tests for this
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
            updateEditorState(activeEditor);
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
                console.warn(`Using saved version: cannot update snapshot on ${path}`);
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
        );
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
        });
        
        oldEditsWithIdx.sort((edit1, edit2) => edit1.edit.line - edit2.edit.line);	// sort in starting line order

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
                "line": fromLine,
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
				oldEditsWithIdx[oldEditIdx].edit.line + oldEditsWithIdx[oldEditIdx].edit.rmLine < fromLine
            ) {
                ++oldEditIdx;
                // oldAdjustedEditsWithIdx.push(oldEditsWithIdx[oldEditIdx]);
            }

            // if current edit is overlapped/adjoined with this diff
			if (
				oldEditIdx < oldEditsWithIdx.length &&
				oldEditsWithIdx[oldEditIdx].edit.line <= toLine
			) {
                // replace the all the overlapped/adjoined old edits with the new edit
                const fromIdx = oldEditIdx;
                while (
                    oldEditIdx < oldEditsWithIdx.length &&
                    oldEditsWithIdx[oldEditIdx].edit.line <= toLine
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

            // now lastLine represents after-edit snapshot line number
			if (!(diff.removed)) {
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
				const s = edit.line - 1;  // zero-based starting line
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
				.sort((edit1, edit2) => edit1.line - edit2.line);
				
			// execute removed edits
			const removedEditsOnPath = editsOnPath.filter((edit) => removedEdits.has(edit));
			this.textBaseSnapshots.set(filePath, performEdits(doc, removedEditsOnPath));
			
			// rebase other edits in file
			let offsetLines = 0;
			for (let edit of editsOnPath) {
				if (removedEdits.has(edit)) {
					offsetLines = offsetLines - edit.rmLine + edit.addLine;
				} else {
					edit.line += offsetLines;
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
    async getSimpleEditList() {
        return this.editList.map((edit) => ({
			"beforeEdit": edit.rmText?.trim() ?? "",
            "afterEdit": edit.addText?.trim() ?? "",
            // "codeAbove": edit.codeAbove.trim(),
            // "codeBelow": edit.codeBelow.trim()
        }));
    }

    async getEditList() {
        return this.editList;
    }

    async updateEditList() {
        const openedDocuments = getOpenedFilePaths();
        const docGetter = async (filePath: string) => {
            return await getStagedFile(openedDocuments, filePath);
        };
        await this.updateAllSnapshotsFromDocument(docGetter);
    }

    async getUpdatedSimpleEditList() {
        await this.updateEditList();
        return await this.getSimpleEditList();
    }

    async getUpdatedEditList() {
        await this.updateEditList();
        return await this.getEditList();
    }
}

export const globalEditDetector = new EditDetector();

export function updateEditorState(editor: vscode.TextEditor | undefined) {
    if (!editor) globalEditorState.inDiffEditor = true;
    else globalEditorState.inDiffEditor = (vscode.window.tabGroups.activeTabGroup.activeTab?.input instanceof vscode.TabInputTextDiff);
    globalEditorState.language = vscode.window.activeTextEditor?.document?.languageId.toLowerCase() ?? "unknown";
    statusBarItem.setStatusDefault(true);

    // update file snapshot
    const currUri = editor?.document?.uri;
    const currPath = currUri?.fsPath;
    if (currUri && currUri.scheme === "file" && currPath && !(globalEditDetector.hasSnapshot(currPath))) {
        globalEditDetector.addSnapshot(currPath, editor.document.getText());
    }

    let isEditDiff = false;
    if (globalEditorState.inDiffEditor) {
        const input = vscode.window.tabGroups.activeTabGroup?.activeTab?.input as any;
        isEditDiff = ((input instanceof vscode.TabInputTextDiff)
            && input.original.scheme === 'temp'
            && input.modified.scheme === 'file') || (input.textDiffs ? true : false);
    }

    console.log(`toPredictLocation: ${globalEditorState.toPredictLocation}`);
    console.log(`length: ${globalQueryContext.getLocations()?.length}`);
    if (vscode.workspace.getConfiguration("coEdPilot").get("predictLocationOnEditAcception") && globalEditorState.toPredictLocation && !(globalQueryContext.getLocations()?.length)) {
        vscode.commands.executeCommand("coEdPilot.predictLocations");
        globalEditorState.toPredictLocation = false;
    }
    vscode.commands.executeCommand('setContext', 'coEdPilot:isEditDiff', isEditDiff);
    vscode.commands.executeCommand('setContext', 'coEdPilot:isLanguageSupported', globalEditorState.isActiveEditorLanguageSupported());
}

export class FileStateMonitor extends DisposableComponent {
    constructor() {
        super();
        this.register(
            vscode.window.onDidChangeActiveTextEditor(updateEditorState)
        );
    }
}