import vscode from "vscode";
import { PredictLocationCommand, GenerateEditCommand } from "./services/query-tasks";
import { limitNum } from "./utils/utils";
import { globalEditDetector } from "./editor-state-monitor";
import { FileEdits } from "./utils/base-types";
import { createVirtualModifiedFileUri } from "./views/compare-view";
// import { addUserStatItem } from "./global-context";

export function registerBasicCommands() {
	return vscode.Disposable.from(
		vscode.commands.registerCommand('coEdPilot.openFileAtLine', async (filePath, fromLine, toLine) => {
			const uri = vscode.Uri.file(filePath); // Replace with dynamic file path

			const document = await vscode.workspace.openTextDocument(uri);
			const editor = await vscode.window.showTextDocument(document);

			const isOverflowed = toLine > document.lineCount - 1;
			fromLine = limitNum(fromLine, 0, document.lineCount - 1);
			toLine = limitNum(toLine, 0, document.lineCount - 1);
			const range = new vscode.Range(
				document.lineAt(fromLine).range.start,
				isOverflowed ? document.lineAt(toLine).range.end : document.lineAt(toLine).range.start,
			);

			editor.selection = new vscode.Selection(range.start, range.end);
			editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
		}),
		vscode.commands.registerCommand('coEdPilot.clearPrevEdits', async () => {
			globalEditDetector.clearEditsAndSnapshots();
			await vscode.window.showInformationMessage("Previous edits cleared!");
			// addUserStatItem("clearPrevEdits");
		}),
		vscode.commands.registerCommand('coEdPilot.openRefactorPreview', openRefactorPreview)
	);
}

export function registerTopTaskCommands() {
	return vscode.Disposable.from(
		new PredictLocationCommand(),
		new GenerateEditCommand()
	);
}

// TODO move this to utils, and upgrade implementing this as a virtual text document, which can also be used when reverting an edit of rename
// Assume: if there is a line break at last, there should be a line of length 0 at the end
class FileOffsetCounter {
	readonly linesLength: number[];
	readonly linesAccumulativeLength: number[];

	constructor(text: string) {
		const linePattern = /.*?(\r\n?|\n|$)/g;  // empty new line at end when it ends with line break
		const lines = text.match(linePattern) ?? [];
		this.linesLength = [];
		this.linesAccumulativeLength = [];
		let ac = 0;
		for (const l of lines) {
			this.linesLength.push(l.length);
			this.linesAccumulativeLength.push(ac);
			ac += l.length;
		}
	}
	
	countFileOffset(pos: vscode.Position) {
		const ll = this.linesLength;
		const l = pos.line;
		const c = pos.character;
		if (l > ll.length) return undefined;
		if (c > ll[l]) return undefined;

		const lal = this.linesAccumulativeLength;
		return lal[l] + c;
	}
}

const internalOpenMultiDiffEditorCommand = '_workbench.openMultiDiffEditor';
interface OpenMultiFileDiffEditorOptions {
	title: string;
	multiDiffSourceUri?: vscode.Uri;
	resources?: { originalUri: vscode.Uri; modifiedUri: vscode.Uri }[];
}

export interface UniqueRefactorEditsSet {
	id: string,
	edits: FileEdits[]
} 

// TODO consider using primitive implementation of code action preview (this seems to have been hidden internally :<) (the command is visible, but the context of command is hidden)
// TODO add cache for the resolved file diffs and virtual files, so that we don't need to re-calculate and re-store them again
// TODO assume refactorEdits[0] is the edit that is done now, but we cannot guarantee, need check
async function openRefactorPreview(refactorEditSet: UniqueRefactorEditsSet) {
	// const rangeEditClassified: Map<string, FileEdits[]> = new Map();
	// for (const rangeEdit of refactorEdits) {
	// 	const editUriString = rangeEdit.location.uri.toString();
	// 	const matchedClass = rangeEditClassified.get(editUriString);
	// 	if (matchedClass) {
	// 		matchedClass.push(rangeEdit);
	// 	} else {
	// 		rangeEditClassified.set(editUriString, [rangeEdit]);
	// 	}
	// }
	
	const changes: [vscode.Uri, vscode.Uri][] = [];
	for (const [originalFileUri, rangeEdits] of refactorEditSet.edits) {
		// const originalContent = readFileSync(originalFileUri.fsPath, { encoding: 'utf-8' });
		const originalContent = (await vscode.workspace.openTextDocument(originalFileUri)).getText();

		const replacements: [number, number, string][] = [];
		const offsetCounter = new FileOffsetCounter(originalContent);
		for (const edit of rangeEdits) {
			const s = offsetCounter.countFileOffset(edit.range.start);
			const e = offsetCounter.countFileOffset(edit.range.end);
			const text = edit.newText;
			if (s && e) {
				replacements.push([s, e, text]);
			}
		}
		
		let modifiedContent = '';
		let lastStop = 0;
		let textEnd = originalContent.length;
		// do all replacements at a time, preventing offset problems
		for (const [start, end, repl] of replacements) {
			modifiedContent += originalContent.slice(lastStop, start) + repl;
			lastStop = end;
		}
		modifiedContent += originalContent.slice(lastStop, textEnd);
		
		const modifiedProxyFileUri = await createVirtualModifiedFileUri(originalFileUri, modifiedContent);
		changes.push([originalFileUri, modifiedProxyFileUri]);
	}
	const options: OpenMultiFileDiffEditorOptions = {
		title: 'Preview',
		multiDiffSourceUri: vscode.Uri.parse(`temp-id:/${refactorEditSet.id}`), // just used as a unique identifier
		resources: changes.map(([originalUri, modifiedUri]) => ({ originalUri, modifiedUri }))
	};
	vscode.commands.executeCommand(internalOpenMultiDiffEditorCommand, options);
}
