import vscode from 'vscode';
import { toPosixPath } from './file';
import { editorState } from './global-context';
import { queryState } from './global-context';
import { BaseComponent, numIn } from './base-component';
import { editLocationView } from './activity-bar';
import path from 'path';

const replaceBackgroundColor = 'rgba(255,0,0,0.3)';
const addBackgroundColor = 'rgba(0,255,0,0.3)';
const replaceIconPath = path.join(__dirname, '../media/edit-red.svg');
const addIconPath = path.join(__dirname, '../media/add-green.svg');

class LocationDecoration extends BaseComponent {
	constructor() {
		super();
		this.replaceDecorationType = vscode.window.createTextEditorDecorationType({
			backgroundColor: replaceBackgroundColor,
			isWholeLine: true,
			rangeBehavior: vscode.DecorationRangeBehavior.ClosedOpen,
			gutterIconPath: replaceIconPath,
			gutterIconSize: "75%"
		});
		this.addDecorationType = vscode.window.createTextEditorDecorationType({
			backgroundColor: addBackgroundColor,
			isWholeLine: true,
			rangeBehavior: vscode.DecorationRangeBehavior.ClosedOpen,
			gutterIconPath: addIconPath,
			gutterIconSize: "75%"
		});
		
		this.disposable = vscode.Disposable.from(
			vscode.window.onDidChangeActiveTextEditor(this.setLocationDecorations, this),
			queryState.onDidChangeLocations(() => this.setLocationDecorations(vscode.window.activeTextEditor), this),
			editLocationView.treeView.onDidChangeSelection(() => this.setLocationDecorations(vscode.window.activeTextEditor), this)
		);
	}

	setLocationDecorations(editor) {
		if (editorState.inDiffEditor) return;

		const uri = editor?.document?.uri;
		if (!uri) return;

		const filePath = toPosixPath(uri.fsPath);
		if (uri.scheme !== 'file') return undefined;

		const decorationsForAlter = [];
		const decorationsForAdd = [];
	
		queryState.locations
			.filter((loc) => loc.targetFilePath === filePath)
			.map((loc) => {
				let startLine = loc.atLines[0];
				let endLine = loc.atLines[loc.atLines.length - 1];
				if (loc.editType === "add") {	// the model was designed to add content after the mark line
					startLine += 1;
					endLine += 1;
				}

				const document = editor.document;
				startLine = numIn(startLine, 0, document.lineCount - 1);
				endLine = numIn(endLine, 0, document.lineCount - 1);
				
				const startPos = editor.document.lineAt(startLine).range.start;
				const endPos = editor.document.lineAt(endLine).range.end;
				const range = new vscode.Range(startPos, endPos);
		
				// Create decoration
				const decoration = {
					range: range
				};
		
				// Add decoration to array
				if (loc.editType == 'add') {
					decorationsForAdd.push(decoration);
				} else {
					decorationsForAlter.push(decoration);
				}

			});
		editor.setDecorations(this.replaceDecorationType, decorationsForAlter);
		editor.setDecorations(this.addDecorationType, decorationsForAdd);			
	}
}

export {
	LocationDecoration,
};
