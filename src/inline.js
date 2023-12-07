import vscode from 'vscode';
import { toPosixPath } from './file';
import { editorState } from './global-context';
import { queryState } from './global-context';
import { BaseComponent } from './base-component';

const replacementBackgroundColor = 'rgba(255,0,0,0.2)';
const additionBackgroundColor = 'rgba(0,255,0,0.2)';

class LocationDecoration extends BaseComponent {
	constructor() {
		super();
		this.replaceDecorationType = vscode.window.createTextEditorDecorationType({
			backgroundColor: replacementBackgroundColor,
			isWholeLine: true
		});
		this.addDecorationType = vscode.window.createTextEditorDecorationType({
			backgroundColor: additionBackgroundColor,
			isWholeLine: true
		});
		
		this.disposable = vscode.Disposable.from(
			vscode.window.onDidChangeActiveTextEditor(this.setLocationDecorations, this),
			queryState.onDidQuery(() => this.setLocationDecorations(vscode.window.activeTextEditor), this)
		);
	}

	setLocationDecorations(editor) {
		if (editorState.inDiffEditor) return;

		const uri = editor?.document?.uri;
		if (!uri) return;

		const filePath = toPosixPath(uri.fsPath);
		console.log(queryState.locatedFilePaths);
		if (uri.scheme !== 'file' || !queryState.locatedFilePaths.includes(filePath)) return undefined;

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
				const startPos = editor.document.lineAt(startLine).range.start;
				const endPos = editor.document.lineAt(endLine).range.end;
				const range = new vscode.Range(startPos, endPos);
		
				// Create decoration
				const decoration = {
					range
				};
		
				// Add decoration to array
				if (loc.editType == 'add') {
					decorationsForAdd.push(decoration);
				} else {
					decorationsForAlter.push(decoration);
				}

				editor.setDecorations(this.replaceDecorationType, decorationsForAlter);
				editor.setDecorations(this.addDecorationType, decorationsForAdd);			
			})
	}
}

export {
	LocationDecoration,
};
