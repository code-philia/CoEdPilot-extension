import vscode from 'vscode';
import { toPosixPath } from '../utils/file-utils';
import { editorState } from '../global-context';
import { queryState } from '../global-context';
import { BaseComponent, numIn } from '../utils/base-component';
import { editLocationView } from '../views/location-tree-view';
import path from 'path';
import { NativeEditLocation } from '../utils/base-types';

const replaceBackgroundColor = 'rgba(255,0,0,0.3)';
const addBackgroundColor = 'rgba(0,255,0,0.3)';
const replaceIconPath = path.join(__dirname, '../media/edit-red.svg');
const addIconPath = path.join(__dirname, '../media/add-green.svg');

class LocationDecoration extends BaseComponent {
	replaceDecorationType: vscode.TextEditorDecorationType;
	addDecorationType: vscode.TextEditorDecorationType;

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
		
		this.register(
			vscode.window.onDidChangeActiveTextEditor(
				(editor) => this.setLocationDecorations(editor, queryState.locations),
				this
			),
			queryState.onDidChangeLocations(
				() => this.setLocationDecorations(vscode.window.activeTextEditor, queryState.locations),
				this
			),
			editLocationView.treeView.onDidChangeSelection(
				() => this.setLocationDecorations(vscode.window.activeTextEditor, queryState.locations),
				this
			)
		);
	}

	setLocationDecorations(editor?: vscode.TextEditor, locations?: NativeEditLocation[]) {
		if (!editor || !locations) return;
		if (editorState.inDiffEditor) return;

		const uri = editor?.document?.uri;
		if (!uri) return;

		const filePath = toPosixPath(uri.fsPath);
		if (uri.scheme !== 'file') return undefined;

		const decorationsForAlter: vscode.DecorationOptions[] = [];
		const decorationsForAdd: vscode.DecorationOptions[] = [];
	
		locations
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
