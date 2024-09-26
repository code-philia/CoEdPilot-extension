import vscode from 'vscode';
import { toPosixPath } from '../utils/file-utils';
import { globalEditorState } from '../global-workspace-context';
import { limitNum } from "../utils/utils";
import path from 'path';
import { BackendApiEditLocation } from '../utils/base-types';
import { liveTextEditorEventHandler } from '../utils/vscode-utils';

const replaceBackgroundColor = 'rgba(255,0,0,0.3)';
const addBackgroundColor = 'rgba(0,255,0,0.3)';
const replaceIconPath = path.join(__dirname, '../media/edit-red.svg');
const addIconPath = path.join(__dirname, '../media/add-green.svg');

export class LocationResultDecoration {
	private replaceDecorationType: vscode.TextEditorDecorationType;
	private addDecorationType: vscode.TextEditorDecorationType;
	private eventHandler: liveTextEditorEventHandler;
	private locations: BackendApiEditLocation[];

	constructor(locations: BackendApiEditLocation[]) {
		this.locations = locations;

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

		this.eventHandler = new liveTextEditorEventHandler(
			this.refreshLocationDecorations,
			this.clearDecorations,
			this
		);
	}

	show() {
		this.eventHandler.handle(vscode.window.activeTextEditor);
	}

	refreshLocationDecorations(editor?: vscode.TextEditor) {
		if (!editor || !this.locations) return;
		if (globalEditorState.inDiffEditor) return;

		const uri = editor?.document?.uri;
		if (!uri) return;

		const filePath = toPosixPath(uri.fsPath);
		if (uri.scheme !== 'file') return undefined;

		const decorationRangesForAlter: vscode.DecorationOptions[] = [];
		const decorationRangesForAdd: vscode.DecorationOptions[] = [];
	
		this.locations
			.filter((loc) => loc.targetFilePath === filePath)
			.forEach((loc) => {
				let startLine = loc.atLines[0];
				let endLine = loc.atLines[loc.atLines.length - 1];
				if (loc.editType === "add") {	// the model was designed to add content after the mark line
					startLine += 1;
					endLine += 1;
				}

				const document = editor.document;
				startLine = limitNum(startLine, 0, document.lineCount - 1);
				endLine = limitNum(endLine, 0, document.lineCount - 1);
				
				const startPos = editor.document.lineAt(startLine).range.start;
				const endPos = editor.document.lineAt(endLine).range.end;
				const range = new vscode.Range(startPos, endPos);
		
				// Create decoration
				const decoration = {
					range: range
				};
		
				// Add decoration to array
				if (loc.editType === 'add') {
					decorationRangesForAdd.push(decoration);
				} else {
					decorationRangesForAlter.push(decoration);
				}

			});
		editor.setDecorations(this.replaceDecorationType, decorationRangesForAlter);
		editor.setDecorations(this.addDecorationType, decorationRangesForAdd);			
	}

	clearDecorations(editor?: vscode.TextEditor) {
		this.locations = [];
		this.refreshLocationDecorations(editor);
	}

	dispose() {
		this.eventHandler.dispose();
	}
}
