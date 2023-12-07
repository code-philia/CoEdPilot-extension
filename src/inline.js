import vscode from 'vscode';
import { toPosixPath } from './file';
import { fileState } from './context';
import { queryState } from './context';
import { predictEditAtRange } from './query-tasks';
import { EditSelector, tempWrite } from './compare-view';
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
		if (fileState.inDiffEditor) return;

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

class InlineFixProvider extends BaseComponent {
	constructor() {
		super();
		this.register(
			vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, this),
		);
	}

	async provideCodeActions(document, range) {
		const currFile = toPosixPath(document?.fileName);
		const newEdits = await predictEditAtRange(document, range);

		if (!newEdits || newEdits.targetFilePath != currFile)
			return [];
		
		const diagnosticRange = new vscode.Range(document.positionAt(newEdits.startPos), document.positionAt(newEdits.endPos));

		const codeActions = newEdits.replacement.map(replacement => {
			// Create a diagnostic
			const diagnostic = new vscode.Diagnostic(diagnosticRange, 'Replace with: ' + replacement, vscode.DiagnosticSeverity.Hint);
			diagnostic.code = 'replaceCode';

			// Create a QuickFix
			const codeAction = new vscode.CodeAction(replacement, vscode.CodeActionKind.QuickFix);
			codeAction.diagnostics = [diagnostic];
			codeAction.isPreferred = true;

			// Create WorkspaceEdit
			const edit = new vscode.WorkspaceEdit();
			const replaceRange = new vscode.Range(document.positionAt(newEdits.startPos), document.positionAt(newEdits.endPos));
			edit.replace(document.uri, replaceRange, replacement);
			codeAction.edit = edit;

			codeAction.command = {
				command: 'editPilot.applyFix',
				title: '',
				arguments: [],
			};

			return codeAction;
		})

		const selector = new EditSelector(
			currFile,
			newEdits.startPos,
			newEdits.endPos,
			newEdits.replacement,
			tempWrite
		);
		await selector.init();
		await selector.editedDocumentAndShowDiff();

		return codeActions;	
	}
}

export {
	LocationDecoration,
	InlineFixProvider
};
