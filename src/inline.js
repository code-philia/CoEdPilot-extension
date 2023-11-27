const vscode = require('vscode');
const { toPosixPath, getActiveFile } = require('./file');
const { queryState } = require('./query');
const { predictEditAtRange } = require('./task');
const { EditSelector } = require('./comp-view');

const fgcolor1 = '#000';
const bgcolor1 = 'rgba(255,0,0,0.2)';
const fgcolor2 = '#000';
const bgcolor2 = 'rgba(0,255,0,0.2)';

const decorationStyleForAlter = vscode.window.createTextEditorDecorationType({
	color: fgcolor1,
	backgroundColor: bgcolor1
});

const decorationStyleForAdd = vscode.window.createTextEditorDecorationType({
	color: fgcolor2,
	backgroundColor: bgcolor2
});


function highlightLocations(locations, editor) {
	const decorationsForAlter = []; // Highlight for replace, delete
	const decorationsForAdd = []; // Highlight for add
	if (!editor) {
		return;
	}
	
	// Iterate through each modification
	locations
		.filter((loc) => loc.targetFilePath == toPosixPath(editor.document.uri.fsPath))
		.map((loc) => {
			const startPos = editor.document.positionAt(loc.startPos);
			const endPos = editor.document.positionAt(loc.endPos);
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
		})
	
	// Apply decorations to editor
	editor.setDecorations(decorationStyleForAlter, decorationsForAlter);
	editor.setDecorations(decorationStyleForAdd, decorationsForAdd);
	// decorationStyleForAlter.dispose();
	// decorationStyleForAdd.dispose();
}

function clearHighlights(editor) {
	highlightLocations([], editor);
}

class LocationDecoration {
	constructor() {
		this.replaceDecorationType = vscode.window.createTextEditorDecorationType({
			color: fgcolor1,
			backgroundColor: bgcolor1
		});
		this.addDecorationType = vscode.window.createTextEditorDecorationType({
			color: fgcolor2,
			backgroundColor: bgcolor2
		});
		
		this.disposable = vscode.Disposable.from(
			vscode.window.onDidChangeActiveTextEditor(this.setLocationDecorations, this),
			queryState.onDidQuery(this.setLocationDecorations, this)
		);
	}

	setLocationDecorations(editor) {
		if (!(editor?.document?.uri)) return undefined;

		const uri = editor.document.uri;
		const filePath = toPosixPath(uri.path);
		if (uri.scheme != 'file' || !queryState.locatedFilePaths.includes(filePath)) return undefined;

		const decorationsForAlter = [];
		const decorationsForAdd = [];
	
		queryState.locations
			.filter((loc) => loc.targetFilePath == filePath)
			.map((loc) => {
				const startPos = editor.document.positionAt(loc.startPos);
				const endPos = editor.document.positionAt(loc.endPos);
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

	dispose() {
		this.disposable.dispose();
	}
}

class InlineFixProvider {
	constructor() {
		this.disposable = vscode.Disposable.from(
			vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, this)
		);
	}

	dispose() {
		this.disposable.dispose;
	}

	async provideCodeActions(document, range) {
		const currFile = getActiveFile();
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
				command: 'extension.applyFix',
				title: '',
				arguments: [],
			};

			return codeAction;
		})

		const selector = new EditSelector(
			currFile,
			newEdits.startPos,
			newEdits.endPos,
			newEdits.replacement
		);
		await selector.init();
		await selector.editedDocumentAndShowDiff();

		return codeActions;
	}	

}

module.exports = {
    // highlightLocations,
	// clearHighlights,
	LocationDecoration,
	InlineFixProvider
}
