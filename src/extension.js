const vscode = require('vscode');
const { globalTempFileProvider, globalDiffTabSelectors } = require('./comp-view')
const { initFileState } = require('./file');
const { LocationTreeProvider } = require('./activity-bar');
const { queryState } = require('./query');
const { predictLocationAtEdit, predictAfterQuickFix } = require('./task');
const { InlineFixProvider, LocationDecoration } = require('./inline');


function activate(context) {
	vscode.workspace.registerFileSystemProvider("temp", globalTempFileProvider, { isReadonly: true });

	function registerDisposables(...disposables) {
		context.subscriptions.push(...disposables);
	}

	function registerCommand(command, callback) {
		context.subscriptions.push(
			vscode.commands.registerCommand(command, callback)
		);
	}

	initFileState(vscode.window.activeTextEditor);

	/*----------------------- Monitor edit behavior --------------------------------*/

	registerDisposables(
		vscode.window.onDidChangeActiveTextEditor(initFileState),      	// Register an event listener that triggers when the editor is switched and initializes global variables
		vscode.window.onDidChangeTextEditorSelection(predictLocationAtEdit)			// Register an event listener that listens for changes in cursor position
	);

	/*----------------------- Provide QuickFix feature -----------------------------*/
	// Register CodeAction Provider to provide QuickFix for the modification position returned by the Python script
	registerDisposables(
		new LocationDecoration(),
		new InlineFixProvider()
	);

	registerCommand('extension.applyFix', async () => {
		console.log('==> applyFix');
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			queryState.clearLocations();
			await predictAfterQuickFix();
		}
	});

	/*----------------------- Edit description input box --------------------------------*/
	const inputBox = vscode.window.createInputBox();
	inputBox.prompt = 'Enter edit description';
	inputBox.ignoreFocusOut = true; // The input box will not be hidden after losing focus

	registerCommand('extension.inputMessage', async function () {
		console.log('==> Edit description input box is displayed')
		inputBox.show();
	});

	registerDisposables(
		inputBox.onDidAccept(() => { // After the user presses Enter to confirm the commit message, recommend the edit range
			const userInput = inputBox.value;
			console.log('==> Edit description:', userInput);
			queryState.commitMessage = userInput;
			inputBox.hide();
		})
	);

	registerCommand('editPilot.openFileAtLine', async (filePath, lineNum) => {
		const uri = vscode.Uri.file(filePath); // Replace with dynamic file path

		try {
			const document = await vscode.workspace.openTextDocument(uri);
			const editor = await vscode.window.showTextDocument(document);
			const range = editor.document.lineAt(lineNum).range;
			editor.selection = new vscode.Selection(range.start, range.end);
			editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
		} catch (error) {
			console.error(error);
		}
	});

	/*----------------------- Activity Bar Container for Edit Points --------------------------------*/
	registerDisposables(new LocationTreeProvider());

	registerCommand("editPilot.last-suggestion", () => {
		const currTab = vscode.window.tabGroups.activeTabGroup.activeTab;
		const selector = globalDiffTabSelectors[currTab];
		selector && selector.switchEdit(-1);
	});
	registerCommand("editPilot.next-suggestion", () => {
		const currTab = vscode.window.tabGroups.activeTabGroup.activeTab;
		const selector = globalDiffTabSelectors[currTab];
		selector && selector.switchEdit(1);
	});
	registerCommand("editPilot.accept-edit", () => { });
	registerCommand("editPilot.dismiss-edit", () => { });
	console.log('==> Congratulations, your extension is now active!');
}

function deactivate() {
}

module.exports = {
	activate,
	deactivate
}
