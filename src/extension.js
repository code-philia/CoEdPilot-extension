/**
 * Module dependencies
 */
const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const glob = require('glob');
const { getWebviewContent } = require('./CompPage');
const { FileNodeProvider } = require('./ActivityBar')

const { query_discriminator, query_locator, query_generator } = require('./ModelClient');
// const { createModifiedDocumentAndShowDiff } = require('./compare-view');

// ------------ Hyper-parameters ------------
let fgcolor1 = '#000';
let bgcolor1 = 'rgba(255,0,0,0.2)';
let fgcolor2 = '#000';
let bgcolor2 = 'rgba(0,255,0,0.2)';
let prevEditNum = 3;

// ------------ Global variants -------------
let modifications = [];
let prevCursorAtLine = 0;
let currCursorAtLine = 0;
let prevSnapshot = undefined;
let currSnapshot = undefined;
let currFile = undefined;
let previousActiveEditor = undefined;
let commitMessage = "";
let prevEdits = [];
let editLock = false;
let isDiffTabOpen = null;
/**
 * Convert Windows-style path to POSIX-style path
 * @param {string} path - Windows-style path
 * @returns {string} POSIX-style path
 */
function toPosixPath(path) {
	return path.replace(/\\/g, '/');
}

/**
 * Get the active file in the editor
 * @param {vscode.TextEditor} editor - The active editor
 * @returns {string} The active file path
 */
function getActiveFile(editor) {
	return toPosixPath(editor.document.fileName);
}

/**
 * Clean global variables when switching editor
 * @param {vscode.TextEditor} editor - The new active editor
 */
function cleanGlobalVariables(editor) {
	if (previousActiveEditor && previousActiveEditor.document.isDirty) {
		previousActiveEditor.document.save(); // Save the document content when switching activeEditor
	}
	previousActiveEditor = editor; // Update previousActiveEditor

	prevCursorAtLine = 0;
	currCursorAtLine = 0;
	prevSnapshot = editor.document.getText();
	currSnapshot = undefined;
	currFile = getActiveFile(editor);
	console.log('==> Active File:', currFile);
	console.log('==> Global variables initialized');
	highlightModifications(modifications, editor);
}

const decorationTypeForAlter = vscode.window.createTextEditorDecorationType({
	color: fgcolor1,
	backgroundColor: bgcolor1
});

const decorationTypeForAdd = vscode.window.createTextEditorDecorationType({
	color: fgcolor2,
	backgroundColor: bgcolor2
});

/**
 * Highlight modifications in the editor
 * @param {Array} modifications - The list of modifications to highlight
 * @param {vscode.TextEditor} editor - The active editor
 */
function highlightModifications(modifications, editor) {
	const decorationsForAlter = []; // Highlight for replace, delete
	const decorationsForAdd = []; // Highlight for add
	if (!editor) {
		return;
	}
	// Iterate through each modification
	for (const modification of modifications) {
		if (modification.targetFilePath != currFile) { // Only highlight modifications in the current file
			continue;
		}
		// Create decoration range
		const startPos = editor.document.positionAt(modification.startPos);
		const endPos = editor.document.positionAt(modification.endPos);
		const range = new vscode.Range(startPos, endPos);

		// Create decoration
		const decoration = {
			range
		};

		// Add decoration to array
		if (modification.editType == 'add') {
			decorationsForAdd.push(decoration);
		} else {
			decorationsForAlter.push(decoration);
		}
	}
	// Apply decorations to editor
	editor.setDecorations(decorationTypeForAlter, decorationsForAlter);
	editor.setDecorations(decorationTypeForAdd, decorationsForAdd);
}

/**
 * Get a list of files in the workspace
 * @param {string} rootPath - The root folder path of the workspace
 * @param {Array} globPatterns - The glob patterns to exclude files
 * @returns {Array} The list of files in the workspace
 */
function globFiles(rootPath, globPatterns) {
	// Built-in glob patterns
	const defaultGlobPatterns = [
		'!.git/**'
	];
	const allPatterns = defaultGlobPatterns.concat(globPatterns);

	// Concatenate glob patterns
	function concatRoot(pattern) {
		const concat_path = pattern[0] == '!' ? '!' + path.join(rootPath, pattern.slice(1)) : path.join(rootPath, pattern);
		return concat_path;
	}
	const allPatternsWithRoot = allPatterns.map(concatRoot).concat([path.join(rootPath, '**')]);

	const pathList = glob.sync('{' + allPatternsWithRoot.join(',') + '}', { windowsPathsNoEscape: true });
	return pathList;
}

/**
 * Replace the current snapshot with the unsaved content of the active file
 * @param {Array} fileList - The list of files in the workspace
 */
function replaceCurrentSnapshot(fileList) {
	for (let file of fileList) {
		if (file[0] === currFile) {
			file[1] = currSnapshot; // Use the unsaved content as the actual file content
			break;
		}
	}
}

/**
 * Get the root folder path and the list of files in the workspace
 * @param {boolean} useSnapshot - Whether to use the unsaved content of the active file as the actual file content
 * @returns {Array} The root folder path and the list of files in the workspace
 */
function getRootAndFiles(useSnapshot = true) {
	// Get the root folder path of the workspace
	const rootPath = toPosixPath(vscode.workspace.workspaceFolders[0].uri.fsPath);
	// Store the list of file contents and names
	const fileList = [];

	// Use glob to exclude certain files and return a list of all valid files
	const filePathList = globFiles(rootPath, []);

	function readFileFromPathList(filePathList, contentList) {
		for (const filePath of filePathList) {
			try {
				const stat = fs.statSync(filePath);
				if (stat.isFile()) {
					const fileContent = fs.readFileSync(filePath, 'utf-8');  // Skip files that cannot be correctly decoded
					contentList.push([filePath, fileContent]);
				}
			} catch (error) {
				console.log("Some error occurs when reading file");
			}
		}
	}

	readFileFromPathList(filePathList, fileList);
	// Replace directly when reading files, instead of replacing later
	if (useSnapshot) {
		replaceCurrentSnapshot(fileList);
	}

	return [rootPath, fileList];
}

/**
 * Show the modifications in a webview
 * @param {Array} modificationList - The list of modifications to show
 */
// function showModificationWebview(modificationList) {
// 	let panel = vscode.window.createWebviewPanel(
// 		"modificationWebview",
// 		"Modifications",
// 		// Use vscode.ViewColumn.One to avoid overlapping with the editor window
// 		vscode.ViewColumn.One,
// 		{ enableScripts: true }
// 	);
// 	const rootPath = vscode.workspace.rootPath;
// 	panel.webview.html = getWebviewContent(modificationList, rootPath);
// 	panel.webview.onDidReceiveMessage(message => {
// 		if (message.command === 'openFile') {
// 			const filePath = message.path;
// 			// Open the file
// 			vscode.workspace.openTextDocument(filePath).then(document => {
// 				vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One });
// 			});
// 		}
// 	});
// }

function showModsInActivityBar(modificationList) {
	vscode.commands.executeCommand('editPilot.refreshEditPoints', modificationList);
}

/**
 * Predict the location of the edit
 * @param {string} rootPath - The root folder path of the workspace
 * @param {Array} files - The list of files in the workspace
 * @param {Array} prevEdits - The list of previous edits
 * @param {vscode.TextEditor} editor - The active editor
 */
async function predictLocation(rootPath, files, prevEdits, editor) {
	/* 
		The input to the discriminator Python script is a dictionary in the following format:
		  {
			"rootPath": str, rootPath,
				"files": list, [[filePath, fileContent], ...],
				"targetFilePath": str, filePath
		}
		The output of the discriminator Python script is a dictionary in the following format:
		  {
			"data": list, [[filePath, fileContent], ...]
		}
	
		The input to the locator Python script is a dictionary in the following format:
		  {
			"files": list, [[filePath, fileContent], ...],
				"targetFilePath": str, filePath,
				"commitMessage": str, commit message,
				"prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}
		}
		The output of the locator Python script is a dictionary in the following format:
		  {
			"data": 
			[ 
				{ 
					"targetFilePath": str, filePath,
					"beforeEdit", str, the content before edit for previous edit,
					"afterEdit", str, the content after edit for previous edit, 
					"toBeReplaced": str, the content to be replaced, 
					"startPos": int, start position of the word,
					"endPos": int, end position of the word,
					"editType": str, the type of edit, add or remove,
					"lineBreak": str, '\n', '\r' or '\r\n'
				}, ...
			]
		}
	 */
	const activeFilePath = toPosixPath(
		path.relative(rootPath, getActiveFile(editor))
	);

	for (var single_file of files) {
		const filePath = single_file[0];
		const relPath = path.relative(rootPath, filePath);
		const relPathPosix = toPosixPath(relPath);
		single_file[0] = relPathPosix;
	}

	try {
		// Send to the discriminator model for analysis
		const disc_input = {
			rootPath: rootPath,
			files: files,
			targetFilePath: activeFilePath,
			commitMessage: commitMessage,
			prevEdits: prevEdits
		};
		console.log('==> Sending to discriminator model');
		const discriminatorOutput = await query_discriminator(disc_input);
		console.log('==> Discriminator model returned successfully');
		if (discriminatorOutput.data.length == 0) {
			console.log('==> No files will be analyzed');
			return;
		}
		console.log('==> Files to be analyzed:');
		discriminatorOutput.data.forEach(file => {
			console.log('\t*' + file);
		});
		console.log('==> Total no. of files:', files.length);
		console.log('==> No. of files to be analyzed:', discriminatorOutput.data.length);

		// Send the selected files to the locator model for location prediction
		const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename));

		console.log("==> Filtered files:")
		console.log(filteredFiles)

		const loc_input = {
			files: filteredFiles,
			targetFilePath: activeFilePath,
			commitMessage: commitMessage,
			prevEdits: prevEdits
		};
		console.log('==> Sending to edit locator model');
		const locatorOutput = await query_locator(loc_input);
		console.log('==> Edit locator model returned successfully');

		// Process the output of the locator Python script
		modifications = locatorOutput.data;
		if (modifications.length == 0) {
			console.log('==> No suggested edit location');
		}
		for (const mod of modifications) {
			mod.targetFilePath = toPosixPath(path.join(rootPath, mod.targetFilePath));
		}
		highlightModifications(modifications, editor);
		// showModificationWebview(modifications);
		showModsInActivityBar(modifications);
	} catch (error) {
		console.error(error);
	}
}

/**
 * Predict the edits to be made
 * @param {Object} modification - The modification to be made
 * @returns {Promise<Object>} The new modification to be made
 */
async function predictEdit(modification) {
	/*
	* The input to the Python script is a dictionary in the following format:
	* { 
	*   "files": list, [[filePath, fileContent], ...],
	*   "targetFilePath": string filePath,
	*   "commitMessage": string, commit message,
	*   "editType": string, edit type,
	*   "prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
	*   "startPos": int, start position,
	*   "endPos": int, end position,
	  *   "atLine": list, of edit line indices
	* }
	* The output of the Python script is a dictionary in the following format:
	* {"data": 
	*   { "targetFilePath": string, filePath of target file,
	*     "editType": string, 'remove', 'add'
	*     "startPos": int, start position,
	*     "endPos": int, end position,
	*     "replacement": list of strings, replacement content   
	*   }
	* }
	*/
	let files = getRootAndFiles()[1];
	const editor = vscode.window.activeTextEditor;

	const input = {
		files: files,
		targetFilePath: modification.targetFilePath,
		commitMessage: commitMessage,
		editType: modification.editType,
		prevEdits: modification.prevEdits,
		startPos: modification.startPos,
		endPos: modification.endPos,
		atLine: modification.atLine
	};
	const strJSON = JSON.stringify(input);

	const output = await query_generator(strJSON);
	let newmodification = output.data;
	console.log('==> Edit generator model returned successfully');
	// Highlight the location of the modification
	highlightModifications([newmodification], editor);
	return newmodification; // Return newmodification
}

/**
 * Detect the edit made by comparing the previous snapshot and the current snapshot
 * @param {string} prev - The previous snapshot
 * @param {string} curr - The current snapshot
 * @returns {Object} The line number where the edit was made
 */
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

/**
 * Add new detected edit to the list.
 * @param {Object} item 
 */
function addEdit(item) {
	prevEdits.push(item);

	if (prevEdits.length > prevEditNum) {
		prevEdits.shift(); // FIFO pop the earliest element
	}
}

async function handleEditEvent(event) {
	if (!event) {
		let [rootPath, files] = getRootAndFiles();
		await predictLocation(rootPath, files, prevEdits, vscode.window.activeTextEditor);
		vscode.workspace.onDidChangeTextDocument((e) => {
			const newUri = vscode.Uri.file("c:/Users/aaa/Desktop/hello.txt");
			if (e.document.uri.toString() === newUri.toString()) {
				e.document.save();
			} else {
				console.log(`${e.document.uri.toString()} != ${newUri.toString()}`)
			}
		});
		return;
	}
	if (editLock) return;
	editLock = true;
	try {

		const line = event.selections[0].active.line;
		currCursorAtLine = line + 1; // VScode API starts counting lines from 0, while our line numbers start from 1, note the +- 1
		console.log(`==> Cursor position: Line ${prevCursorAtLine} -> ${currCursorAtLine}`);
		currSnapshot = vscode.window.activeTextEditor.document.getText(); // Read the current text in the editor
		if (prevCursorAtLine != currCursorAtLine && prevCursorAtLine != 0) { // When the pointer changes position and is not at the first position in the editor
			let edition = detectEdit(prevSnapshot, currSnapshot); // Detect changes compared to the previous snapshot

			if (edition.beforeEdit != edition.afterEdit) {
				// Add the modification to prevEdit
				addEdit(edition);
				console.log('==> Before edit:\n', edition.beforeEdit);
				console.log('==> After edit:\n', edition.afterEdit);
				prevSnapshot = currSnapshot;
				console.log('==> Send to LLM (After cursor changed line)');
				let [rootPath, files] = getRootAndFiles();
				await predictLocation(rootPath, files, prevEdits, vscode.window.activeTextEditor);
			}
		}
		prevCursorAtLine = currCursorAtLine; // Update the line number where the mouse pointer is located
	} finally {
		editLock = false;
	}
}

// When the user adopts the suggestion of QuickFix, the modified version is sent to LLM to update modifications without waiting for the pointer to change lines
async function predictAfterQuickFix(text) {
	if (editLock) return;
	editLock = true;
	try {

		let edition = detectEdit(prevSnapshot, text); // Detect changes compared to the previous snapshot
		currSnapshot = text;
		if (edition.beforeEdit != edition.afterEdit) {
			// Add the modification to prevEdit
			addEdit(edition);
			console.log('==> Before edit:\n', edition.beforeEdit);
			console.log('==> After edit:\n', edition.afterEdit);
			prevSnapshot = currSnapshot;
			console.log('==> Send to LLM (After apply QucikFix)');
			let [rootPath, files] = getRootAndFiles();
			await predictLocation(rootPath, files, prevEdits, vscode.window.activeTextEditor);
		}
		prevCursorAtLine = currCursorAtLine; // Update the line number where the mouse pointer is located
	} finally {
		editLock = false;
	}
}


function startUpEditorEvent() {
	// Get the current file name when starting up
	let editor = vscode.window.activeTextEditor;
	if (editor) {
		// Get the currently active text editor
		const activeEditor = editor;
		var file = activeEditor.document.fileName;
		const activeFilePath = file;
		console.log('==> Active File:', activeFilePath);
		currFile = activeFilePath;
	}
}

async function getModification(doc, range) {
	if (editLock == true)
		return;
	editLock = true;
	try {
		const filePath = toPosixPath(doc.uri.fsPath);
		const startPos = doc.offsetAt(range.start);
		const endPos = doc.offsetAt(range.end);
		console.log(`===> selected offset: ${startPos} to ${endPos}`)
		for (let modification of modifications) {
			if (filePath == modification.targetFilePath && modification.startPos <= startPos && endPos <= modification.endPos) {
				let highlightedRange = new vscode.Range(doc.positionAt(modification.startPos), doc.positionAt(modification.endPos));
				const currentToBeReplaced = doc.getText(highlightedRange).trim();
				// console.log(`===> highlighted range: ${doc.getText(highlightedRange)}`);
				// console.log(`===> to be replaced: ${modification.toBeReplaced}`);
				if (currentToBeReplaced == modification.toBeReplaced) {
					// When the user does not follow the recommended modification, for example, the recommended modification is the word "good", but the user deletes "good", this will cause the content corresponding to the highlighted position to be offset, so there is no need to recommend modifying the content
					return await predictEdit(modification);
				} else {
					console.log('==> The suggested edit target:');
					console.log(doc.getText(highlightedRange));
					console.log('==> Current edit target:');
					console.log(modification.toBeReplaced);
					console.log('==> Highlighted range is not the suggested edit range');

					// At this time, clear modifications and cancel all highlights
					modifications = [];
					highlightModifications(modifications, vscode.window.activeTextEditor);
					return undefined;
				}
			}
		}
	} finally {
		editLock = false;
	}
}

// Clear modifications to avoid triggering runPythonScript2 while the pointer is still at the suggested modification position
// New suggested modification positions may take some time to highlight, during which time this modification position needs to avoid further highlighting
function clearUpModsAndHighlights(editor) {
	modifications = [];
	highlightModifications(modifications, editor);
}

/**
 * @param {vscode.ExtensionContext} context 
 */
function activate(context) {
	console.log('==> Congratulations, your extension is now active!');

	/*----------------------- Monitor edit behavior --------------------------------*/
	startUpEditorEvent(); // Get the currently active editor and the file name and path opened in the editor

	previousActiveEditor = vscode.window.activeTextEditor;
	// When there is a default activeTextEditor opened in VSCode, automatically read the current text content as prevSnapshot
	prevSnapshot = vscode.window.activeTextEditor.document.getText();
	currSnapshot = vscode.window.activeTextEditor.document.getText(); // Read the current text in the editor

	// DEBUGGING
	// handleEditEvent();

	context.subscriptions.push(
		vscode.window.onDidChangeActiveTextEditor(cleanGlobalVariables),      	// Register an event listener that triggers when the editor is switched and initializes global variables
		vscode.window.onDidChangeTextEditorSelection(handleEditEvent)			// Register an event listener that listens for changes in cursor position
	);

	/*----------------------- Provide QuickFix feature -----------------------------*/
	// Register CodeAction Provider to provide QuickFix for the modification position returned by the Python script
	const codeActionsProvider = vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, {
		async provideCodeActions(document, range) {
			const newmodification = await getModification(document, range);

			if (!newmodification || newmodification.targetFilePath != currFile)
				return [];

			const diagnosticRange = new vscode.Range(document.positionAt(newmodification.startPos), document.positionAt(newmodification.endPos));

			const codeActions = newmodification.replacement.map(replacement => {
				// Create a diagnostic
				const diagnostic = new vscode.Diagnostic(diagnosticRange, 'Replace with: ' + replacement, vscode.DiagnosticSeverity.Hint);
				diagnostic.code = 'replaceCode';

				// Create a QuickFix
				const codeAction = new vscode.CodeAction(replacement, vscode.CodeActionKind.QuickFix);
				codeAction.diagnostics = [diagnostic];
				codeAction.isPreferred = true;

				// Create WorkspaceEdit
				const edit = new vscode.WorkspaceEdit();
				const replaceRange = new vscode.Range(document.positionAt(newmodification.startPos), document.positionAt(newmodification.endPos));
				edit.replace(document.uri, replaceRange, replacement);
				codeAction.edit = edit;

				// if (! isDiffTabOpen) {
				// 	createModifiedDocumentAndShowDiff(document.uri);
				// 	isDiffTabOpen = true;
				// }

				codeAction.command = {
					command: 'extension.applyFix',
					title: '',
					arguments: [],
				};

				return codeAction;
			})

			return codeActions;
		}
	});

	const applyFixCmd = vscode.commands.registerCommand('extension.applyFix', async () => {
		console.log('==> applyFix');
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			clearUpModsAndHighlights(editor);
			await predictAfterQuickFix(editor.document.getText());
		}
	})

	context.subscriptions.push(
		codeActionsProvider,
		applyFixCmd
	);

	/*----------------------- Edit description input box --------------------------------*/
	const inputBox = vscode.window.createInputBox();
	inputBox.prompt = 'Enter edit description';
	inputBox.ignoreFocusOut = true; // The input box will not be hidden after losing focus

	const inputMsgCmd = vscode.commands.registerCommand('extension.inputMessage', async function () {
		console.log('==> Edit description input box is displayed')
		inputBox.show();
	});

	const inputBoxAcceptEvent = inputBox.onDidAccept(() => { // After the user presses Enter to confirm the commit message, recommend the edit range
		if (editLock) return;
		editLock = true;
		const userInput = inputBox.value;
		console.log('==> Edit description:', userInput);
		commitMessage = userInput;
		let [rootPath, files] = getRootAndFiles();
		predictLocation(rootPath, files, prevEdits, vscode.window.activeTextEditor).then(() => {
			editLock = false;
		}, () => {
			editLock = false;
		});
		inputBox.hide();
	});

	const openFileAtLineCmd = vscode.commands.registerCommand('editPilot.openFileAtLine', async (filePath, lineNum) => {
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

	context.subscriptions.push(
		inputMsgCmd,
		inputBoxAcceptEvent,
		openFileAtLineCmd
	);

	/*----------------------- Activity Bar Container for Edit Points --------------------------------*/
	const fileNodeProvider = new FileNodeProvider();
	const editPointsProv = vscode.window.registerTreeDataProvider('editPoints', fileNodeProvider);
	const refreshEditPointsCmd = vscode.commands.registerCommand('editPilot.refreshEditPoints', modList => fileNodeProvider.refresh(modList));

	context.subscriptions.push(
		editPointsProv,
		refreshEditPointsCmd
	)

	// const  vscode.window.onDidChangeVisibleTextEditors(editors => {
	// 	if (!editors.some(editor => editor.document.uri.scheme === 'diff')) {
	// 		isDiffTabOpen = false;
	// 	}
	// });

}

function deactivate() {
	// Clear the decorator
	decorationTypeForAlter.dispose();
	decorationTypeForAdd.dispose();
}

module.exports = {
	activate,
	deactivate
}
