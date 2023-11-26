const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const glob = require('glob');
const diff = require('diff');

const { FileNodeProvider } = require('./activity-bar')
const { query_discriminator, query_locator, query_generator } = require('./model-client');
const { EditSelector, globalTempFileProvider, globalDiffTabSelectors } = require('./comp-view')

// ------------ Hyper-parameters ------------
const fgcolor1 = '#000';
const bgcolor1 = 'rgba(255,0,0,0.2)';
const fgcolor2 = '#000';
const bgcolor2 = 'rgba(0,255,0,0.2)';
const prevEditNum = 3;

// ------------ Extension States -------------
let modifications = [];
let prevCursorAtLine = 0;
let currCursorAtLine = 0;
let prevSnapshot = undefined;
let currSnapshot = undefined;
let currFile = undefined;
let commitMessage = "";
let prevEdits = [];
let editLock = false;

class editDetector {
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
		 * 			"addText": string or null, number of added text, could be null
		 * 		},
		 * 		...
		 * ]
		 */ 
		this.editList = [];
	}

	hasSnapshot(path) {
		return this.textBaseSnapshots.has(path);
	}

	updateSnapshot(path, text) {
		this.textBaseSnapshots[path] = text;
	}

	updateAllSnapshotsFromDocument() {
		for (const [path, ] of this.textBaseSnapshots) {
			try {
				const text = fs.readFileSync(path, 'utf-8');	// won't throw error on non-utf-8 character here
				this.updateEdits(path, text);
			} catch (err) {
				console.log('Cannot update snapshot on ${path}$')
			} 
		}
	}

	updateEdits(path, text) {
		if (this.textBaseSnapshots.has(path)) {
			return;
		}

		// Compare old `editList` with new diff on a document
		// All new diffs should be added to edit list, but merge some of them to the old ones
		// Merge "-" (removed) diff into an overlapped old edit
		// Merge "+" (added) diff into an old edit only if its precedented "-" hunk (a zero-line "-" hunk if there's no) wraps the old edit's "-" hunk
		// By default, there could only be zero or one "+" hunk following a "-" hunk
		const newDiffs = diff.diffTrimmedLines(
			this.textBaseSnapshots[path],
			text
		)
		const oldEditsWithIdx = this.editList.
			filter((edit) => edit.path === path)
			.map((edit, idx) => {
				return {
					idx: idx,
					edit: edit
				}
			});		// keep index order (aka. time order)
		
		oldEditsWithIdx.sort((edit1, edit2) => edit1.edit.s - edit2.edit.s);	// sort in starting line order
		
		const oldAdjustedEditsWithIdx = [];
		const newEdits = [];
		let lastLine = 1;
		let oldEditIdx = 0;

		function mergeDiff(rmDiff, addDiff) {
			const fromLine = lastLine;
			const toLine = lastLine + (rmDiff.count ?? 0);
			
			// construct new edit
			const newEdit = {
				"path": path,
				"s": fromLine,
				"rmLine": rmDiff.count ?? 0,
				"rmText": rmDiff.value ?? null,
				"addLine": addDiff.count ?? 0,
				"addText": addDiff.count ?? null,
			};

			// skip all old edits between this diff and the last diff
			while (
				oldEditIdx < oldEditsWithIdx.length &&
				oldEditsWithIdx[oldEditIdx].edit.s +
				oldEditsWithIdx[oldEditIdx].edit.rmLine <= lastLine
			) {
				++oldEditIdx;
				oldAdjustedEditsWithIdx.push(oldEditsWithIdx[oldEditIdx]);
			}
			
			// if current edit is overlapped with this diff
			if (oldEditsWithIdx[oldEditIdx].edit.s < toLine) {
				// replace the all the overlapped old edits with the new edit
				const fromIdx = oldEditIdx;
				while (
					oldEditIdx < oldEditsWithIdx.length &&
					oldEditsWithIdx[oldEditIdx].edit.s +
					oldEditsWithIdx[oldEditIdx].edit.rmLine <= toLine
					) {
						++oldEditIdx;
					}
				// use the maximum index of the overlapped old edits	---------->  Is it necessary?
				const minIdx = Math.max.apply(
					null,
					oldEditsWithIdx.slice(fromIdx, oldEditIdx).map((edit) => edit.idx)
				);
				oldAdjustedEditsWithIdx.push({
					idx: minIdx,
					edit: newEdit
				})
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
					mergeDiff(diff, null);
				}
			} else if (diff.added) {
				// deal with a "+" diff not following a "-" diff
				mergeDiff(null, diff);
			}

			lastLine += diff.count;
		}
	}

	// Shift editList if out of capacity
	// For every overflown edit, apply it and update the document snapshots on which the edits base
	shiftEdits() {
		// filter all removed edits
		const numRemovedEdits = this.editList.length - this.editLimit;
		if (numRemovedEdits <= 0){
			return;
		}
		const removedEdits = this.editList.slice(
			0,
			numRemovedEdits
		);

		// for each file involved in the removed edits
		// rebase other edits in file
		const affectedPaths = new Set(
			removedEdits.map((edit) => edit.path)
		);
		for (const path of affectedPaths) {
			const editsOnPath = this.editList
				.filter((edit) => edit.path === path)
				.sort((edit1, edit2) => edit1.s - edit2.s);
			
			let offsetLines = 0;
			for (let edit of editsOnPath) {
				if (edit in removedEdits) {
					offsetLines = offsetLines - edit.rmLine + edit.addLine;
				} else {
					edit.s += offsetLines;
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
	getEditListText() {
		return this.editList.map((edit) => ({
			"beforeEdit": edit.rmText ?? "",
			"afterEdit": edit.addText ?? ""
		}))
	}
}

const globalEditDetector = editDetector;

// Convert Windows-style path to POSIX-style path
function toPosixPath(path) {
	return path.replace(/\\/g, '/');
}

function getActiveFile(editor) {
	return toPosixPath(editor.document.fileName);
}

function refreshEditorState(editor) {
	prevCursorAtLine = 0;
	currCursorAtLine = 0;
	prevSnapshot = editor.document.getText();
	currSnapshot = undefined;
	currFile = getActiveFile(editor);
	console.log('==> Active File:', currFile);
	console.log('==> Global variables initialized');
	highlightModifications(modifications, editor);
}

const decorationStyleForAlter = vscode.window.createTextEditorDecorationType({
	color: fgcolor1,
	backgroundColor: bgcolor1
});

const decorationStyleForAdd = vscode.window.createTextEditorDecorationType({
	color: fgcolor2,
	backgroundColor: bgcolor2
});

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
	editor.setDecorations(decorationStyleForAlter, decorationsForAlter);
	editor.setDecorations(decorationStyleForAdd, decorationsForAdd);
	decorationStyleForAlter.dispose();
	decorationStyleForAdd.dispose();
}

// glob files with specific patterns
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

function replaceCurrentSnapshot(fileList) {
	for (let file of fileList) {
		if (file[0] === currFile) {
			file[1] = currSnapshot; // Use the unsaved content as the actual file content
			break;
		}
	}
}

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

function showModsInActivityBar(modificationList) {
	vscode.commands.executeCommand('editPilot.refreshEditPoints', modificationList);
}

async function predictLocation(rootPath, files, prevEdits) {
	/* 
		Discriminator:
		input:
		{
			"rootPath": str, rootPath,
			"files": list, [[filePath, fileContent], ...],
			"targetFilePath": str, filePath
		}
		output:
		{
			"data": list, [[filePath, fileContent], ...]
		}
	
		Locator:
		input:
		{
			"files": list, [[filePath, fileContent], ...],
			"targetFilePath": str, filePath,
			"commitMessage": str, commit message,
			"prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}
		}
		output:
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
		path.relative(rootPath, getActiveFile(vscode.window.activeTextEditor))
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
		const filteredFiles = files.filter(([filename, _]) => discriminatorOutput.data.includes(filename) || filename == activeFilePath);

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
		highlightModifications(modifications, vscode.window.activeTextEditor);
		// showModificationWebview(modifications);
		showModsInActivityBar(modifications);
	} catch (error) {
		console.error(error);
	}
}

async function predictEdit(modification) {
	/* 	
		Generator:
		input:
		{ 
			"files": list, [[filePath, fileContent], ...],
			"targetFilePath": string filePath,
			"commitMessage": string, commit message,
			"editType": string, edit type,
			"prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
			"startPos": int, start position,
			"endPos": int, end position,
			"atLine": list, of edit line indices
		}
		output:
		{
			"data": 
			{ 
				"targetFilePath": string, filePath of target file,
				"editType": string, 'remove', 'add'
				"startPos": int, start position,
				"endPos": int, end position,
				"replacement": list of strings, replacement content   
			}
		} 
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

	const output = await query_generator(input);
	let newmodification = output.data;
	console.log('==> Edit generator model returned successfully');
	// Highlight the location of the modification
	highlightModifications([newmodification], editor);
	return newmodification; // Return newmodification
}

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

function pushEdit(item) {
	prevEdits.push(item);

	if (prevEdits.length > prevEditNum) {
		prevEdits.shift(); // FIFO pop the earliest element
	}
}

//	Try updating prevEdits
// 	If there's new edits return prevEdits
//  Else return null
function getPrevEdits(event) {
	const line = event.selections[0].active.line;
	currCursorAtLine = line + 1; // VScode API starts counting lines from 0, while our line numbers start from 1, note the +- 1
	console.log(`==> Cursor position: Line ${prevCursorAtLine} -> ${currCursorAtLine}`);
	currSnapshot = vscode.window.activeTextEditor.document.getText(); // Read the current text in the editor
	if (prevCursorAtLine != currCursorAtLine && prevCursorAtLine != 0) { // When the pointer changes position and is not at the first position in the editor
		let edition = detectEdit(prevSnapshot, currSnapshot); // Detect changes compared to the previous snapshot

		if (edition.beforeEdit != edition.afterEdit) {
			// Add the modification to prevEdit
			pushEdit(edition);
			console.log('==> Before edit:\n', edition.beforeEdit);
			console.log('==> After edit:\n', edition.afterEdit);
			prevSnapshot = currSnapshot;		
			return prevEdits;
		}
		prevCursorAtLine = currCursorAtLine; // Update the line number where the mouse pointer is located
		return null;
	}
	return null;
}

async function handleEditEvent(event) {
	if (editLock) return;
	editLock = true;
	try {
		console.log('==> Send to LLM (After cursor changed line)');
		const [rootPath, files] = getRootAndFiles();
		const currentPrevEdits = getPrevEdits(event);
		if (currentPrevEdits) {
			await predictLocation(rootPath, files, currentPrevEdits);
		}
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
			pushEdit(edition);
			console.log('==> Before edit:\n', edition.beforeEdit);
			console.log('==> After edit:\n', edition.afterEdit);
			prevSnapshot = currSnapshot;
			console.log('==> Send to LLM (After apply QucikFix)');
			let [rootPath, files] = getRootAndFiles();
			await predictLocation(rootPath, files, prevEdits);
		}
		prevCursorAtLine = currCursorAtLine; // Update the line number where the mouse pointer is located
	} finally {
		editLock = false;
	}
}

function getModAtRange(mods, document, range) {
	const filePath = toPosixPath(document.uri.fsPath);
	const startPos = document.offsetAt(range.start);
	const endPos = document.offsetAt(range.end);
	return mods.find((mod) => {
		if (filePath == mod.targetFilePath && mod.startPos <= startPos && endPos <= mod.endPos) {
			let highlightedRange = new vscode.Range(document.positionAt(mod.startPos), document.positionAt(mod.endPos));
			const currentToBeReplaced = document.getText(highlightedRange).trim();
			if (currentToBeReplaced == mod.toBeReplaced.trim()) {
				return true;
			} else {
				return false;
			}	
		}
	})
}

async function predictEditAtRange(document, range) {
	if (editLock == true)
		return;
	editLock = true;

	try {
		const targetMod = getModAtRange(modifications, document, range);
		if (targetMod) {
			const replacedRange = new vscode.Range(document.positionAt(targetMod.startPos), document.positionAt(targetMod.endPos));
			const replacedContent = document.getText(replacedRange).trim();
			const predictResult = await predictEdit(targetMod);
			predictResult.replacement = predictResult.replacement.filter((snippet) => snippet.trim() !== replacedContent);
			return predictResult;
		} else {
			return;
		}
	} finally {
		editLock = false;
	}
}

function clearUpModsAndHighlights(editor) {
	modifications = [];
	highlightModifications(modifications, editor);
}

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

	// When there is a default activeTextEditor opened in VSCode, automatically read the current text content as prevSnapshot
	prevSnapshot = vscode.window.activeTextEditor.document.getText();
	currSnapshot = vscode.window.activeTextEditor.document.getText(); // Read the current text in the editor
	
	/*----------------------- Monitor edit behavior --------------------------------*/

	registerDisposables(
		vscode.window.onDidChangeActiveTextEditor(refreshEditorState),      	// Register an event listener that triggers when the editor is switched and initializes global variables
		vscode.window.onDidChangeTextEditorSelection(handleEditEvent)			// Register an event listener that listens for changes in cursor position
	);

	/*----------------------- Provide QuickFix feature -----------------------------*/
	// Register CodeAction Provider to provide QuickFix for the modification position returned by the Python script
	registerDisposables(
		vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, {
			async provideCodeActions(document, range) {
				const newmodification = await predictEditAtRange(document, range);

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

					codeAction.command = {
						command: 'extension.applyFix',
						title: '',
						arguments: [],
					};

					return codeAction;
				})

				const selector = new EditSelector(currFile, newmodification.startPos, newmodification.endPos, newmodification.replacement);
				await selector.init();
				await selector.editedDocumentAndShowDiff();

				return codeActions;
			}
		})
	);

	registerCommand('extension.applyFix', async () => {
		console.log('==> applyFix');
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			clearUpModsAndHighlights(editor);
			await predictAfterQuickFix(editor.document.getText());
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
		inputBox.onDidAccept(async () => { // After the user presses Enter to confirm the commit message, recommend the edit range
			if (editLock) return;
			editLock = true;
			const userInput = inputBox.value;
			console.log('==> Edit description:', userInput);
			commitMessage = userInput;
			let [rootPath, files] = getRootAndFiles();
			inputBox.hide();
			try {
				await predictLocation(rootPath, files, prevEdits);
			} finally {
				editLock = false;
			}
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
	const fileNodeProvider = new FileNodeProvider();
	registerDisposables(
		vscode.window.registerTreeDataProvider('editPoints', fileNodeProvider)
	);

	registerCommand('editPilot.refreshEditPoints', modList => fileNodeProvider.refresh(modList));

	registerCommand("edit-pilot.last-suggestion", () => {
		const currTab = vscode.window.tabGroups.activeTabGroup.activeTab;
		const selector = globalDiffTabSelectors[currTab];
		selector && selector.switchEdit(-1);
	 });
	registerCommand("edit-pilot.next-suggestion", () => {
		const currTab = vscode.window.tabGroups.activeTabGroup.activeTab;
		const selector = globalDiffTabSelectors[currTab];
		selector && selector.switchEdit(1);
	 });
	registerCommand("edit-pilot.accept-edit", () => { });
	registerCommand("edit-pilot.dismiss-edit", () => { });
	console.log('==> Congratulations, your extension is now active!');
}

function deactivate() {
	// Clear the decorator
	decorationStyleForAlter.dispose();
	decorationStyleForAdd.dispose();
}

module.exports = {
	activate,
	deactivate
}
