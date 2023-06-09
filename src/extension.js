const vscode = require('vscode');
var exec = require('child_process').exec;
const fs = require('fs')
const highlight=require('./decorate');
const {Edit}=require('./edit');
const {handleRange,replaceRange}=require('./decorate');
const {getWebviewContent}=require('./sidebar')

/**
 * @param {vscode.ExtensionContext} context
 */

function getFileContent(activeEditor, selection = false) {
	if (selection)
		return activeEditor.document.getText(activeEditor.selection);
	else
		return activeEditor.document.getText(new vscode.Range(0, 0, activeEditor.document.lineCount, 0));
};

function arr1(arr) {
	return Array.from(new Set(arr))
}

var Poslist;

function getActiveEditor(callback) {
	activeEditor = vscode.window.activeTextEditor;
	if (!activeEditor) {
	  vscode.window.showErrorMessage("No editor opened.");
	  return;
	}
	callback(activeEditor);
}

function executeSecondCode(activeEditor) {
	let content = getFileContent(activeEditor, selection = true),fullcontent=getFileContent(activeEditor, selection = false);

	let model='F:\\CodeSuggestion-demo\\src\\model.py',target='F:\\CodeSuggestion-demo\\src';
	var editInstance =new Edit();
	var data;
	data={'code':fullcontent}
	fs.writeFileSync(target+'\\source.json', JSON.stringify(data, null, 2), 'utf8');
	exec('conda activate py310 && python'+' '+model+' '+target,function(err,stdout,stderr){
		if(err)
		{
			console.log('stderr',err);
			return;
		}
		if(stdout)
		{
			console.log(stdout);
			const jsonString = fs.readFileSync(target+'\\source.json', 'utf-8');
			const jsonData = JSON.parse(jsonString);
			editInstance.getDisplay(activeEditor,jsonData.data,fullcontent);
		}
	});
	return editInstance;
}

function activate(context) {

	console.log('Congratulations, your extension is now active!');

	let suggestion=vscode.commands.registerCommand('demo.code_suggestion',async function(){
		console.log('code suggestion!')
		getActiveEditor(function(editor) {
			var edit = executeSecondCode(editor);
			context.subscriptions.push(
				vscode.languages.registerCodeActionsProvider('javascript', edit, {
					providedCodeActionKinds: Edit.providedCodeActionKinds
				})
			);
			context.subscriptions.push(
				vscode.commands.registerCommand('extension.applyFix', idx => {
					console.log('applyFix ',idx)
					edit.relateChange(idx);
				})
			);
			edit.selectionControl();
		});
	});

}

function deactivate() {
	edit
}

module.exports = {
	activate,
	deactivate
}
