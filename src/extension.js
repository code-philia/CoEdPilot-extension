const vscode = require('vscode');
const { spawn } = require('child_process');
/**
 * @param {vscode.ExtensionContext} context
 */

// ------------ Hyper-parameters ------------
let fontcolor = '#000';
let bgcolor = 'rgba(255,0,0,0.3)';
let listenAnyChanges = false; // true: 监听编辑器的每次一变化; false: 当鼠标指针所在行数变化时监听
let pythonScriptPath = '/Users/russell/Downloads/Code-Edit-main/src/model.py';
// ------------ Global variants -------------
let modifications = [];
let prevCursorAtLine = 0;
let beforeEdit = undefined;
let afterEdit = undefined;
let prevLineCount = undefined;
let currLineCount = undefined;
// let timer;
// let textEditorListener;
// let textDocumentListener;


function convertStringToList(inputString) {
	var replacedString = inputString.replace(/'/g, '"');
	var parsedArray = JSON.parse(replacedString);
	let result = [];
	
	for (var i = 0; i < parsedArray.data.length; i++) {
		var modification = parsedArray.data[i];
		var startIndex = modification[0];
		var endIndex = modification[1];
		var replaceContent = modification[2];

		result.push([startIndex, endIndex, replaceContent]);
	}

	return result;
}

let decorationType = vscode.window.createTextEditorDecorationType({
    color: fontcolor,
	backgroundColor: bgcolor
});
  
function highlightModifications(modifications, editor) {
	const decorations = [];
	if (!editor) {
		return;
	}
	// 遍历每个修改
	for (const modification of modifications) {
		const [start, end, content] = modification;
		// 创建装饰器范围
		const startPos = editor.document.positionAt(start);
		const endPos = editor.document.positionAt(end);
		const range = new vscode.Range(startPos, endPos);
	
		// 创建装饰器
		const decoration = {
			range,
			hoverMessage: 'Modified content: ' + content
		};
	
		// 添加装饰器到数组
		decorations.push(decoration);
	}
	// 应用装饰器到编辑器
	editor.setDecorations(decorationType, decorations);
}
  
function runPythonScript(text, beforeEdit, afterEdit, editor) {
	/*
	* Python 脚本的输出为字典格式: {"data": [start1, end1, replacement1], ...}
	*/
	const pythonProcess = spawn('python', [pythonScriptPath]);
	
	// 将文本写入标准输入流
	pythonProcess.stdin.setEncoding('utf-8');
	pythonProcess.stdin.write(text);
	pythonProcess.stdin.end();

	// 处理 Python 脚本的输出
	pythonProcess.stdout.on('data', (data) => {
		const output = data.toString();
		// 解析 Python 脚本的输出为三元组（修改起始位置、修改结束位置、修改内容）
		modifications = convertStringToList(output);
		// 高亮显示修改的位置
		highlightModifications(modifications, editor);
	});
  
	// 处理 Python 脚本的错误
	pythonProcess.stderr.on('data', (data) => {
	  	console.error(data.toString());
	});
}

function handleTextEditorEvent(editor) {
	if (editor) {
		const document = editor.document;
		const fulltext = document.getText();
		// 在这里执行您对文本内容的操作
		// 您可以访问 `text` 变量来获取当前编辑器内的内容
		runPythonScript(fulltext, undefined, undefined, editor);
	}
}	

function handleTextDocumentEvent(event) {
	const document = event.document;
	const fulltext = document.getText();
	// 在这里执行您对文本内容的操作
	// 您可以访问 `text` 变量来获取当前编辑器内的内容
	runPythonScript(fulltext, undefined, undefined, vscode.window.activeTextEditor);
}

function handleTextEditorSelectionEvent(event) {
	const line = event.selections[0].active.line;
	const currCursorAtLine = line + 1; // VScode API 行数从 0 开始，我们的行数从 1 开始，注意 +- 1
	console.log(`Current cursor position: Line ${currCursorAtLine}`);
	const fulltext = vscode.window.activeTextEditor.document.getText(); // 读取当前编辑器内文本
	if (prevCursorAtLine != currCursorAtLine) {
		if (prevCursorAtLine != 0){ // 当此次鼠标指针位置变化不是第一次选中时
			currLineCount = vscode.window.activeTextEditor.document.lineCount;
			if (currLineCount == prevLineCount - 1) { // 当鼠标指针变化是由于删除了一整行引起的时
				afterEdit = ''; // 此时直接读取上次一指针所在行得到的实际是上一次所在行的下一行
			} else {
				// 获取鼠标指针上一次所在行修改后的内容
				afterEdit = vscode.window.activeTextEditor.document.lineAt(prevCursorAtLine - 1).text; 
			}
		}
		// 立即获取鼠标当前指针修改前的内容，避免因用户在 runPythonScript 运行中修改导致 beforeEdit 内容不准
		let newLineBeforeEdit = vscode.window.activeTextEditor.document.lineAt(currCursorAtLine - 1).text; 
		// 立即获取文本当前行数，避免因用户在 runPythonScript 运行中修改导致行数不准
		let newLineCount = vscode.window.activeTextEditor.document.lineCount;
		console.log(`Line ${prevCursorAtLine} before edit:`,beforeEdit);
		console.log(`Line ${prevCursorAtLine} after  edit:`,afterEdit);
		runPythonScript(fulltext, beforeEdit, afterEdit, vscode.window.activeTextEditor);
		prevCursorAtLine = currCursorAtLine; // 更新鼠标指针所在行数
		beforeEdit = newLineBeforeEdit; 
		prevLineCount = newLineCount;
	}
}

function activate(context) {

	console.log('Congratulations, your extension is now active!');

	if (listenAnyChanges) { // 保留两个实时监听器用于快速 debug
		// 注册一个事件监听器，当编辑器激活时触发
		context.subscriptions.push(
			vscode.window.onDidChangeActiveTextEditor(handleTextEditorEvent)
		);

		// 注册一个事件监听器，当文本发生更改时触发
		context.subscriptions.push(
			vscode.workspace.onDidChangeTextDocument(handleTextDocumentEvent)
		);
	}
	else {
		// 注册一个事件监听器，监听光标位置的变化
		vscode.window.onDidChangeTextEditorSelection(handleTextEditorSelectionEvent);
	}

	// 注册 CodeAction Provider，为 Python 脚本返回的修改位置提供 QuickFix
    let codeActionsProvider = vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, {
        provideCodeActions(document) {
            const codeActions = modifications.map(modifiedContent => {
                const [startPos, endPos, replaceContent] = modifiedContent;
                const diagnosticRange = new vscode.Range(document.positionAt(startPos), document.positionAt(endPos));

                // 创建诊断
                const diagnostic = new vscode.Diagnostic(diagnosticRange, 'Replace with: ' + replaceContent, vscode.DiagnosticSeverity.Hint);
                diagnostic.code = 'replaceCode';

                // 创建快速修复
				const textInRange = document.getText(diagnosticRange);
                const codeAction = new vscode.CodeAction(textInRange+'-->'+replaceContent, vscode.CodeActionKind.QuickFix);
                codeAction.diagnostics = [diagnostic];
                codeAction.isPreferred = true;

                // 创建 WorkspaceEdit
                const edit = new vscode.WorkspaceEdit();
                const replaceRange = new vscode.Range(document.positionAt(startPos), document.positionAt(endPos));
                edit.replace(document.uri, replaceRange, replaceContent);
                codeAction.edit = edit;

                return codeAction;
            });

            return codeActions;
        }
    });

    context.subscriptions.push(codeActionsProvider);
}
 
function deactivate() {
	// 清除装饰器
	decorationType.dispose();
}

module.exports = {
	activate,
	deactivate
}

// Obselet code
// function startTimer() {
//     if (timer) {
//         clearTimeout(timer);
//     }

//     // 设置一个10秒的计时器
// 	stopListeners();
// 	timer = setTimeout(() => {
// 		startListeners();
// 		timer = undefined;
// 	}, listen_interval);
// }

// function stopListeners() {
//     textEditorListener.dispose();
//     textDocumentListener.dispose();
// 	console.log('Listener sleep for '+ String(listen_interval/1000) + 's.')
// }

// function startListeners() {
//     // 注册一个事件监听器，当编辑器激活时触发
//     textEditorListener = vscode.window.onDidChangeActiveTextEditor(handleTextEditorEvent);

//     // 注册一个事件监听器，当文本发生更改时触发
//     textDocumentListener = vscode.workspace.onDidChangeTextDocument(handleTextDocumentEvent);

// 	// 监听器重新工作
// 	console.log('Listener reactivate.')
// }
