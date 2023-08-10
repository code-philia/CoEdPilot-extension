const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const { spawn } = require('child_process');

const { getWebviewContent }=require('./sidebar')
/**
 * @param {vscode.ExtensionContext} context
 */

// ------------ Hyper-parameters ------------
// 获取当前文件所在目录的绝对路径
const extensionDirectory = __dirname;
let fontcolor1 = '#000';
let bgcolor1 = 'rgba(255,0,0,0.3)';
let fontcolor2 = '#000';
let bgcolor2 = 'rgba(0, 255, 0, 0.3)';
let prevEditNum = 3;
let pyPathEditRange = path.join(extensionDirectory, "range_model.py");
let pyPathEditContent = path.join(extensionDirectory, "content_model.py");
let PyInterpreter = "python";
// ------------ Global variants -------------
let modifications = [];
let prevCursorAtLine = 0;
let currCursorAtLine = 0;
let prevSnapshot = undefined;
let currSnapshot = undefined;
let currFile = undefined;
let previousActiveEditor = undefined;
let commitMessage = "";
let prevEdits = []; // 堆栈，长度为 prevEditNum，记录历史修改

function cleanGlobalVariables(editor) {
	if (previousActiveEditor && previousActiveEditor.document.isDirty) {
		previousActiveEditor.document.save(); // 当切换 activeEditor 时，自动保存文档内容
	}
	previousActiveEditor = editor; // 更新 previousActiveEditor

	// modifications = []; // 切换 editor 不需要清空建议的修改位置
	prevCursorAtLine = 0;
	currCursorAtLine = 0;
	prevSnapshot = editor.document.getText();
	currSnapshot = undefined;
	currFile = editor.document.fileName;
	console.log('==> Active File:', currFile);
	console.log('==> Global variables initialized');
	highlightModifications(modifications, editor);
}

let decorationTypeForAlter = vscode.window.createTextEditorDecorationType({
    color: fontcolor1,
	backgroundColor: bgcolor1
});

let decorationTypeForAdd = vscode.window.createTextEditorDecorationType({
    color: fontcolor2,
	backgroundColor: bgcolor2
});
  
function highlightModifications(modifications, editor) {
	const decorationsForAlter = []; // highlight for replace, delete
	const decorationsForAdd = []; // highlight for add
	if (!editor) {
		return;
	}
	// 遍历每个修改
	for (const modification of modifications) {
		if(modification.targetFilePath != currFile) { // 只高亮当前文件的修改
			continue;
		}
		// 创建装饰器范围
		const startPos = editor.document.positionAt(modification.startPos);
		const endPos = editor.document.positionAt(modification.endPos);
		const range = new vscode.Range(startPos, endPos);
	
		// 创建装饰器
		const decoration = {
			range
		};

		// 添加装饰器到数组
		if (modification.editType == 'add') {
			decorationsForAdd.push(decoration);
		} else {
			decorationsForAlter.push(decoration);
		}
	}
	// 应用装饰器到编辑器
	editor.setDecorations(decorationTypeForAlter, decorationsForAlter);
	editor.setDecorations(decorationTypeForAdd, decorationsForAdd);
}

function getFiles() {
	/* 
	* FilesList: Workspace 打开的文件夹内的所有文件及其内容 [[filePath, fileContent], ...]
	*/
	// 获取当前工作区的根文件夹路径
	const rootPath = vscode.workspace.rootPath;
    // 存储所有文件的文本和名称列表
    const filesList = [];

	function readFiles(folderPath) {
		const files = fs.readdirSync(folderPath);
		files.forEach(file => {
			let filePath = undefined;
			filePath = path.join(folderPath, file);
			const fileStat = fs.statSync(filePath);
	
			if (fileStat.isFile()) {
				// 读取文件内容
				const fileContent = fs.readFileSync(filePath, 'utf-8');
				filesList.push([filePath,fileContent]);
			} else if (fileStat.isDirectory()) {
				// 递归遍历子文件夹
				readFiles(filePath);
			}
		});
	}
	
    // 开始遍历当前工作区根文件夹
    readFiles(rootPath);
    return filesList;
}

function showModificationsWebview(modificationsList) {
	let panel = vscode.window.createWebviewPanel(
		"modificationsWebview",
		"Modifications",
		// 当 Webview 和 editor 窗口并列时，
		// 用户回到 editor 的第一次点击会被识别成 onDidChangeActiveTextEditor，
		// 若第一次点击在高亮位置，则不会激活 generator，对用户造成困扰
		// 因此使用 vscode.ViewColumn.One 避免并列
		vscode.ViewColumn.One, 
		{ enableScripts: true }
	);
	const rootPath = vscode.workspace.rootPath;
    panel.webview.html = getWebviewContent(modificationsList, rootPath);
	panel.webview.onDidReceiveMessage(message => {
        if (message.command === 'openFile') {
            const filePath = message.path;
            // 打开文件
            vscode.workspace.openTextDocument(filePath).then(document => {
                vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One });
            });
        }
    });
}

function runPythonScript1(files, prevEdits, editor) {
	/*
	* 输入 Python 脚本的内容为字典格式: {"files": list, [[filePath, fileContent], ...],
	*                                "targetFilePath": str, filePath,
	*								 "commitMessage": str, commit message,
	*								 "prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}}
	* Python 脚本的输出为字典格式: {"data": , [ { "targetFilePath": str, filePath,
    *                                          "beforeEdit", str, the content before edit for previous edit,
	*                                          "afterEdit", str, the content after edit for previous edit, 
	*										   "toBeReplaced": str, the content to be replaced, 
	*                                          "startPos": int, start position of the word,
	*                                          "endPos": int, end position of the word,
	* 										   "editType": str, the type of edit, add or remove,
	*										   "lineBreak": str, '\n', '\r' or '\r\n'}, ...]}
	*/
	const pythonProcess = spawn(PyInterpreter, [pyPathEditRange]);
	const activeFilePath = editor.document.fileName;
	const input = {files: files, 
				   targetFilePath: activeFilePath,
				   commitMessage: commitMessage,
			 	   prevEdits: prevEdits};
	const strJSON = JSON.stringify(input);

	// 将文本写入标准输入流
	pythonProcess.stdin.setEncoding('utf-8');
	pythonProcess.stdin.write(strJSON);
	pythonProcess.stdin.end();
	console.log('==> Sent to edit locator model');

	// 处理 Python 脚本的输出
	pythonProcess.stdout.on('data', (data) => {
		const output = data.toString();
		// 解析 Python 脚本的输出为三元列表（文件名，修改起始位置，修改结束位置）
		// var replacedString = output.replace(/'/g, '"');
		var parsedJSON = JSON.parse(output);
		modifications = parsedJSON.data;
		console.log('==> Edit locator model returned successfully');
		// 高亮显示修改的位置
		highlightModifications(modifications, editor);
		showModificationsWebview(modifications);
	});
	
	// 处理 Python 脚本的错误
	pythonProcess.stderr.on('data', (data) => {
	  	console.error(data.toString());
	});
}

function runPythonScript2(modification) {
	/*
	* 输入 Python 脚本的内容为字典格式: { "files": list, [[filePath, fileContent], ...],
	* 								"targetFilePath": string filePath,
	*								"commitMessage": string, commit message,
	*								"editType": string, edit type,
	*								"prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
	*								"startPos": int, start position,
	*								"endPos": int, end position}
	* 输出 Python 脚本的内容为字典格式: {"data": 
	*                                       { "targetFilePath": string, filePath of target file,
	* 										  "editType": string, 'remove', 'add'
	*                                         "startPos": int, start position,
	*                                         "endPos": int, end position,
	*                                         "replacement": list of strings, replacement content   
	*                                       }
	*                               }
	*/
	return new Promise((resolve, reject) => {
		let files = getFiles();
		const pythonProcess = spawn(PyInterpreter, [pyPathEditContent]);
		const editor = vscode.window.activeTextEditor;

		for (let file of files) {
			if(file[0] === currFile) {
				file[1]=currSnapshot; // 把未保存的文件内容作为实际文件内容
				break;
			} 
		}

		const input = {
			files: files,
			targetFilePath: modification.targetFilePath,
			commitMessage: commitMessage,
			editType: modification.editType,
			prevEdits: modification.prevEdits,
			startPos: modification.startPos,
			endPos: modification.endPos
		};
		const strJSON = JSON.stringify(input);
	
		// 将文本写入标准输入流
		pythonProcess.stdin.setEncoding('utf-8');
		pythonProcess.stdin.write(strJSON);
		pythonProcess.stdin.end();
		console.log('==> Sent to edit generator model');
	
		// 处理 Python 脚本的输出
		pythonProcess.stdout.on('data', (data) => {
			const output = data.toString();
			// 解析 Python 脚本的输出为三元列表（文件名，修改起始位置，修改结束位置）
			// var replacedString = output.replace(/'/g, '"');
			var parsedJSON = JSON.parse(output);
			let newmodification = parsedJSON.data;
			console.log('==> Edit generator model returned successfully');
			// 高亮显示修改的位置
			highlightModifications([newmodification], editor);
			resolve(newmodification); // 返回 newmodification
		});
	
		// 处理 Python 脚本的错误
		pythonProcess.stderr.on('data', (data) => {
			console.error(data.toString());
			reject(data.toString());
		});
	});
}

function detectEdit(prev, curr) {
	// 将字符串按行分割成字符串列表
	const prevSnapshotStrList = prev.match(/(.*?(?:\r\n|\n|\r|$))/g).slice(0, -1); // 保留每行尾部换行符
	const currSnapshotStrList = curr.match(/(.*?(?:\r\n|\n|\r|$))/g).slice(0, -1); // 保留每行尾部换行符

	// 从头部找到不同的行号
	let start = 0;
	while (start < prevSnapshotStrList.length && start < currSnapshotStrList.length && prevSnapshotStrList[start] === currSnapshotStrList[start]) {
	  start++;
	}
	
	// 从尾部找到不同的行号
	let end = 0;
	while (end < prevSnapshotStrList.length - start && end < currSnapshotStrList.length - start && prevSnapshotStrList[prevSnapshotStrList.length - 1 - end] === currSnapshotStrList[currSnapshotStrList.length - 1 - end]) {
	  end++;
	}
	
	// 将剩余的行重新组合成字符串
	const beforeEdit = prevSnapshotStrList.slice(start, prevSnapshotStrList.length - end).join('');
	const afterEdit = currSnapshotStrList.slice(start, currSnapshotStrList.length - end).join('');
	
	// 返回结果
	return {
	  beforeEdit: beforeEdit.trim(),
	  afterEdit: afterEdit.trim()
	};
}

function pushToStack(item) {
	prevEdits.push(item);
  
	if (prevEdits.length > prevEditNum) {
		prevEdits.shift(); // 先入先出弹出最早的元素
	}
}

function handleTextEditorSelectionEvent(event) {
	const line = event.selections[0].active.line;
	currCursorAtLine = line + 1; // VScode API 行数从 0 开始，我们的行数从 1 开始，注意 +- 1
	console.log(`==> Cursor position: Line ${prevCursorAtLine} -> ${currCursorAtLine}`);
	currSnapshot = vscode.window.activeTextEditor.document.getText(); // 读取当前编辑器内文本
	if (prevCursorAtLine != currCursorAtLine && prevCursorAtLine != 0) { // 当指针改变位置 且 不是在编辑器第一次获得位置时
		let edition = detectEdit(prevSnapshot, currSnapshot); // 检测相比上一次快照的变化

		if (edition.beforeEdit != edition.afterEdit) {
			// 把该修改增加到 prevEdit 中
			pushToStack(edition);
			console.log('==> Before edit:\n', edition.beforeEdit);
			console.log('==> After edit:\n', edition.afterEdit); 
			prevSnapshot = currSnapshot;
			console.log('==> Send to LLM (After cursor changed line)');
			let files = getFiles();
			//因为fs方式只能拿到已保存的代码文本
			for (let file of files) {
				if(file[0] === currFile) {
					file[1] = currSnapshot; // 把未保存的文件内容作为实际文件内容
					runPythonScript1(files, prevEdits, vscode.window.activeTextEditor);
					break;
				} 
			}
		}
	}
	prevCursorAtLine = currCursorAtLine; // 更新鼠标指针所在行数
}

function updateAfterApplyQuickFix(text) {
	/*
	* 当用户采纳了 QuickFix 的建议时，不需要等到指针所在行变化，直接将修改后的版本送入 LLM 更新 modifications
	*/
	let edition = detectEdit(prevSnapshot, text); // 检测相比上一次快照的变化
	currSnapshot = text;
	if (edition.beforeEdit != edition.afterEdit) {
		// 把该修改增加到 prevEdit 中
		pushToStack(edition);
		console.log('==> Before edit:\n', edition.beforeEdit); 
		console.log('==> After edit:\n', edition.afterEdit);
		prevSnapshot = currSnapshot;
		console.log('==> Send to LLM (After apply QucikFix)');
		let files = getFiles();
        for (let file of files) {
            if(file[0] === currFile) {
                file[1] = currSnapshot;
                runPythonScript1(files, prevEdits, vscode.window.activeTextEditor);
                break;
            } 
        }
	}
	prevCursorAtLine = currCursorAtLine; // 更新鼠标指针所在行数
}

function OriginEditorEvent() {
	//在启动时获得当前文件名
	let editor = vscode.window.activeTextEditor;
	if (editor) {
        // 获取当前活动的文本编辑器
        const activeEditor = editor;
		var file = activeEditor.document.fileName;
        const activeFilePath = file;
        console.log('==> Active File:', activeFilePath);
		currFile = activeFilePath;
    }
}	

async function getModification(doc, range) {
	const filePath = doc.uri.fsPath;
	const startPos = doc.offsetAt(range.start);
    const endPos = doc.offsetAt(range.end);
	for(let modification of modifications) {
		if (filePath == modification.targetFilePath && modification.startPos <= startPos && modification.endPos >= endPos) {
			let highlightedRange = new vscode.Range(doc.positionAt(modification.startPos), doc.positionAt(modification.endPos));
			if (doc.getText(highlightedRange) == modification.toBeReplaced) { 
				// 当用户不按照推荐修改时，例如推荐修改单词 good，但用户删除了 good，此时会导致高亮位置对应的内容偏移，则无需进行修正内容推荐
				return await runPythonScript2(modification);
			} else {
				console.log('==> The suggested edit target:');
				console.log(doc.getText(highlightedRange));
				console.log('==> Current edit target:');
				console.log(modification.toBeReplaced);
				console.log('==> Highlighted range is not the suggested edit range');
				// 此时清空 modifications，取消所有高亮
				modifications = [];
				highlightModifications(modifications, vscode.window.activeTextEditor);
				return undefined;
			}
		}
	}
	return undefined;
}

function activate(context) {
	console.log('==> Congratulations, your extension is now active!');

	/*----------------------- Monitor edit behavior --------------------------------*/
	OriginEditorEvent(); // 获取当前活跃编辑器和在编辑器内打开的文件名与路径

	previousActiveEditor = vscode.window.activeTextEditor;
	// 当 VSCode 存在默认打开的 activeTextEditor 时，自动读取当前文本内容作为 prevSnapshot
	prevSnapshot = vscode.window.activeTextEditor.document.getText();
	currSnapshot = vscode.window.activeTextEditor.document.getText(); // 读取当前编辑器内文本

	// 注册一个事件监听器，当切换编辑器激活时触发，初始化全局变量
	context.subscriptions.push(
		vscode.window.onDidChangeActiveTextEditor(cleanGlobalVariables)
	);

	// 注册一个事件监听器，监听光标位置的变化
	vscode.window.onDidChangeTextEditorSelection(handleTextEditorSelectionEvent);
	/*----------------------- Monitor edit behavior --------------------------------*/

	/*----------------------- Provide QuickFix feature -----------------------------*/
	// 注册 CodeAction Provider，为 Python 脚本返回的修改位置提供 QuickFix
    let codeActionsProvider = vscode.languages.registerCodeActionsProvider({ scheme: 'file' }, {
        async provideCodeActions(document, range) {
			const newmodification = await getModification(document, range);
			
			if (! newmodification || newmodification.targetFilePath != currFile )
				return [];

			const diagnosticRange = new vscode.Range(document.positionAt(newmodification.startPos), document.positionAt(newmodification.endPos));
			
			const codeActions = newmodification.replacement.map(replacement => {
				// 创建诊断
				const diagnostic = new vscode.Diagnostic(diagnosticRange, 'Replace with: ' + replacement, vscode.DiagnosticSeverity.Hint);
				diagnostic.code = 'replaceCode';
				
				// 创建快速修复
				const codeAction = new vscode.CodeAction(replacement, vscode.CodeActionKind.QuickFix);
				codeAction.diagnostics = [diagnostic];
				codeAction.isPreferred = true;

				// 创建 WorkspaceEdit
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

			return codeActions;
        }
    });

    context.subscriptions.push(codeActionsProvider);

    context.subscriptions.push(
		vscode.commands.registerCommand('extension.applyFix', async () => {
			console.log('==> applyFix');
			const editor = vscode.window.activeTextEditor;
			if (editor) {
				modifications = []; // 清除 modifications 内容，避免因为指针还处于建议修改位置时触发 runPythonScript2
				highlightModifications(modifications, editor); // 出现新的推荐位置高亮可能会耗费一定时间，该段时间内此修改位置需避免继续高亮
				updateAfterApplyQuickFix(editor.document.getText());
			}
		})
	);
	/*----------------------- Provide QuickFix feature ---------------------------------*/

	/*----------------------- Edit description input box --------------------------------*/
	const inputBox = vscode.window.createInputBox();
	inputBox.prompt = 'Enter edit description';
	inputBox.ignoreFocusOut = true; // 输入框在失去焦点后不会隐藏

	vscode.commands.registerCommand('extension.commit_message',async function(){
		console.log('==> Edit description input box is displayed')
		inputBox.show();
	});

	inputBox.onDidAccept(() => { // 用户回车确认 commit message 后进行 edit range 推荐
		const userInput = inputBox.value;
		console.log('==> Edit description:', userInput);
		commitMessage = userInput;
		let files = getFiles();
		for (let file of files) {
			if(file[0] === currFile) {
				file[1] = currSnapshot;
				runPythonScript1(files, prevEdits, vscode.window.activeTextEditor);
				break;
			} 
		}
	});

	inputBox.onDidHide(() => {
		// 输入框被隐藏后的处理逻辑
		console.log('==> Edit description input box is hidden');
	});
	/*----------------------- Edit description input box --------------------------------*/

	/*----------------------- Webview of modification list ------------------------------*/
	
}
 
function deactivate() {
	// 清除装饰器
	decorationTypeForAlter.dispose();
	decorationTypeForAdd.dispose();
}

module.exports = {
	activate,
	deactivate
}
