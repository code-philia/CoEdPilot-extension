const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const { spawn } = require('child_process');

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
			console.log('==> Edit pused to prevEdit stack');
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

	/*----------------------- Get selected text ------------------------------------*/
	let getRemoveEditLocation = vscode.commands.registerCommand('extension.generate_remove_edit', () => {
		let editor = vscode.window.activeTextEditor;
		if (editor) {
			let document = editor.document;
			let selection = editor.selection;
		
			let text = document.getText(selection);
			let targetFilePath = document.uri.path;
			let startLineIdx = selection.start.line; // 行号从 0 开始
			let endLineIdx = selection.end.line; // 行号从 0 开始
			// console.log(text);
			// console.log(targetFilePath);
			// console.log(startLineIdx, endLineIdx);

			// 处理 Pyth
			let files = getFiles();
			const pythonProcess = spawn(PyInterpreter, [pyPathEditContent]);

			for (let file of files) {
				if(file[0] === currFile) {
					file[1]=currSnapshot; // 把未保存的文件内容作为实际文件内容
					break;
				} 
			}
			const input = {
				files: files,
				targetFilePath: targetFilePath,
				commitMessage: commitMessage,
				editType: "remove",
				prevEdits: prevEdits,
				startPos: document.offsetAt(selection.start),
				endPos: document.offsetAt(selection.end),
				atLine: Array.from({ length: endLineIdx - startLineIdx + 1 }, (_, i) => i + startLineIdx)
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
				console.log('Input\n', parsedJSON.input);
				console.log('==> Returned successfully');
				editor.edit(editBuilder => {
					// 标记原先的内容
					for (let i = startLineIdx; i <= endLineIdx; i++) {
						let position = new vscode.Position(i, 0); // 创建表示行起始位置的Position对象
						editBuilder.insert(position, '-'); // 在该位置插入一个 "-" 号
					}
				  
					// 插入返回的修改内容
					let newPosition = new vscode.Position(endLineIdx+1, 0);
					let cnt = 1;
					for (let str of newmodification.replacement) {
						editBuilder.insert(newPosition, cnt + '+ ' + str + '\n');
						cnt = cnt + 1;
					}
				});
			});
			pythonProcess.stderr.on('data', (data) => {
				console.error(`stderr: ${data}`);
			});  
		}
	});

	context.subscriptions.push(getRemoveEditLocation);

	let getAddEditLocation = vscode.commands.registerCommand('extension.generate_add_edit', () => {
		let editor = vscode.window.activeTextEditor;
		if (editor) {
			let document = editor.document;
			let selection = editor.selection;
		
			let text = document.getText(selection);
			let targetFilePath = document.uri.path;
			let startLineIdx = selection.start.line + 1; // 行号从 1 开始
			let endLineIdx = selection.end.line + 1; // 行号从 1 开始
			// console.log(text);
			// console.log(targetFilePath);
			// console.log(startLineIdx, endLineIdx);

			// 处理 Pyth
			let files = getFiles();
			const pythonProcess = spawn(PyInterpreter, [pyPathEditContent]);

			for (let file of files) {
				if(file[0] === currFile) {
					file[1]=currSnapshot; // 把未保存的文件内容作为实际文件内容
					break;
				} 
			}
			const input = {
				files: files,
				targetFilePath: targetFilePath,
				commitMessage: commitMessage,
				editType: "add",
				prevEdits: prevEdits,
				startPos: document.offsetAt(selection.start),
				endPos: document.offsetAt(selection.end),
				atLine: Array.from({ length: endLineIdx - startLineIdx + 1 }, (_, i) => i + startLineIdx)
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
				console.log('==> Returned successfully');
				// 插入返回的修改内容
				let newPosition = new vscode.Position(endLineIdx, 0);
				editor.edit(editBuilder => {
					let cnt = 1;
					for (let str of newmodification.replacement) {
					  editBuilder.insert(newPosition, cnt+'+ '+str+'\n');
					  cnt = cnt + 1;
					}
				  });
			});
				
		}
	});

	context.subscriptions.push(getAddEditLocation);

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
	});

	inputBox.onDidHide(() => {
		// 输入框被隐藏后的处理逻辑
		console.log('==> Edit description input box is hidden');
	});
	/*----------------------- Edit description input box --------------------------------*/	
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
