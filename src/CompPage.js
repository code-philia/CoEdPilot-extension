function escapeHtmlTags(input) { // 避免代码内因为包含左右尖括号而被 html 转义为 html 元素
	return input.replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function getWebviewContent(modifications, rootPath) {
	let elements = "";
  
	modifications.forEach((modification) => {
        const absoluteTargetFilePath = modification.targetFilePath.replace(/\\/g, '\\\\');
        const relativeTargetFilePath = modification.targetFilePath.replace(rootPath, ".");
        const toBeReplaced = modification.toBeReplaced;
        const atLine = modification.atLine.map(element => element + 1); // atLine 从 0 开始，但是显示的行数从 1 开始
        const element = `
            <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
            <p style="font-weight: bold; color: #222;">Target File Path:</p>
            <pre style="background-color: #f4f4f4; padding: 5px; margin: 0; font-family: Consolas, monospace; color: #222; font-weight: 600; overflow-x: auto;"><a href="#" onclick="openFile('${absoluteTargetFilePath}'); return false;">${relativeTargetFilePath}</a></pre>
            <p style="font-weight: bold; color: #222;">Code to be edited:</p>
            <pre style="background-color: #f4f4f4; padding: 5px; margin: 0; font-family: Consolas, monospace; color: #222; font-weight: 600; overflow-x: auto;">Line ${atLine}:\n${escapeHtmlTags(toBeReplaced)}</pre>
            </div>
        `;
	    elements += element; 
	});
  
	return `<!DOCTYPE html>
	<html lang="en">
	<head>
	  <meta charset="UTF-8">
	  <meta name="viewport" content="width=device-width, initial-scale=1.0">
	  <title>Code suggestion</title>
	  <style>
		body {
		  font-family: Arial, sans-serif;
		  margin: 20px;
		  background-color: #eee;
		  color: #333;
		}
		h1 {
		  font-size: 24px;
		  margin-bottom: 20px;
		  display: flex;
		  align-items: center;
		}
		#suggestions {
		  margin-bottom: 20px;
		}
		.feedback-container {
		  padding-left:10px;
		  padding-right:10px;
		  border: 1px solid #ccc;
		  border-radius: 5px;
		  background-color: #f4f4f4;
		  position: fixed;
		  bottom: 15px;
		  left: 20px;
		  right: 20px;
		}
		.feedback-title {
		  font-weight: bold;
		  margin-bottom: 10px;
		  font-size: 16px;
		}
		.feedback-input-container {
		  display: flex;
		}
		.feedback-input {
		  flex: 1;
		  padding: 8px;
		  margin-bottom: 10px;
		  border: 1px solid #ccc;
		  border-radius: 3px;
		  box-sizing: border-box;
		}
		.feedback-button {
		  background-color: #555;
		  color: white;
		  border: none;
		  margin-bottom: 10px;
		  padding: 8px 16px;
		  font-size: 14px;
		  cursor: pointer;
		  margin-left: 10px;
		}
		.apply-all-button {
		  background-color: #555;
		  color: white;
		  border: none;
		  padding: 8px 16px;
		  font-size: 14px;
		  cursor: pointer;
		  margin-left: 10px;
		}
	  </style>
	</head>
	<body>
        <h1> Suggested edit locations </h1>
        <div id="modificationsWebview">
            ${elements}
        </div>
	</body>
    <script>
        const vscode = acquireVsCodeApi();
        function openFile(filePath) {
            // 发送消息给扩展，让扩展打开对应的文件，并指定要操作的 Webview 面板
            vscode.postMessage({
                command: 'openFile',
                path: filePath,
            });
        }
    </script>
	</html>
	`;
}

module.exports = {
    getWebviewContent
};