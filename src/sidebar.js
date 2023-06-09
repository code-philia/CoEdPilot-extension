function getWebviewContent(content) {
	let codeSuggestions = "";
  
	content.forEach((suggestion, index) => {
	  const [oldCode, newCode] = suggestion;
	  const suggestionElement = `
		<div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
		  <p style="font-weight: bold; color: #222;">Old Code:</p>
		  <pre style="background-color: #f4f4f4; padding: 5px; margin: 0; font-family: Consolas, monospace; color: #222; font-weight: 600;">${oldCode}</pre>
		  <p style="font-weight: bold; color: #222;">New Code:</p>
		  <pre style="background-color: #f4f4f4; padding: 5px; margin: 0; font-family: Consolas, monospace; color: #222; font-weight: 600;">${newCode}</pre>
		  <div>
			<button style="background-color: #555; color: white; border: none; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 10px;" onclick="handleButtonClick(${index})">Apply</button>
			<button style="background-color: #888; color: white; border: none; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 10px; margin-left: 5px;" onclick="handleUndoButtonClick(${index})">Undo</button>
		  </div>
		</div>
	  `;
	  codeSuggestions += suggestionElement;
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
	  <h1>Please check code edits <button class="apply-all-button" onclick="handleApplyAllButtonClick()">Apply All</button></h1>
  
	  <div id="suggestions">
		${codeSuggestions}
	  </div>
  
	  <div class="feedback-container">
		<h2 class="feedback-title">Submit Feedback</h2>
		<div class="feedback-input-container">
		  <input type="text" class="feedback-input" placeholder="Enter your feedback">
		  <button class="feedback-button" onclick="submitFeedback()">Submit</button>
		</div>
	  </div>
  
	  <script>
		const vscode = acquireVsCodeApi();
  
		function handleButtonClick(index) {
		  const apply = {  "applyAll": false ,"idx": index, "undo": false };
		  vscode.postMessage(apply);
		}
  
		function handleUndoButtonClick(index) {
		  const undo = { "applyAll": false , "idx": index, "undo": true };
		  vscode.postMessage(undo);
		}
		
		function handleApplyAllButtonClick() {
		  const applyAll = { "applyAll": true };
		  vscode.postMessage(applyAll);
		}
  
		function submitFeedback() {
		  const feedbackInput = document.querySelector('.feedback-input');
		  const feedback = feedbackInput.value.trim();
		  if (feedback) {
			vscode.postMessage({ "feedback": feedback });
			feedbackInput.value = '';
		  }
		}
	  </script>
	</body>
	</html>
	`;
  }
  
  module.exports = {
	getWebviewContent
  };
  