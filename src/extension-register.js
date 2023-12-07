import vscode from "vscode";
import { PredictLocationCommand, GenerateEditCommand } from "./query-tasks";
import { registerCommand } from "./base-component";

export function registerBasicCommands() {
	return vscode.Disposable.from(
		registerCommand('editPilot.openFileAtLine', async (filePath, lineNum) => {
			const uri = vscode.Uri.file(filePath); // Replace with dynamic file path

			const document = await vscode.workspace.openTextDocument(uri);
			const editor = await vscode.window.showTextDocument(document);
			const range = editor.document.lineAt(lineNum).range;
			editor.selection = new vscode.Selection(range.start, range.end);
			editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
		})
	);
}

export function registerTopTaskCommands() {
	return vscode.Disposable.from(
		new PredictLocationCommand(),
		new GenerateEditCommand()
	);
}
