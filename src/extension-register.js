import vscode from "vscode";
import { PredictLocationCommand, GenerateEditCommand } from "./query-tasks";
import { registerCommand, numIn } from "./base-component";

export function registerBasicCommands() {
	return vscode.Disposable.from(
		registerCommand("coEdPilot.openFileAtLine", async (filePath, fromLine, toLine) => {
			const uri = vscode.Uri.file(filePath); // Replace with dynamic file path

			const document = await vscode.workspace.openTextDocument(uri);
			const editor = await vscode.window.showTextDocument(document);

			fromLine = numIn(fromLine, 0, document.lineCount - 1);
			toLine = numIn(toLine, 0, document.lineCount - 1);
			const range = new vscode.Range(
				editor.document.lineAt(fromLine).range.start,
				editor.document.lineAt(toLine).range.start,
			);

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
