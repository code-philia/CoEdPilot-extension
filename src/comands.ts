import vscode from "vscode";
import { PredictLocationCommand, GenerateEditCommand } from "./query-tasks";
import { registerCommand, numIn } from "./base-component";
import { globalEditDetector } from "./file";
// import { addUserStatItem } from "./global-context";

export function registerBasicCommands() {
	return vscode.Disposable.from(
		registerCommand('coEdPilot.openFileAtLine', async (filePath, fromLine, toLine) => {
			const uri = vscode.Uri.file(filePath); // Replace with dynamic file path

			const document = await vscode.workspace.openTextDocument(uri);
			const editor = await vscode.window.showTextDocument(document);

			const isOverflowed = toLine > document.lineCount - 1
			fromLine = numIn(fromLine, 0, document.lineCount - 1);
			toLine = numIn(toLine, 0, document.lineCount - 1);
			const range = new vscode.Range(
				document.lineAt(fromLine).range.start,
				isOverflowed ? document.lineAt(toLine).range.end : document.lineAt(toLine).range.start,
			)

			editor.selection = new vscode.Selection(range.start, range.end);
			editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
		}),
		registerCommand('coEdPilot.clearPrevEdits', async () => {
			globalEditDetector.clearEditsAndSnapshots();
			await vscode.window.showInformationMessage("Previous edits cleared!");
			// addUserStatItem("clearPrevEdits");
		})
	);
}

export function registerTopTaskCommands() {
	return vscode.Disposable.from(
		new PredictLocationCommand(),
		new GenerateEditCommand()
	);
}
