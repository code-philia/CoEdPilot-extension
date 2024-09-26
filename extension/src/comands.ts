import vscode from "vscode";
import { PredictLocationCommand, GenerateEditCommand } from "./services/query-tasks";
import { limitNum } from "./utils/utils";
import { globalEditDetector } from "./editor-state-monitor";
// import { addUserStatItem } from "./global-context";

export function registerBasicCommands() {
	return vscode.Disposable.from(
		vscode.commands.registerCommand('coEdPilot.openFileAtLine', async (filePath, fromLine, toLine) => {
			const uri = vscode.Uri.file(filePath); // Replace with dynamic file path

			const document = await vscode.workspace.openTextDocument(uri);
			const editor = await vscode.window.showTextDocument(document);

			const isOverflowed = toLine > document.lineCount - 1;
			fromLine = limitNum(fromLine, 0, document.lineCount - 1);
			toLine = limitNum(toLine, 0, document.lineCount - 1);
			const range = new vscode.Range(
				document.lineAt(fromLine).range.start,
				isOverflowed ? document.lineAt(toLine).range.end : document.lineAt(toLine).range.start,
			);

			editor.selection = new vscode.Selection(range.start, range.end);
			editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
		}),
		vscode.commands.registerCommand('coEdPilot.clearPrevEdits', async () => {
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
