import vscode from "vscode";
import { compareTempFileSystemProvider } from "./compare-view";
import { FileStateMonitor, initFileState } from "./file";
import { editorState, queryState } from "./global-context";
import { editLocationView } from "./activity-bar";
import { LocationDecoration } from "./inline";
import { registerBasicCommands, registerTopTaskCommands } from "./extension-register";
import { statusBarItem } from "./status-bar";
import { modelServerProcess } from "./model-client";
import { activate_notiy } from "./notification";

function activate(context) {

	initFileState(vscode.window.activeTextEditor);

	context.subscriptions.push(
		editorState,
		queryState,
		compareTempFileSystemProvider,
		statusBarItem,
		editLocationView,
		modelServerProcess
	);

	context.subscriptions.push(
		registerBasicCommands(),
		registerTopTaskCommands(),
	);

	context.subscriptions.push(
		new FileStateMonitor(),
		new LocationDecoration(),	
	);

	activate_notiy(context);

	console.log("==> Congratulations, your extension is now active!");
}

function deactivate() {
}

export {
	activate,
	deactivate
};
