import * as vscode from 'vscode';
import { compareTempFileSystemProvider } from './compare-view';
import { FileStateMonitor, initFileState } from './file';
import { fileState } from './context';
import { EditLocationView } from './activity-bar';
import { queryState } from './context';
import { LocationDecoration } from './inline';
import { registerBasicCommands, registerTopTaskCommands } from './extension-register';

function activate(context) {

	initFileState(vscode.window.activeTextEditor);

	context.subscriptions.push(
		fileState,
		queryState,
		compareTempFileSystemProvider
	)

	context.subscriptions.push(
		registerBasicCommands(),
		registerTopTaskCommands(),
	);

	context.subscriptions.push(
		new FileStateMonitor(),
		new LocationDecoration(),
		new EditLocationView()
		// new DiffTabCodelensProvider()
	);

	console.log('==> Congratulations, your extension is now active!');
}

function deactivate() {
}

export {
	activate,
	deactivate
};
