import * as vscode from 'vscode';
import { compareTempFileSystemProvider } from './compare-view';
import { FileStateMonitor, initFileState } from './file';
import { EditLocationView } from './activity-bar';
import { CommitMessageInput } from './queries';
import { LocationDecoration } from './inline';
import { registerBasicCommands, registerTopTaskCommands } from './extension-register';

function activate(context) {

	initFileState(vscode.window.activeTextEditor);

	context.subscriptions.push(
		// fileState,
		// queryState,
		compareTempFileSystemProvider
	)

	context.subscriptions.push(
		registerBasicCommands(),
		registerTopTaskCommands(),
	);

	context.subscriptions.push(
		new FileStateMonitor(),
		new LocationDecoration(),
		new CommitMessageInput(),
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
