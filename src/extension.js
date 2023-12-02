import * as vscode from 'vscode';
import { compareTempFileSystemProvider, DiffTabCodelensProvider } from './compare-view';
import { FileStateMonitor, initFileState } from './file';
import { LocationTreeProvider } from './activity-bar';
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
		new LocationTreeProvider(),
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
