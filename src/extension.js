import vscode from 'vscode';
import { compareTempFileSystemProvider } from './compare-view';
import { FileStateMonitor, initFileState } from './file';
import { editorState, queryState } from './global-context';
import { editLocationView } from './activity-bar';
import { LocationDecoration } from './inline';
import { registerBasicCommands, registerTopTaskCommands } from './extension-register';
import { statusBarItem } from './status-bar';
import { modelServerProcess } from './model-client';

function activate(context) {
	context.subscriptions.push(
		editorState,
		queryState,
		compareTempFileSystemProvider,
		statusBarItem,
		editLocationView,
		modelServerProcess
		)
		
	context.subscriptions.push(
		registerBasicCommands(),
		registerTopTaskCommands(),
		);
			
	context.subscriptions.push(
		new FileStateMonitor(),
		new LocationDecoration(),	
		);
				
	initFileState(vscode.window.activeTextEditor);
	console.log('==> Congratulations, your extension is now active!');
}

function deactivate() {
}

export {
	activate,
	deactivate
};
