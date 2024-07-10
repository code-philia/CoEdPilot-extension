import vscode from 'vscode';
import { compareTempFileSystemProvider } from './compare-view';
import { FileStateMonitor, initFileState } from './file';
import { editorState, queryState } from './global-context';
import { editLocationView } from './location-tree';
import { LocationDecoration } from './location-decoration';
import { registerBasicCommands, registerTopTaskCommands } from './comands';
import { statusBarItem } from './status-bar';
import { modelServerProcess } from './client';

function activate(context: vscode.ExtensionContext) {
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
