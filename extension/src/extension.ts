import vscode from 'vscode';
import { compareTempFileSystemProvider } from './views/compare-view';
import { FileStateMonitor, initFileState } from './utils/file-utils';
import { editorState, queryState } from './global-context';
import { editLocationView } from './views/location-tree-view';
import { LocationDecoration } from './ui/location-decoration';
import { registerBasicCommands, registerTopTaskCommands } from './comands';
import { statusBarItem } from './ui/progress-indicator';
import { modelServerProcess } from './services/backend-requests';

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
