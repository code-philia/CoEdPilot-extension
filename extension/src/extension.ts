import vscode from 'vscode';
import { globalEditorState } from './global-workspace-context';
import { FileStateMonitor, updateEditorState } from './editor-state-monitor';
import { compareTempFileSystemProvider } from './views/compare-view';
import { globalLocationViewManager } from './views/location-tree-view';
import { registerBasicCommands, registerTopTaskCommands } from './comands';
import { statusBarItem } from './ui/progress-indicator';
import { modelServerProcess } from './services/backend-requests';

function activate(context: vscode.ExtensionContext) {
	context.subscriptions.push(
		globalEditorState,
		compareTempFileSystemProvider,
		statusBarItem,
		globalLocationViewManager,
		modelServerProcess
		)
		
	context.subscriptions.push(
		registerBasicCommands(),
		registerTopTaskCommands(),
		);
			
	context.subscriptions.push(
		new FileStateMonitor(),
		);
				
	updateEditorState(vscode.window.activeTextEditor);
}

function deactivate() {
}

export {
	activate,
	deactivate
};
