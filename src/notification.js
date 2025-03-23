import * as vscode from 'vscode';
import axios from 'axios';

export function activate_notiy(context) {
    var first_success = true;
    async function validateBackendConnection(queryURL, showMessage = true) {
        try {
            const response = await axios.get(queryURL + "/check");
            if (response.status === 200) {
                if (showMessage || first_success) {
                    vscode.window.showInformationMessage('✅ Connect to backend successfully! 🎉');
                }
                first_success = false;
            } else {
                first_success = true;
                vscode.window.showErrorMessage('Backend connection failed: Invalid response.');
            }
        } catch (error) {
            first_success = true;
            vscode.window.showErrorMessage(`Backend connection failed: ${error.message}`);
        }
    }

    context.subscriptions.push(
        vscode.workspace.onDidChangeWorkspaceFolders(() => {
            const queryURL = vscode.workspace.getConfiguration('coEdPilot').get('queryURL');
            if (queryURL) {
                validateBackendConnection(queryURL);
            } else {
                vscode.window.showWarningMessage('Query URL is not set in settings.');
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('coEdPilot.queryURL')) {
                const queryURL = vscode.workspace.getConfiguration('coEdPilot').get('queryURL');
                if (queryURL) {
                    validateBackendConnection(queryURL);
                } else {
                    vscode.window.showWarningMessage('Query URL is not set in settings.');
                }
            }
        })
    );

    const interval = setInterval(() => {
        const queryURL = vscode.workspace.getConfiguration('coEdPilot').get('queryURL');
        if (queryURL) {
            validateBackendConnection(queryURL, false);
        } else {
            vscode.window.showWarningMessage('Query URL is not set in settings.');
        }
    }, 5000);

    context.subscriptions.push({
        dispose: () => clearInterval(interval)
    });

    const initialQueryUrl = vscode.workspace.getConfiguration('coEdPilot').get('queryURL');
    if (initialQueryUrl) {
        validateBackendConnection(initialQueryUrl);
    } else {
        vscode.window.showWarningMessage('Query URL is not set in settings.');
    }
}

export function deactivate() { }