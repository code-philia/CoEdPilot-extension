import vscode from 'vscode';
import { BaseComponent } from './base-component';
import { isLanguageSupported } from './global-context';

class ProgressDisplayStatusBarItem extends BaseComponent {
    constructor() {
        super();
        this.loadingIconId = "loading~spin";
        this.item = vscode.window.createStatusBarItem('editPilot.progressDisplay', vscode.StatusBarAlignment.Right, 1000);
        this.setStatusDefault();
        this.item.show();
        this.item.command = "editPilot.showCommands";
        this.busy = false;

        this.register(
            vscode.commands.registerCommand("editPilot.showCommands", () => { })
        )
    }

    setItemText(iconId, text) {
        this.item.text = `\$\(${iconId}\) ${text}`;
    }

    setStatusDefault(quite = false) {
        if (quite && this.busy) return;
        this.busy = false;
        const iconId = isLanguageSupported() ? "edit" : "circle-slash";
        this.setItemText(iconId, "Edit Pilot");
        this.item.backgroundColor = isLanguageSupported()
            ? undefined
            : new vscode.ThemeColor('statusBarItem.warningBackground');
        this.item.tooltip = isLanguageSupported()
            ? "Edit Pilot is ready 🛫"
            : "Edit Pilot doesn't support this language yet 💤";
    } 

    setStatusLoadingFiles() {
        this.busy = true;
        this.setItemText(this.loadingIconId, "Loading files...");
        this.item.backgroundColor = undefined;
        this.item.tooltip = "Edit Pilot is working on local files 🔍";
    }

    setStatusQuerying(modelName) {
        this.busy = true;
        this.setItemText(this.loadingIconId, `Querying ${modelName}...`);
        this.item.backgroundColor = undefined;
        this.item.tooltip = "Edit Pilot is using language model to analyze 🔬";
    }

    setStatustProblem(errorMessage) {
        this.busy = true;
        this.setItemText("close", "Edit Pilot");
        this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
        this.item.tooltip = errorMessage;
    }
}

export const statusBarItem = new ProgressDisplayStatusBarItem();