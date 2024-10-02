import vscode from 'vscode';
import { BaseComponent } from './base-component';
import { isLanguageSupported } from './global-context';

class ProgressDisplayStatusBarItem extends BaseComponent {
    constructor() {
        super();
        this.loadingIconId = "loading~spin";
        this.item = vscode.window.createStatusBarItem('coEdPilot.progressDisplay', vscode.StatusBarAlignment.Right, 1000);
        this.setStatusDefault();
        this.item.show();
        this.item.command = "coEdPilot.showCommands";
        this.busy = false;

        this.register(
            vscode.commands.registerCommand("coEdPilot.showCommands", () => { })
        )
    }

    setItemText(iconId, text) {
        this.item.text = `\$\(${iconId}\) ${text}`;
    }

    setStatusDefault(quite = false) {
        if (quite && this.busy) return;
        this.busy = false;
        const iconId = isLanguageSupported() ? "edit" : "circle-slash";
        this.setItemText(iconId, "CoEdPilot");
        this.item.backgroundColor = isLanguageSupported()
            ? undefined
            : new vscode.ThemeColor('statusBarItem.warningBackground');
        this.item.tooltip = isLanguageSupported()
            ? "CoEdPilot is ready 🛫"
            : "CoEdPilot doesn't support this language yet 💤";
    } 

    setStatusLoadingFiles() {
        this.busy = true;
        this.setItemText(this.loadingIconId, "Loading files...");
        this.item.backgroundColor = undefined;
        this.item.tooltip = "CoEdPilot is working on local files 🔍";
    }

    setStatusQuerying(modelName) {
        this.busy = true;
        this.setItemText(this.loadingIconId, `Querying ${modelName}...`);
        this.item.backgroundColor = undefined;
        this.item.tooltip = "CoEdPilot is using language model to analyze 🔬";
    }

    setStatusProblem(errorMessage) {
        this.busy = true;
        this.setItemText("close", "CoEdPilot");
        this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
        this.item.tooltip = errorMessage;
    }
}

export const statusBarItem = new ProgressDisplayStatusBarItem();