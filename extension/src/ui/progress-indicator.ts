import vscode from 'vscode';
import { DisposableComponent } from '../utils/base-component';
import { globalEditorState } from '../global-workspace-context';

class ProgressDisplayStatusBarItem extends DisposableComponent {
    item: vscode.StatusBarItem;
    loadingIconId: string;
    busy: boolean;

    constructor() {
        super();
        this.loadingIconId = "loading~spin";
        this.item = vscode.window.createStatusBarItem('coEdPilot.progressDisplay', vscode.StatusBarAlignment.Right, 1000);
        this.setStatusDefault();
        this.item.show();
        this.item.command = "coEdPilot.showCommands";
        this.busy = false;

        this.register(
            vscode.commands.registerCommand("coEdPilot.showCommands", () => {
                // TODO showing a command context menu, to be implemented
            })
        );
    }

    setItemText(iconId: string, text: string) {
        this.item.text = `\$\(${iconId}\) ${text}`;
    }

    setStatusDefault(quite = false) {
        if (quite && this.busy) return;
        this.busy = false;
        let iconId: string;
        if (globalEditorState.isActiveEditorLanguageSupported()) {
            iconId = "edit";
            this.item.backgroundColor = undefined;
            this.item.tooltip = "CoEdPilot is ready 🛫";
        } else {
            iconId = "circle-slash";
            this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            this.item.tooltip = "CoEdPilot doesn't support this language yet 💤";
        }
        this.setItemText(iconId, "CoEdPilot");
    } 

    setStatusLoadingFiles() {
        this.busy = true;
        this.setItemText(this.loadingIconId, "Loading files...");
        this.item.backgroundColor = undefined;
        this.item.tooltip = "CoEdPilot is working on local files 🔍";
    }

    setStatusQuerying(modelName: string) {
        this.busy = true;
        this.setItemText(this.loadingIconId, `Querying ${modelName}...`);
        this.item.backgroundColor = undefined;
        this.item.tooltip = "CoEdPilot is using language model to analyze 🔬";
    }

    setStatustProblem(errorMessage: string) {
        this.busy = true;
        this.setItemText("close", "CoEdPilot");
        this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
        this.item.tooltip = errorMessage;
    }
}

export const statusBarItem = new ProgressDisplayStatusBarItem();