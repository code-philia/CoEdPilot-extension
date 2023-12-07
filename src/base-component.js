import vscode from "vscode";

export class BaseComponent {
    constructor() {
        this._disposables = [];
    }

    dispose() {
        this._disposables.forEach((e) => e.dispose());
    }

    register(...disposables) {
        this._disposables.push(...disposables);
    }
}

export function registerCommand(command, callback, thisArg) {
    return vscode.commands.registerCommand(command, callback, thisArg);
}
