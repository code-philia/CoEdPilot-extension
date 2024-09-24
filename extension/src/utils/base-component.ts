import vscode from "vscode";

export class DisposableComponent implements vscode.Disposable {
    private _disposables: vscode.Disposable[];

    constructor() {
        this._disposables = [];
    }

    dispose() {
        this._disposables.forEach((e) => e.dispose());
    }

    register(...disposables: vscode.Disposable[]) {
        this._disposables.push(...disposables);
    }
}
