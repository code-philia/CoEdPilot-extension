import vscode from "vscode";

export class BaseComponent {
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

export function registerCommand(command: string, callback: (...args: any[]) => any, thisArg?: any) {
    return vscode.commands.registerCommand(command, callback, thisArg);
}

export function numIn(x: number, lower: number, upper: number) {
	if (x <= lower)
		x = lower;
	if (x >= upper)
		x = upper;
	return x;
}