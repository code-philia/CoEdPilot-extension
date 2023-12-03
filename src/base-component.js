import vscode from 'vscode';

class BaseComponent {
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

export {
    BaseComponent
};