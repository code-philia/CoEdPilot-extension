import vscode from 'vscode';

class BaseComponent {
    constructor() {
        this.disposable = {
            dispose: () => {}
        };
    }

    dispose() {
        this.disposable.dispose();
    }

    register(...disposables) {
        this.disposable = vscode.Disposable.from(...disposables);
    }
}

export {
    BaseComponent
};