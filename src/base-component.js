const vscode = require('vscode');

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

module.exports = {
    BaseComponent
};