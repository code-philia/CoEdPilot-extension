import * as vscode from 'vscode';

// TODO implement this as vscode event
export class liveTextEditorEventHandler {
    private readonly callback: (editor: vscode.TextEditor | undefined) => any;
    private readonly disposable: vscode.Disposable;
    private readonly disposeCallback: (editor: vscode.TextEditor | undefined) => any;

    constructor(
        callback: (editor: vscode.TextEditor | undefined) => any,
        disposeCallback: (editor: vscode.TextEditor | undefined) => any,
        thisArg?: any
    ) {
        this.callback = callback.bind(thisArg);
        this.disposeCallback = disposeCallback.bind(thisArg);

        this.disposable = vscode.Disposable.from(
            vscode.window.onDidChangeActiveTextEditor(this.callback)
        );
    }
    
    handle(editor: vscode.TextEditor | undefined) {
        // NOTE callback is asynchronously called here, no order is promised
        this.callback(editor);
    }

    dispose() {
        this.disposable.dispose();
        this.disposeCallback(vscode.window.activeTextEditor);
    }
}
