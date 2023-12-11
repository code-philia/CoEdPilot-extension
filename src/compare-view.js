import vscode from 'vscode';
import crypto from 'crypto';
import util from 'util';
import path from 'path';
import { BaseComponent } from './base-component';
import { toPosixPath } from './file';
import { defaultLineBreak, editorState, queryState } from './global-context';

class BaseTempFileProvider extends BaseComponent {
    constructor() {
        super();
        this._onDidChangeFile = new vscode.EventEmitter();
        this.onDidChangeFile = this._onDidChangeFile.event;
    }

    async readFile(uri) { return new Uint8Array(); }

    // The following member functions are to complement the vscode.FileSystemProvider interface

    async stat(uri) {
        return {
            type: vscode.FileType.File,
            ctime: Date.now(),
            mtime: Date.now(),
            size: 0,
            name: uri.fsPath
        }
    }

    watch(uri, options) { return { dispose: () => { } }; }

    async readDirectory(uri) { return []; }

    async createDirectory(uri) { }

    async writeFile(uri, content, options) { }

    async delete(uri, options) { }

    async rename(oldUri, newUri, options) { }

    async copy(source, destination, options) { }

}

class CompareTempFileProvider extends BaseTempFileProvider { // impletements vscode.FileSystemProvider
    constructor() {
        super();
        this.tempFiles = new Map();

        this.register(
            vscode.workspace.registerFileSystemProvider("temp", this, { isReadonly: true })
        );
    }

    async writeFile(uri, content, options) {
        this.tempFiles.set(uri.path, content);
    }

    async readFile(uri) {
        return this.tempFiles.get(uri.path);
    }

    getAsyncWriter() {
        return (async (path, str) => {
            const encoder = new util.TextEncoder();
            return this.writeFile(vscode.Uri.parse(`temp:${path}`), encoder.encode(str));
        }).bind(this);
    }
}

const compareTempFileSystemProvider = new CompareTempFileProvider();
const tempWrite = compareTempFileSystemProvider.getAsyncWriter();
const diffTabSelectors = new Map();

/**
 * Use a series of suggested edits to generate a live editable diff view for the user to make the decision
 */
class EditSelector {
    constructor(path, fromLine, toLine, edits, originWrite, isAdd = false) {
        this.path = path;
        this.fromLine = fromLine;
        this.toLine = toLine;  // toLine is exclusive
        this.edits = edits;
        this.tempWrite = originWrite ?? tempWrite;
        this.isAdd = isAdd;

        this.originalContent = "";
        this.modAt = 0;
    }

    async init() {
        // Save the original content
        this.document = await vscode.workspace.openTextDocument(this.path);
        this.originalContent = this.document.getText();

        // Store the originalContent in a temporary readonly file system
        this.id = this._getPathId();
        this.tempWrite(
            `/${this.id}`,
            this.originalContent
        );
    }

    /**
     * Find the editor where the document is open then change its 
     * @param {*} replacement 
     */
    async _performMod(replacement) {
        const x = this.originalContent.match(/\r?\n|\n/);
        const lineBreak = x?.at(0) ?? defaultLineBreak;
        const lines = this.originalContent.split(lineBreak);
        const numLines = lines.length + 1;
        const fromLine = Math.max(0, this.fromLine);
        // If change type is "add", simply insert replacement content at the first line 
        const toLine = this.isAdd ? fromLine : Math.min(this.toLine, numLines);
        
        const modifiedText = (lines.slice(0, fromLine)).join(lineBreak)
            + (fromLine > 0 ? lineBreak : '')
            + replacement
            + (toLine < numLines ? lineBreak : '')
            + (lines.slice(toLine, numLines)).join(lineBreak);
        
        this._replaceDocument(modifiedText);
    }

    async _replaceDocument(fullText) {
        const editor = vscode.window.visibleTextEditors.find(
            (editor) => editor.document === this.document
        );
        
        const fullRange = new vscode.Range(
            this.document.positionAt(0),
            this.document.positionAt(this.document.getText().length)
        );
        
        await editor.edit(editBuilder => {
            editBuilder.replace(fullRange, fullText)
        }, { undoStopBefore: false, undoStopAfter: false });
    }

    async _showDiffView() {
        // Open a diff view to compare the original and the modified document
        await vscode.commands.executeCommand('vscode.diff',
            vscode.Uri.parse(`temp:/${this.id}`),
            vscode.Uri.file(this.path),
            `EDIT ${this.modAt+1}/${this.edits.length}: ${path.basename(this.path)}`
        );

        const tabGroups = vscode.window.tabGroups;
        const activeTab = tabGroups.activeTabGroup.activeTab;
        diffTabSelectors.set(activeTab, this);
        let removeSelectorEvent = null;
        const event = tabGroups.onDidChangeTabs((e) => {
            if (e.closed.includes(activeTab)) {
                diffTabSelectors.delete(activeTab);
                removeSelectorEvent.dispose();
            }
        });
        removeSelectorEvent = event;
    }

    async editDocumentAndShowDiff() {
        await this._performMod(this.edits[this.modAt]);
        if (editorState.inDiffEditor) {     // refresh existed diff editor
            await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
        }
        await this._showDiffView();
    }

    async switchEdit(offset = 1) {
        this.modAt = (this.modAt + offset + this.edits.length) % this.edits.length;
        await this.editDocumentAndShowDiff();
    }

    async clearEdit() {
        // await vscode.commands.executeCommand('undo');
        await this._replaceDocument(this.originalContent);
    }

    async acceptEdit() {
        const locations = queryState.locations;
        locations.forEach((loc, i) => {
            const offset = loc.editType === "add" ? 1 : 0;
            if (loc.atLines
            && loc.atLines[0] + offset < this.toLine
            && loc.atLines[loc.atLines.length - 1] + 1 + offset > this.fromLine) {
                locations.splice(i, 1);
            }
            queryState._onDidChangeLocations.fire();
        })
    }

    _getPathId() {
        this.pathId = crypto.createHash('sha256').update(this.path).digest('hex') + path.extname(this.path);
        return this.pathId;
    }
}

class DiffTabCodelensProvider extends BaseComponent {
    constructor() {
        super();
        this.originalContentLabel = "Original";
        this.modifiedContentLabel = "Modified";
        this._onDidChangeCodeLenses = new vscode.EventEmitter();
	    this.onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;
        this.register(
            // vscode.languages.registerCodeLensProvider("*", this),
            vscode.window.onDidChangeActiveTextEditor(() => {
                console.log("+++ firing code lenses");
                this._onDidChangeCodeLenses.fire();
            })
        );
    }
    
    provideCodeLenses(document, token) {
        this.codelenses = [];
        if (document.uri.scheme === 'temp') {
            this.codelenses.push(this.codelenseAtTop(this.originalContentLabel));
        }
        else if (document.uri.scheme === 'file') {
            for (const [tab, selector] of diffTabSelectors) {
                if (selector.path == toPosixPath(document.path)) {
                    this.codelenses.push(this.codelenseAtTop(this.modifiedContentLabel));
                    break;
                }
            }
        }
        return this.codelenses;
    }

    resolveCodeLens(codeLens, token) {
        return codeLens;
    }

    codelenseAtTop(title) {
        return new vscode.CodeLens(
            new vscode.Range(0, 0, 0, 0),
            {
                title: title
            }
        )
    }
}

export {
    EditSelector,
    CompareTempFileProvider,
    diffTabSelectors,
    compareTempFileSystemProvider,
    tempWrite,
    DiffTabCodelensProvider
};
