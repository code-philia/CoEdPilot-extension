import vscode from 'vscode';
import crypto from 'crypto';
import util from 'util';
import { BaseComponent } from './base-component';

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
            name: uri.path
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
        this.tempFiles[uri.path] = content;
    }

    async readFile(uri) {
        return this.tempFiles[uri.path];
    }

    getAsyncWriter() {
        return (async (path, str) => {
            const encoder = new util.TextEncoder();
            return this.writeFile(vscode.Uri.parse(`temp:${path}`), encoder.encode(str));
        }).bind(this);
    }
}

const globalCompareTempFileProvider = new CompareTempFileProvider();
const globalTempWrite = globalCompareTempFileProvider.getAsyncWriter();
const globalDiffTabSelectors = {};

/**
 * Use a series of suggested edits to generate a live editable diff view for the user to make the decision
 */
class EditSelector {
    constructor(path, startPos, endPos, edits, tempWrite) {
        this.path = path;
        this.startPos = startPos;
        this.endPos = endPos;
        this.edits = edits;
        this.tempWrite = tempWrite;

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
        const editor = vscode.window.visibleTextEditors.find(
            (editor) => editor.document === this.document
        );

        const fullRange = new vscode.Range(
            this.document.positionAt(0),
            this.document.positionAt(this.document.getText().length)
        );
        const modifiedText = this.originalContent.substring(0, this.startPos) +
            replacement +
            this.originalContent.substring(this.endPos);

        await editor.edit(editBuilder => {
            editBuilder.replace(fullRange, modifiedText)
        });
    }

    async _showDiffView() {
        // Open a diff view to compare the original and the modified document
        await vscode.commands.executeCommand('vscode.diff',
            vscode.Uri.parse(`temp:/${this.id}`),
            vscode.Uri.file(this.path),
            "Original vs. Modified"
        );
        globalDiffTabSelectors[vscode.window.tabGroups.activeTabGroup.activeTab] = this;
        // await vscode.commands.executeCommand('moveActiveEditor', {
        //     to: 'right',
        //     by: 'group'
        // });
    }

    async editedDocumentAndShowDiff() {
        await this._performMod(this.edits[this.modAt]);
        await this._showDiffView();
    }

    async switchEdit(offset = 1) {
        this.modAt = (this.modAt + offset) % this.edits.length;
        await this._performMod(this.edits[this.modAt]);
        await this._showDiffView();
    }

    _getPathId() {
        return crypto.createHash('sha256').update(this.path).digest('hex');
    }
}

export {
    EditSelector,
    CompareTempFileProvider,
    globalDiffTabSelectors,
    globalCompareTempFileProvider,
    globalTempWrite
};
