const vscode = require('vscode');
const path = require('path');
const crypto = require('crypto')
class CompareTempFileProvider { // impletements vscode.FileSystemProvider

    constructor(){
        this.tempFiles = new Map();
    }

    onDidChangeFile() {
        // do nothing
    }

    storeTempFile(rel_path, content) {
        const encoder = new TextEncoder();
        this.tempFiles[rel_path] = encoder.encode(content);
    }

    readFile(uri) {
        console.log(`Temp file reading: ${uri.path}`);
        return this.tempFiles[uri.path];
    }

    // The following member functions are to complement the vscode.FileSystemProvider interface

    stat(uri) {
        // do nothing
        console.log(`Looking up stat: ${uri}`);
        return {
            type: vscode.FileType.File,
            ctime: Date.now(),
            mtime: Date.now(),
            size: 0,
            name: uri.path
        }
    }

    watch(uri, options) {
        // do nothing
        return new vscode.Disposable(() => { });
    }


    readDirectory(uri) {
        console.log(`Reading directory: ${uri}`);
        // do nothing
    }

    createDirectory(uri) {
        // do nothing
    }

    writeFile(uri, content, options) {
        // do nothing
    }

    delete(uri, options) {
        // do nothing
    }

    rename(oldUri, newUri, options) {
        // do nothing
    }

    copy(source, destination, options) {
        // do nothing
    }
}

const globalTempFileProvider = new CompareTempFileProvider();

const globalDiffTabSelectors = {};

/**
 * Use a series of suggested edits to generate a live editable diff view for the user to make the decision
 */
class EditSelector {
    constructor(path, startPos, endPos, edits) {
        this.path = path;
        this.startPos = startPos;
        this.endPos = endPos;
        this.edits = edits;

        this.originalContent = "";
        this.modAt = 0;
    }

    async init() {
        // Save the original content
        this.document = await vscode.workspace.openTextDocument(this.path);
        this.originalContent = this.document.getText();

        // Store the originalContent in a temporary readonly file system
        this.id = this._getPathId();
        globalTempFileProvider.storeTempFile(`/${this.id}`, this.originalContent);
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
        // await this._showDiffView();
    }

    _getPathId() {
        return crypto.createHash('sha256').update(this.path).digest('hex');
    }
}



module.exports = {
    EditSelector,
    globalTempFileProvider,
    globalDiffTabSelectors
};
