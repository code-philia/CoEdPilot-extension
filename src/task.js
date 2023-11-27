const vscode = require('vscode');
const { getRootPath, getFiles, updatePrevEdits, getPrevEdits, getEditAtRange, } = require('./file');
const { queryLocationFromModel, queryEditFromModel, queryState } = require('./query');


class EditLock {
    constructor() {
        this.isLocked = false;
    }

    tryWithLock(callback) {
        if (this.isLocked) return undefined;
        this.isLocked = true;
        try {
            return callback();
        } catch (err) {
            console.log(`Error occured when running in edit lock: \n${err}`);
            return undefined;
        } finally {
            this.isLocked = false;
        }
    }

    async tryWithLockAsync(asyncCallback) {
        if (this.isLocked) return undefined;
        this.isLocked = true;
        try {
            return await asyncCallback();
        } catch (err) {
            console.log(`Error occured when running in edit lock (async): \n${err}`);
            return undefined;
        } finally {
            this.isLocked = false;
        }
    }
}

const globalEditLock = new EditLock();
Object.freeze(global);

async function predictLocationAtEdit(event) {
    return await globalEditLock.tryWithLockAsync(async () => {
        console.log('==> Send to LLM (After cursor changed line)');
        const rootPath = getRootPath();
        const files = getFiles();
        const hasNewEdits = updatePrevEdits(event.selections[0].active.line);
        if (hasNewEdits) {
            const currentPrevEdits = getPrevEdits();
            await queryLocationFromModel(rootPath, files, currentPrevEdits, queryState.commitMessage);
        }
    });
}

// When the user adopts the suggestion of QuickFix, 
// the modified version is immediately sent to LLM to 
// update modifications without waiting for the pointer to change lines
async function predictAfterQuickFix() {
    return await globalEditLock.tryWithLockAsync(async () => {
        if (updatePrevEdits()) {
            const rootPath = getRootPath();
            const files = getFiles();
            await queryLocationFromModel(rootPath, files, getPrevEdits(), queryState.commitMessage);
        }
    })
}

async function predictEditAtRange(document, range) {
    return await globalEditLock.tryWithLockAsync(async () => {
        const targetMod = getEditAtRange(queryState.locations, document, range);
        if (targetMod) {
            const replacedRange = new vscode.Range(document.positionAt(targetMod.startPos), document.positionAt(targetMod.endPos));
            const replacedContent = document.getText(replacedRange).trim();
            const predictResult = await queryEditFromModel(targetMod);
            predictResult.replacement = predictResult.replacement.filter((snippet) => snippet.trim() !== replacedContent);
            return predictResult;
        }
    });
}

module.exports = {
    predictLocationAtEdit,
    predictAfterQuickFix,
    predictEditAtRange
}