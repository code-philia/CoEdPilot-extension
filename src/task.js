import vscode from 'vscode';
import { getRootPath, getFiles, updatePrevEdits, getPrevEdits, getLocationAtRange } from './file';
import { queryLocationFromModel, queryEditFromModel, queryState } from './query';


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
            throw err;
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
            throw err;
        } finally {
            this.isLocked = false;
        }
    }
}

const globalEditLock = new EditLock();

async function predictLocation() {
    return await globalEditLock.tryWithLockAsync(async () => {
        console.log('==> Send to LLM (After cursor changed line)');
        const rootPath = getRootPath();
        const files = getFiles();
        const currentPrevEdits = getPrevEdits();
        await queryLocationFromModel(rootPath, files, currentPrevEdits, queryState.commitMessage);
    });
}

async function predictLocationIfHasEditAtSelectedLine(event) {
    const hasNewEdits = updatePrevEdits(event.selections[0].active.line);
    if (hasNewEdits) {
        await predictLocation();
    }
}

async function predictEdit(document, location) {
    return await globalEditLock.tryWithLockAsync(async () => {
        const predictResult = await queryEditFromModel(
            document.getText(),
            location,
            queryState.commitMessage
        );
        const replacedRange = new vscode.Range(document.positionAt(location.startPos), document.positionAt(location.endPos));
        const replacedContent = document.getText(replacedRange).trim();
        predictResult.replacement = predictResult.replacement.filter((snippet) => snippet.trim() !== replacedContent);
        return predictResult;
    });
}

async function predictEditAtRange(document, range) {
    const targetLocation = getLocationAtRange(queryState.locations, document, range);    
    if (targetLocation) {
        return predictEdit(document, targetLocation)
    } 
    return undefined;
}

export {
    predictLocation,
    predictLocationIfHasEditAtSelectedLine,
    predictEdit,
    predictEditAtRange
};