import vscode from 'vscode';
import { getRootPath, getFiles, updatePrevEdits, getPrevEdits, getLocationAtRange, fileState, toPosixPath } from './file';
import { queryLocationFromModel, queryEditFromModel, queryState } from './queries';
import { BaseComponent } from './base-component';
import { EditSelector, diffTabSelectors, tempWrite } from './compare-view';
import { registerCommand } from './extension-register';

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
        const files = await getFiles();
        const currentPrevEdits = getPrevEdits();
        try {
            await queryLocationFromModel(rootPath, files, currentPrevEdits, queryState.commitMessage);
        } catch (err) {
            console.log(err);
        }
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
            location.editType,
            location.atLines,
            fileState.prevEdits,
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

class PredictLocationCommand extends BaseComponent{
	constructor() {
		super();
		this.register(
			vscode.commands.registerCommand("editPilot.predictLocations", () => { predictLocation(); })
		);
	}
}

class GenerateEditCommand extends BaseComponent{
	constructor() {
		super();
        this.register(
            this.registerEditSelectionCommands(),
			vscode.commands.registerCommand("editPilot.generateEdits", async (...args) => {
				if (args.length != 1 || !(args[0] instanceof vscode.Uri)) return;
				
				const uri = args[0];
				const activeEditor = vscode.window.activeTextEditor;
				const activeDocument = activeEditor.document;
				if (activeDocument.uri.toString() !== uri.toString()) return;
                const atLines = [];
                const selectedRange = activeEditor.selection;
                
                const fromLine = selectedRange.start.line;
                let toLine = selectedRange.end.line;
                let editType = "";
                if (selectedRange.isEmpty) {
                    editType = "add";
                    atLines.push(fromLine);
                } else {
                    editType = "replace";
                    // If only the beginning of the last line is included, exclude the last line
                    if (selectedRange.end.character === 0) {
                        toLine -= 1;
                    }
                    for (let i = fromLine; i <= toLine; ++i) {
                        atLines.push(i);
                    }
                }
                
                const targetFileContent = activeDocument.getText();
                const selectedContent = activeDocument.getText(
                    new vscode.Range(
                        activeDocument.lineAt(fromLine).range.start,
                        activeDocument.lineAt(toLine).range.end
                    )
                );

                const queryResult = await queryEditFromModel(
                    targetFileContent,
                    editType,
                    atLines,
                    fileState.prevEdits,
                    queryState.commitMessage
                );
                
                // Remove syntax-level unchanged replacements
				queryResult.replacement = queryResult.replacement.filter((snippet) => snippet.trim() !== selectedContent.trim());
		
				try {
					const selector = new EditSelector(
						toPosixPath(uri.fsPath),
						fromLine,
						toLine+1,
                        queryResult.replacement,
                        tempWrite,
                        editType == "add"
					);
					await selector.init();
					await selector.editedDocumentAndShowDiff();
				} catch (err) {
					console.log(err);
				}
			})
		);
    }
    
    registerEditSelectionCommands() {
        function getSelectorOfCurrentTab() {
            const currTab = vscode.window.tabGroups.activeTabGroup.activeTab;
            const selector = diffTabSelectors[currTab];
            return selector;
        }
        function switchEdit(offset) {
            const selector = getSelectorOfCurrentTab();
            selector && selector.switchEdit(offset);
        }
        function clearEdit() {
            const selector = getSelectorOfCurrentTab();
            selector && selector.clearEdit();
        }
        function closeTab() {
            const tabGroups = vscode.window.tabGroups;
            tabGroups.close(tabGroups.activeTabGroup.activeTab, true);
        }
        return vscode.Disposable.from(
            registerCommand("editPilot.last-suggestion", () => {
                switchEdit(-1);
            }),
            registerCommand("editPilot.next-suggestion", () => {
                switchEdit(1);
            }),
            registerCommand("editPilot.accept-edit", () => {
                closeTab();
            }),
            registerCommand("editPilot.dismiss-edit", () => {
                clearEdit();
                closeTab();
            })
        )
    }
}

export {
    predictLocation,
    predictLocationIfHasEditAtSelectedLine,
    predictEdit,
    predictEditAtRange,
    PredictLocationCommand,
    GenerateEditCommand
};
