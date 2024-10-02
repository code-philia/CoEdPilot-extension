import vscode from 'vscode';
import { getRootPath, readGlobFiles, updatePrevEdits, toPosixPath, globalEditDetector } from './file';
import { editorState, isLanguageSupported, queryState } from './global-context';
import { queryLocationFromModel, queryEditFromModel } from './queries';
import { BaseComponent } from './base-component';
import { EditSelector, diffTabSelectors, tempWrite } from './compare-view';
import { registerCommand } from "./base-component";
import { globalEditLock } from './global-context';
import { statusBarItem } from './status-bar';

async function predictLocation() {
    if (!isLanguageSupported()) {
        vscode.window.showInformationMessage(`Predicting location canceled: language ${editorState.language} not supported yet.`);
        return;
    }
    return await globalEditLock.tryWithLockAsync(async () => {
        const commitMessage = await queryState.requireCommitMessage();
        if (commitMessage === undefined) return;

        statusBarItem.setStatusLoadingFiles();
        const rootPath = getRootPath();
        const files = await readGlobFiles();
        // const currentPrevEdits = getPrevEdits();
        try {
            console.log("++++++++++++ getting updates")
            const currentPrevEdits = await globalEditDetector.getUpdatedEditList();
            statusBarItem.setStatusQuerying("locator");
            await queryLocationFromModel(rootPath, files, currentPrevEdits, commitMessage, editorState.language);
            statusBarItem.setStatusDefault();
        } catch (err) {
            console.error(err);
            statusBarItem.setStatusProblem("Some error occured when predicting locations");
            throw err;
        }
    });
}

async function predictLocationIfHasEditAtSelectedLine(event) {
    const hasNewEdits = updatePrevEdits(event.selections[0].active.line);
    if (hasNewEdits) {
        await predictLocation();
    }
}

async function predictEdit() {
    if (!isLanguageSupported()) {
        vscode.window.showInformationMessage(`Predicting edit canceled: language ${editorState.language} not supported yet.`);
        return;
    }
    
    const commitMessage = await queryState.requireCommitMessage();
    if (commitMessage === undefined) return;
    
    const activeEditor = vscode.window.activeTextEditor;
    const activeDocument = activeEditor?.document;
    if (!(activeEditor && activeDocument)) return;
    if (activeDocument.uri.scheme !== "file") return;

    statusBarItem.setStatusLoadingFiles();
    const uri = activeDocument.uri;
    const filePath = toPosixPath(uri.fsPath);
    const atLines = [];
    const selectedRange = activeEditor.selection;
    
    const fromLine = selectedRange.start.line;
    let toLine = selectedRange.end.line;
    let editType = "";

    // NOTE only starting line of the selection is used for matching
    const locationIncludesLine = (loc, line) => {
        return (loc.editType === "replace" && loc.atLines.includes(line))
            || (loc.editType === "add" && loc.atLines.includes(line - 1));
    }
    const firstMatchLocation = queryState.locations
        .find(loc => loc.targetFilePath === filePath
            && locationIncludesLine(loc, fromLine));
    
    if (firstMatchLocation) {
        editType = firstMatchLocation.editType;
        atLines.push(...firstMatchLocation.atLines);
    } else if (selectedRange.isEmpty) {
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
    
    statusBarItem.setStatusQuerying("generator");
    try {
        const queryResult = await queryEditFromModel(
            targetFileContent,
            editType,
            atLines,
            await globalEditDetector.getUpdatedEditList(),
            commitMessage,
            editorState.language
        );
        
        // Remove syntax-level unchanged replacements
        queryResult.replacement = queryResult.replacement.filter((snippet) => snippet.trim() !== selectedContent.trim());

        const selector = new EditSelector(
            toPosixPath(uri.fsPath),
            fromLine,
            toLine+1,
            queryResult.replacement,
            tempWrite,
            editType == "add"
        );
        await selector.init();
        await selector.editDocumentAndShowDiff();
        statusBarItem.setStatusDefault();
    } catch (err) {
        console.error(err);
        statusBarItem.setStatusProblem("Some error occured when predicting edits");
        throw err;
    }
}

class PredictLocationCommand extends BaseComponent{
	constructor() {
		super();
		this.register(
            vscode.commands.registerCommand("coEdPilot.predictLocations", predictLocation),
            vscode.commands.registerCommand("coEdPilot.clearLocations", async () => {
                await queryState.clearLocations();
            })
		);
	}
}

class GenerateEditCommand extends BaseComponent{
	constructor() {
		super();
        this.register(
            this.registerEditSelectionCommands(),
            vscode.commands.registerCommand("coEdPilot.generateEdits", predictEdit)
		);
    }
    
    registerEditSelectionCommands() {
        function getSelectorOfCurrentTab() {
            const currTab = vscode.window.tabGroups.activeTabGroup.activeTab;
            const selector = diffTabSelectors.get(currTab);
            return selector;
        }
        function switchEdit(offset) {
            const selector = getSelectorOfCurrentTab();
            selector && selector.switchEdit(offset);
        }
        function closeTab() {
            const tabGroups = vscode.window.tabGroups;
            tabGroups.close(tabGroups.activeTabGroup.activeTab, true);
        }
        function clearEdit() {
            const selector = getSelectorOfCurrentTab();
            selector && selector.clearEdit();
        }
        function acceptEdit() {
            const selector = getSelectorOfCurrentTab();
            selector && selector.acceptEdit();
        }
        return vscode.Disposable.from(
            registerCommand("coEdPilot.last-suggestion", () => {
                switchEdit(-1);
            }),
            registerCommand("coEdPilot.next-suggestion", () => {
                switchEdit(1);
            }),
            registerCommand("coEdPilot.accept-edit", () => {
                acceptEdit();
                closeTab();
                editorState.toPredictLocation = true;
            }),
            registerCommand("coEdPilot.dismiss-edit", () => {
                clearEdit();
                closeTab();
            })
        )
    }
}

export {
    predictLocation,
    predictLocationIfHasEditAtSelectedLine,
    PredictLocationCommand,
    GenerateEditCommand
};
