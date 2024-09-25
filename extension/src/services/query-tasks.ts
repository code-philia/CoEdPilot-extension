import vscode from 'vscode';
import { getRootPath, readGlobFiles, updatePrevEdits, toPosixPath } from '../utils/file-utils';
import { globalQueryContext } from '../global-result-context';
import { globalEditorState } from '../global-workspace-context';
import { globalEditDetector } from '../editor-state-monitor';
import { startLocationQueryProcess, startEditQueryProcess } from './query-processes';
import { DisposableComponent } from '../utils/base-component';
import { EditSelector, diffTabSelectors, tempWrite } from '../views/compare-view';
import { globalEditLock } from '../global-result-context';
import { statusBarItem } from '../ui/progress-indicator';
import { EditType } from '../utils/base-types';

async function predictLocation() {
    if (!globalEditorState.isActiveEditorLanguageSupported()) {
        vscode.window.showInformationMessage(`Predicting location canceled: language ${globalEditorState.language} not supported yet.`);
        return;
    }
    return await globalEditLock.tryWithLock(async () => {
        const commitMessage = await globalQueryContext.querySettings.requireCommitMessage();
        if (commitMessage === undefined) return;

        statusBarItem.setStatusLoadingFiles();
        const rootPath = getRootPath();
        const files = await readGlobFiles();
        // const currentPrevEdits = getPrevEdits();
        try {
            console.log("++++++++++++ getting updates")
            const currentPrevEdits = await globalEditDetector.getUpdatedEditList();
            statusBarItem.setStatusQuerying("locator");
            await startLocationQueryProcess(rootPath, files, currentPrevEdits, commitMessage, globalEditorState.language);
            statusBarItem.setStatusDefault();
        } catch (err) {
            console.error(err);
            vscode.window.showErrorMessage("Oops! Something went wrong with the query request 😦")
            statusBarItem.setStatustProblem("Some error occured when predicting locations");
            throw err;
        }
    });
}

async function predictLocationIfHasEditAtSelectedLine(event: vscode.TextEditorSelectionChangeEvent) {
    const hasNewEdits = updatePrevEdits(event.selections[0].active.line);
    if (hasNewEdits) {
        await predictLocation();
    }
}

async function predictEdit() {
    if (!globalEditorState.isActiveEditorLanguageSupported()) {
        vscode.window.showInformationMessage(`Predicting edit canceled: language ${globalEditorState.language} not supported yet.`);
        return;
    }
    
    const commitMessage = await globalQueryContext.querySettings.requireCommitMessage();
    if (commitMessage === undefined) return;
    
    const activeEditor = vscode.window.activeTextEditor;
    const activeDocument = activeEditor?.document;
    if (!(activeEditor && activeDocument)) return;
    if (activeDocument.uri.scheme !== "file") return;

    statusBarItem.setStatusLoadingFiles();
    const uri = activeDocument.uri;
    const atLines = [];
    const selectedRange = activeEditor.selection;
    
    const fromLine = selectedRange.start.line;
    let toLine = selectedRange.end.line;
    let editType: EditType;

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
    
    statusBarItem.setStatusQuerying("generator");
    try {
        const queryResult = await startEditQueryProcess(
            targetFileContent,
            editType,
            atLines,
            await globalEditDetector.getUpdatedEditList(),
            commitMessage,
            globalEditorState.language
        );
        
        // Remove syntax-level unchanged replacements
        queryResult.replacement = queryResult.replacement.filter((snippet: string) => snippet.trim() !== selectedContent.trim());

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
        vscode.window.showErrorMessage("Oops! Something went wrong with the query request 😦")
        statusBarItem.setStatustProblem("Some error occured when predicting edits");
        throw err;
    }
}

class PredictLocationCommand extends DisposableComponent{
	constructor() {
		super();
		this.register(
            vscode.commands.registerCommand("coEdPilot.predictLocations", predictLocation),
            vscode.commands.registerCommand("coEdPilot.clearLocations", async () => {
                globalQueryContext.updateLocations(undefined);
            })
		);
	}
}

class GenerateEditCommand extends DisposableComponent{
	constructor() {
		super();
        this.register(
            this.registerEditSelectionCommands(),
            vscode.commands.registerCommand("coEdPilot.generateEdits", predictEdit)
		);
    }
    
    registerEditSelectionCommands() {
        function getSelectorOfCurrentTab() {
            const currTab = vscode.window.tabGroups?.activeTabGroup?.activeTab;
            if (currTab && currTab.input instanceof vscode.TabInputTextDiff) {
                const selector = diffTabSelectors.get(currTab.input.modified.toString());
                if (selector) {
                    selector.manuallyEdited = false;
                }
                return selector;
            }
            return undefined;
        }
        async function switchEdit(offset: number) {
            const selector = getSelectorOfCurrentTab();
            selector && await selector.switchEdit(offset);
        }
        async function closeTab() {
            const tabGroups = vscode.window.tabGroups;
            const activeTab = tabGroups.activeTabGroup.activeTab;
            if (activeTab) {
                await tabGroups.close(tabGroups.activeTabGroup.activeTab, true);
            }
        }
        async function clearEdit() {
            const selector = getSelectorOfCurrentTab();
            selector && await selector.clearEdit();
        }
        async function acceptEdit() {
            const selector = getSelectorOfCurrentTab();
            selector && await selector.acceptEdit();
        }
        return vscode.Disposable.from(
            vscode.commands.registerCommand("coEdPilot.lastSuggestion", async () => {
                await switchEdit(-1);
            }),
            vscode.commands.registerCommand("coEdPilot.nextSuggestion", async () => {
                await switchEdit(1);
            }),
            vscode.commands.registerCommand("coEdPilot.acceptEdit", async () => {
                await acceptEdit();
                await closeTab();
                globalEditorState.toPredictLocation = true;
            }),
            vscode.commands.registerCommand("coEdPilot.dismissEdit", async () => {
                await clearEdit();
                await closeTab();
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
