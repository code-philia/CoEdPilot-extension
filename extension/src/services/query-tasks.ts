import vscode from 'vscode';
import { getRootPath, updatePrevEdits, toPosixPath, readGlobFiles } from '../utils/file-utils';
import { globalQueryContext, globalEditLock } from '../global-result-context';
import { globalEditorState } from '../global-workspace-context';
import { globalEditDetector } from '../editor-state-monitor';
import { requestAndUpdateLocation, requestAndUpdateEdit, requestAndUpdateLocationByNavEdit } from './query-processes';
import { DisposableComponent } from '../utils/base-component';
import { EditSelector, diffTabSelectors, tempWrite } from '../views/compare-view';
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
        // wait 1 second
        await new Promise((resolve) => setTimeout(resolve, 300));
        const files: [string, string][] = [];
        try {
            const currentPrevEdits = await globalEditDetector.getUpdatedSimpleEditList();
            statusBarItem.setStatusQuerying("locator");
            // TODO depart this step, because it is not parallel to other steps
            await requestAndUpdateLocation(rootPath, files, currentPrevEdits, commitMessage, globalEditorState.language);
            statusBarItem.setStatusDefault();
        } catch (err) {
            vscode.window.showErrorMessage("Oops! Something went wrong with the query request 😦");
            statusBarItem.setStatustProblem("Some error occured when predicting locations");
            throw err;
        }
    });
}

async function predictLocationByNavEdit() {
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
        try {
            const currentPrevEdits = await globalEditDetector.getUpdatedEditList();
            statusBarItem.setStatusQuerying("locator");
            // TODO depart this step, because it is not parallel to other steps
            await requestAndUpdateLocationByNavEdit(rootPath, files, currentPrevEdits, commitMessage, globalEditorState.language);
            statusBarItem.setStatusDefault();
        } catch (err) {
            vscode.window.showErrorMessage("Oops! Something went wrong with the query request 😦");
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

    // extract uri
    const uri = activeDocument.uri;

    // extract selected line numbers
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
        const queryResult = await requestAndUpdateEdit(
            targetFileContent,
            editType,
            atLines,
            await globalEditDetector.getUpdatedSimpleEditList(),
            commitMessage,
            globalEditorState.language
        );

        if (!queryResult) { return; }
        
        // Remove syntax-level unchanged replacements
        // TODO specify this step to a function
        queryResult.replacement = queryResult.replacement.filter((snippet: string) => snippet.trim() !== selectedContent.trim());

        const selector = new EditSelector(
            toPosixPath(uri.fsPath),
            fromLine,
            toLine+1,
            queryResult.replacement,
            tempWrite,
            false
        );
        await selector.init();
        await selector.editDocumentAndShowDiff();
        statusBarItem.setStatusDefault();
    } catch (err) {
        // TODO add a error logging channel to "Outputs"
        vscode.window.showErrorMessage("Oops! Something went wrong with the query request 😦");
        statusBarItem.setStatustProblem("Some error occured when predicting edits");
        throw err;
    }
}

class PredictLocationCommand extends DisposableComponent {
	constructor() {
		super();
		this.register(
            vscode.commands.registerCommand("coEdPilot.predictLocations", predictLocationByNavEdit),
            vscode.commands.registerCommand("coEdPilot.clearLocations", async () => {
                globalQueryContext.clearResults();
            })
		);
	}
}

class GenerateEditCommand extends DisposableComponent {
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
            if (selector) {
                await selector.acceptEdit();
            } else {
                globalQueryContext.applyRefactor();
            }
        }
        return vscode.Disposable.from(
            vscode.commands.registerCommand("coEdPilot.lastSuggestion", async () => {
                await switchEdit(-1);
            }),
            vscode.commands.registerCommand("coEdPilot.nextSuggestion", async () => {
                await switchEdit(1);
            }),
            vscode.commands.registerCommand("coEdPilot.acceptEdit", async () => {
                globalEditorState.toPredictLocation = true;
                await acceptEdit();
                await closeTab();
            }),
            vscode.commands.registerCommand("coEdPilot.dismissEdit", async () => {
                await clearEdit();
                await closeTab();
            })
        );
    }
}

export {
    predictLocation,
    predictLocationIfHasEditAtSelectedLine,
    PredictLocationCommand,
    GenerateEditCommand
};
