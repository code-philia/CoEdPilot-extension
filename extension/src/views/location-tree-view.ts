import vscode, { TreeItem, TreeItemCollapsibleState } from 'vscode';
import path from 'path';
import { DisposableComponent } from '../utils/base-component';
import { getRootPath, toRelPath } from '../utils/file-utils';
import { EditType, BackendApiEditLocation, FileEdits } from '../utils/base-types';

export class LocationTreeDataProvider implements vscode.TreeDataProvider<FileItem | ModItem>  {
    private _onDidChangeTreeData: vscode.EventEmitter<FileItem | undefined> = new vscode.EventEmitter<FileItem | undefined>();
    onDidChangeTreeData: vscode.Event<FileItem | undefined> = this._onDidChangeTreeData.event;
    private _onDidChangeLocationNumber: vscode.EventEmitter<number> = new vscode.EventEmitter<number>();
    onDidChangeLocationNumber: vscode.Event<number> = this._onDidChangeLocationNumber.event;

    modTree: FileItem[];

    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this._onDidChangeLocationNumber = new vscode.EventEmitter();
        this.onDidChangeLocationNumber = this._onDidChangeLocationNumber.event;
        this.modTree = [];
    }

    empty() {
        this.modTree = [];
        this.notifyChangeofTree();
    }

    reloadData(modList: BackendApiEditLocation[]) {
        this.modTree = this.buildModTree(modList);
        this.notifyChangeofTree();
    }

    // TODO reimplement this in another provider class
    async reloadRefactorData(editList: FileEdits[]) {
        const workspaceFolderPath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!workspaceFolderPath) {
            return [];
        }

        this.modTree = this.buildRefactorTree(editList);
        this.notifyChangeofTree();
    }

    notifyChangeofTree() {
        this._onDidChangeTreeData.fire(undefined);
        this._onDidChangeLocationNumber.fire(this.numOfLocation());
    }

    numOfLocation() {
        if (this.modTree === null) return 0;

        let num = 0;
        for (const fileItem of this.modTree) {
            num += fileItem?.mods?.length ?? 0;
        }

        return num;
    }

    getTreeItem(element: FileItem | ModItem) {
        return element;
    }

    /**
     * Structure of modList should be like
     * {
     *     "atLines": [
     *         11
     *     ],
     *     "editType": "add",
     *     "endPos": 386,
     *     "lineBreak": "\r\n",
     *     "prevEdits": [
     *         {
     *             "afterEdit": "export { BaseComponent as Component } from './component';",
     *             "beforeEdit": "export { Component } from './component';"
     *         }
     *     ],
     *     "startPos": 334,
     *     "targetFilePath": "c:/Users/aaa/Desktop/page.js/compat/src/PureComponent.js",
     *     "toBeReplaced": "PureComponent.prototype.isPureReactComponent = true;"
     * },
     * 
     * Construct Mod Tree:
     * {
     *     "files": [
     *         {
     *             "fileName": "",
     *             "filePath": "",
     *             "mods": [
     *                 {
     *                     "atLines": 0,
     *                     "start": 0,
     *                     "end": 0,
     *                     "toBeReplaced": ""
     *                 }
     *             ]
     *         }, ...
     *     ]
     * }
     */
    
    
    buildModTree(modList: BackendApiEditLocation[]) {
        const categorizeByAttr = (arr: any[], attr: any) => 
            arr.reduce((acc, obj) => {
                const key = obj[attr];
                if (!acc[key]) acc[key] = [];
                acc[key].push(obj);
                return acc;
            }, {});

        const modListCategorizedByFilePath = categorizeByAttr(modList, 'targetFilePath');

        var modTree = [];
        for (const filePath in modListCategorizedByFilePath) {  
            modTree.push(this.getFileItem(filePath, modListCategorizedByFilePath[filePath]));
        }

        return modTree;
    }

    buildRefactorTree(editList: FileEdits[]) {
        var modTree = [];
        for (const [uri, edits] of editList) {
            // TODO the conversion here is lossy
            modTree.push(this.getRefactorFileItem(uri, edits, editList));
        }

        return modTree;
    }

    getChildren(element?: FileItem) {
        if (element) {
            return element.mods;
        } else {
            return this.modTree;
        }
    }

    getParent(element: ModItem) {
        if (element.fileItem) {
            return element.fileItem;
        } else {
            return undefined;
        }
    }

    getFileItem(filePath: string, fileMods: BackendApiEditLocation[]) {
        const modListOnPath = fileMods;
        const fileName = path.basename(filePath); 
        var fileItem = new FileItem(
            fileName,
            vscode.TreeItemCollapsibleState.Expanded,
            fileName,
            filePath,
            []
        );

        for (const loc of modListOnPath) {
            let fromLine = loc.editType === "add" ? loc.atLines[0] + 1 : loc.atLines[0];
            let toLine = loc.editType === "add" ? loc.atLines[loc.atLines.length - 1] + 2 : loc.atLines[loc.atLines.length - 1] + 1;
            fileItem.mods.push(
                new ModItem(
                    `Line ${fromLine + 1}`,
                    vscode.TreeItemCollapsibleState.None,
                    fileItem,
                    fromLine,
                    toLine,
                    loc.lineInfo.text,
                    loc.editType
                )
            );
        }

        return fileItem;
    }

    // TODO use another tree view implementation
    // All ModItems of the refactor tree points to one (single) refactor review command
    getRefactorFileItem(fileUri: vscode.Uri, inFileEdits: vscode.TextEdit[], refactorEdits: FileEdits[]) {
        const fileName = path.basename(fileUri.fsPath);
        // TODO the relPath could be incorrect if there are multiple workspace folders
        const relPath = path.relative(vscode.workspace.workspaceFolders?.[0].uri.fsPath ?? '', fileUri.fsPath);
        var fileItem = new FileItem(
            fileName,
            vscode.TreeItemCollapsibleState.Expanded,
            fileName,
            relPath,
            []
        );

        for (const edit of inFileEdits) {
            // TODO don't use fromLine toLine, use range. Though it is no impact here
            let fromLine = edit.range.start.line;
            let toLine = edit.range.end.line;
            fileItem.mods.push(
                new ModItem(
                    `Line ${fromLine + 1}`,
                    vscode.TreeItemCollapsibleState.None,
                    fileItem,
                    fromLine,
                    toLine,
                    edit.newText.split('\n')[0],     // this is inefficient, should be readLine
                    'replace',
                    true,
                    refactorEdits
                )
            );
        }

        return fileItem;
    }
}

class FileItem extends vscode.TreeItem {
    fileName: string;
    filePath: string;   // TODO don't use relative path here. Resolve absolute path, i.e., use a relative path with context linked to specific workspace folder
    mods: ModItem[];

    constructor(label: string, collapsibleState: TreeItemCollapsibleState, fileName: string, filePath: string, mods: ModItem[]) {
        super(label, collapsibleState);
        this.fileName = fileName;
        this.filePath = filePath;
        this.mods = mods;
        this.tooltip = this.fileName;
        this.description = `   ${toRelPath(getRootPath(), this.filePath)}`;
        this.resourceUri = vscode.Uri.file(this.filePath);
    }
    
    iconPath = vscode.ThemeIcon.File;

    contextValue = 'file';
}

class ModItem extends vscode.TreeItem {
    fileItem: FileItem;
    fromLine: number;
    toLine: number;
    lineContent: string;
    editType: EditType;
    text: string;

    constructor(label: string, collapsibleState: TreeItemCollapsibleState, fileItem: FileItem, fromLine: number, toLine: number, lineContent: string, editType: EditType, isRefactor: boolean = true, refactorEdits: FileEdits[] | undefined = undefined) {
        super(label, collapsibleState);
        this.collapsibleState = collapsibleState;
        this.fileItem = fileItem;
        this.fromLine = fromLine;
        this.toLine = toLine;
        this.lineContent = lineContent;
        this.editType = editType;
        this.text = `    ${this.lineContent.trim()}`;

        this.tooltip = `Line ${this.fromLine}`; // match real line numbers in the gutter
        this.description = this.text;
        if (isRefactor && refactorEdits) {
            this.command = {
                command: 'coEdPilot.openRefactorPreview',
                title: 'Open Refactor View',
                arguments: [refactorEdits]
            };
        } else {
            this.command = {
                command: 'coEdPilot.openFileAtLine',
                title: '',
                arguments: [
                    this.fileItem.filePath,
                    this.fromLine,
                    editType === "add" ? this.fromLine : this.toLine  // edit of type "add" will only place the cursor at the starting of line
                ]
            };
        }
        
        // FIXME this way to get assets is alkward
        this.iconPath = {
            light: path.join(__filename, '..', '..', '..', 'assets', this.getIconFileName()),
            dark: path.join(__filename, '..', '..', '..', 'assets', this.getIconFileName()),
        };
        this.label = this.getLabel();
    }


    getIconFileName() {
        switch (this.editType) {
            case 'add':
                return 'add-green.svg';
            case 'remove':
                return 'remove.svg';
            default:
                return 'edit-red.svg';
        }
    }

    getLabel() {
        // switch (this.editType) {
        //     case 'add':
        //         return `Adding at line ${this.atLine}`;
        //     case 'remove':
        //         return `Removing line ${this.atLine}`;
        //     default:
        //         return `Modifying line ${this.atLine}`;
        // }
        return `Line ${this.fromLine + 1}`;
    }

    contextValue = 'mod';
}

class EditLocationViewManager extends DisposableComponent {
    provider: LocationTreeDataProvider;
    treeView: vscode.TreeView<FileItem | TreeItem>;

    constructor() {
        super();
        this.provider = new LocationTreeDataProvider();
        
        const treeViewOptions: vscode.TreeViewOptions<FileItem | ModItem> = {
            treeDataProvider: this.provider,
            showCollapseAll: true
        };
        // TODO do not always display the treeview, but only when there are locations
        const treeView = vscode.window.createTreeView('editLocations', treeViewOptions);
        this.treeView = treeView;

        this.register(
            treeView
        );
    }

    setUpBadge(numOfLocation: number) {
        this.treeView.badge = {
            tooltip: `${numOfLocation} possible edit locations`,
            value: numOfLocation
        };
    }

    reloadLocations(locations: BackendApiEditLocation[]) {
        this.provider.reloadData(locations);
        this.setUpBadge(locations.length);
        Promise.resolve(async () => {
            if (!this.treeView.visible) {
                await vscode.commands.executeCommand('editLocations.focus');
            }
            await this.treeView.reveal(this.provider.modTree[0], { expand: 2 });
        });
    }
}

export const globalLocationViewManager = new EditLocationViewManager();

class RefactorPreviewViewManager implements vscode.Disposable {
    provider: LocationTreeDataProvider;
    treeView: vscode.TreeView<FileItem | TreeItem>;

    constructor() {
        this.provider = new LocationTreeDataProvider();
        
        const treeViewOptions: vscode.TreeViewOptions<FileItem | ModItem> = {
            treeDataProvider: this.provider,
            showCollapseAll: true
        };
        // TODO do not always display the treeview, but only when there are locations
        const treeView = vscode.window.createTreeView('editLocations', treeViewOptions);
        this.treeView = treeView;
    }

    setUpBadge(numOfLocation: number) {
        this.treeView.badge = {
            tooltip: `${numOfLocation} possible edit locations`,
            value: numOfLocation
        };
    }

    reloadLocations(locations: FileEdits[]) {
        this.provider.reloadRefactorData(locations);
        this.setUpBadge(locations.length);
        Promise.resolve(async () => {
            if (!this.treeView.visible) {
                await vscode.commands.executeCommand('editLocations.focus');
            }
            await this.treeView.reveal(this.provider.modTree[0], { expand: 2 });
        });
    }

    dispose() {
        this.treeView.dispose();
    }
}

export const globalRefactorPreviewViewManager = new RefactorPreviewViewManager();
