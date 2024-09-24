import vscode, { TreeItem, TreeItemCollapsibleState } from 'vscode';
import path from 'path';
import { queryState } from '../global-context';
import { BaseComponent } from '../utils/base-component';
import { getRootPath, toRelPath } from '../utils/file-utils';
import { EditType, NativeEditLocation } from '../utils/base-types';

export class LocationTreeProvider implements vscode.TreeDataProvider<FileItem | ModItem>  {
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

    refresh(modList: NativeEditLocation[]) {
        this.modTree = this.transformModTree(modList);
        this.notifyChangeofTree();
    }

    notifyChangeofTree() {
        this._onDidChangeTreeData.fire(undefined);
        this._onDidChangeLocationNumber.fire(this.numOfLocation());
    }

    numOfLocation() {
        if (this.modTree == null) return 0;

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
    
    
    transformModTree(modList: NativeEditLocation[]) {
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
            modTree.push(this.getFileItem(filePath, modListCategorizedByFilePath[filePath]))
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

    getFileItem(filePath: string, fileMods: NativeEditLocation[]) {
        const modListOnPath = fileMods;
        const fileName = path.basename(filePath); 
        var fileItem = new FileItem(
            fileName,
            vscode.TreeItemCollapsibleState.Collapsed,
            fileName,
            filePath,
            []
        )

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
}

class FileItem extends vscode.TreeItem {
    fileName: string;
    filePath: string;
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
    text: string

    constructor(label: string, collapsibleState: TreeItemCollapsibleState, fileItem: FileItem, fromLine: number, toLine: number, lineContent: string, editType: EditType) {
        super(label, collapsibleState);
        this.collapsibleState = collapsibleState;
        this.fileItem = fileItem;
        this.fromLine = fromLine;
        this.toLine = toLine
        this.lineContent = lineContent
        this.editType = editType;
        this.text = `    ${this.lineContent.trim()}`;

        this.tooltip = `Line ${this.fromLine}`; // match real line numbers in the gutter
        this.description = this.text;
        this.command = {
            command: 'coEdPilot.openFileAtLine',
            title: '',
            arguments: [
                this.fileItem.filePath,
                this.fromLine,
                editType === "add" ? this.fromLine : this.toLine  // edit of type "add" will only place the cursor at the starting of line
            ]
        }
        
        this.iconPath = {
            light: path.join(__filename, '..', '..', 'media', this.getIconFileName()),
            dark: path.join(__filename, '..', '..', 'media', this.getIconFileName()),
        }
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

class EditLocationView extends BaseComponent {
    provider: LocationTreeProvider;
    treeView: vscode.TreeView<FileItem | TreeItem>;

    constructor() {
        super();
        this.provider = new LocationTreeProvider();
        
        const treeViewOptions = {
            treeDataProvider: this.provider,
            showCollapseAll: true
        }
        const treeView = vscode.window.createTreeView('editLocations', treeViewOptions);
        this.treeView = treeView;

        this.register(
            treeView,
            this.provider.onDidChangeLocationNumber(async (num) => {
                // Set the whole badge here. Only setting the value won't trigger update
                this.treeView.badge = {
                    tooltip: `${num} possible edit locations`,
                    value: num
                }
                
            }, this),
            queryState.onDidChangeLocations(async (qs) => {
                this.provider.refresh(qs.locations);
                if (!this.treeView.visible) {
                    await vscode.commands.executeCommand('editLocations.focus');
                }
                await this.treeView.reveal(this.provider.modTree[0], { expand: 2 });
            }, this),
        );
    }
}

export const editLocationView = new EditLocationView();
