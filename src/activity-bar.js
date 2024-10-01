import vscode from 'vscode';
import path from 'path';
import { queryState } from './global-context';
import { BaseComponent } from './base-component';
import { getRootPath, toRelPath } from './file';

class LocationTreeProvider  {
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

    refresh(modList) {
        this.modTree = this.transformModTree(modList);
        this.notifyChangeofTree();
    }

    notifyChangeofTree() {
        this._onDidChangeTreeData.fire();
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

    getTreeItem(element) {
        return element;
    }

    /**
     * Structure of modList should be like
     * {
     *     "atLine": [
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
     *                     "atLine": 0,
     *                     "start": 0,
     *                     "end": 0,
     *                     "toBeReplaced": ""
     *                 }
     *             ]
     *         }, ...
     *     ]
     * }
     */
    
    
    transformModTree(modList) {
        const categorizeByAttr = (arr, attr) => 
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

    getChildren(element) {
        if (element) {
            return element.mods;
        } else {
            return this.modTree;
        }
    }

    getParent(element) {
        if (element.fileItem) {
            return element.fileItem;
        } else {
            return undefined;
        }
    }

    getFileItem(filePath, fileMods) {
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
            let fromLine = loc.atLines[0];
            let toLine = loc.atLines.at(-1) + 1; // fromLine is inclusive, and toLine is exclusive
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
    constructor(label, collapsibleState, fileName, filePath, mods) {
        super(label, collapsibleState);
        this.label = label;
        this.collapsibleState = collapsibleState;
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
    constructor(label, collapsibleState, fileItem, fromLine, toLine, lineContent, editType) {
        super(label, collapsibleState);
        this.collapsibleState = collapsibleState;
        this.fileItem = fileItem;
        this.fromLine = fromLine;
        this.toLine = toLine
        this.editType = editType;
        this.lineContent = lineContent
        this.text = `    ${this.lineContent.trim()}`;

        this.tooltip = `Line ${this.fromLine + 1}`;
        this.description = this.text;
        this.command = {
            command: 'coEdPilot.openFileAtLine',
            title: '',
            arguments: editType === "add" ? [
                this.fileItem.filePath,
                this.fromLine,
                this.toLine  // edit of type "add" will only place the cursor at the starting of line
            ] : [
                this.fileItem.filePath,
                this.toLine,
                this.toLine
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
