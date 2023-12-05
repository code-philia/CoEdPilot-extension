import vscode from 'vscode';
import path from 'path';
import { queryState } from './queries';
import { BaseComponent } from './base-component';

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
            this.provider.onDidChangeTreeData((num) => {
                // Set the whole badge here. Only setting the value won't trigger update
                this.treeView.badge = {
                    tooltip: `${num} possible edit locations`,
                    value: num
                }
                vscode.commands.executeCommand('editLocations.focus');
            }, this),
            queryState.onDidQuery((qs) => this.provider.refresh(qs.locations), this)
        );
    }
}

class LocationTreeProvider  {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
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
        this._onDidChangeTreeData.fire(
            this.numOfLocation(),
        );
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
            fileItem.mods.push(
                new ModItem(
                    `Line ${loc.atLines[0]}`,
                    vscode.TreeItemCollapsibleState.None,
                    fileItem,
                    loc.atLines[0],
                    loc.lineInfo.text,
                    loc.editType
                )
            )
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
        this.description = `   ${this.filePath}`;
    }
    
    iconPath = {
        light: path.join(__filename, '..', '..', 'media', 'file.svg'),
        dark: path.join(__filename, '..', '..', 'media', 'file.svg')
    }

    contextValue = 'file';
}

class ModItem extends vscode.TreeItem {
    constructor(label, collapsibleState, fileItem, atLine, lineContent, editType) {
        super(label, collapsibleState);
        this.collapsibleState = collapsibleState;
        this.fileItem = fileItem;
        this.atLine = atLine;
        this.editType = editType;
        this.lineContent = lineContent
        this.text = `    ${this.lineContent.trim()}`;

        this.tooltip = `line ${this.atLine + 1}`; // match real line numbers in the gutter
        this.description = this.text;
        this.command = {
            command: 'editPilot.openFileAtLine',
            title: '',
            arguments: [this.fileItem.filePath, this.atLine]
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
                return 'add.png';
            case 'remove':
                return 'remove.png';
            default:
                return 'edit.png';
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
        return `Line ${this.atLine}`;
    }

    contextValue = 'mod';
}

export {
    EditLocationView
};
