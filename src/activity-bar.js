const vscode = require('vscode');
const path = require('path');
const { queryState } = require('./query');
const { BaseComponent } = require('./base-component');


class LocationTreeProvider extends BaseComponent {
    constructor() {
        super();
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.modTree = [];

        this.disposable = vscode.Disposable.from(
            queryState.onDidQuery((qs) => this.refresh(qs.locations), this),
            vscode.window.registerTreeDataProvider('editPoints', this),
            vscode.commands.registerCommand('editPilot.refreshEditPoints', (modList) => this.refresh(modList))
        );
    }

    empty() {
        this.modTree = [];
        this._onDidChangeTreeData.fire();
    }

    refresh(modList) {
        this.modTree = this.transformModTree(modList);
        this._onDidChangeTreeData.fire();
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

        for (const mod of modListOnPath) {
            fileItem.mods.push(
                new ModItem(
                    `Line ${mod.atLine[0]}`,
                    vscode.TreeItemCollapsibleState.None,
                    fileItem,
                    mod.atLine[0],
                    mod.startPos,
                    mod.endPos,
                    `    ${mod.toBeReplaced.trim()}`,
                    mod.editType
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
    constructor(label, collapsibleState, fileItem, atLine, start, end, toBeReplaced, editType) {
        super(label, collapsibleState);
        this.collapsibleState = collapsibleState;
        this.fileItem = fileItem;
        this.atLine = atLine;
        this.start = start;
        this.end = end;
        this.toBeReplaced = toBeReplaced;
        this.editType = editType;
        
        this.tooltip = `Line ${this.atLine}`;
        this.description = this.toBeReplaced;
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

module.exports = {
    LocationTreeProvider
};
