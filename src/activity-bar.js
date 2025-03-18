import vscode from 'vscode';
import path from 'path';
import { queryState } from './global-context';
import { BaseComponent, registerCommand } from './base-component';
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
                    loc.editType,
                    loc.confidence
                )
            );
        }

        return fileItem;
    }

    sortByLineNumber(asc) {
        if (asc) {
            this.modTree.forEach((fileItem) => {
                fileItem.mods.sort((a, b) => a.fromLine - b.fromLine);
            });
        } else {
            this.modTree.forEach((fileItem) => {
                fileItem.mods.sort((a, b) => b.fromLine - a.fromLine);
            });
        }
    }

    sortByConfidence(asc) {
        if (asc) {
            this.modTree.forEach((fileItem) => {
                fileItem.mods.sort((a, b) => a.confidence - b.confidence);
            });
        } else {
            this.modTree.forEach((fileItem) => {
                fileItem.mods.sort((a, b) => b.confidence - a.confidence);
            });
        }
    }

    sort(criterion = 'confidence', order = 'asc') {
        if (criterion === 'lineNumber') {
            this.sortByLineNumber(order === 'asc');
        } else {
            this.sortByConfidence(order === 'asc');
        }
        this.notifyChangeofTree();
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
    constructor(label, collapsibleState, fileItem, fromLine, toLine, lineContent, editType, confidence) {
        super(label, collapsibleState);
        this.collapsibleState = collapsibleState;
        this.fileItem = fileItem;
        this.fromLine = fromLine;
        this.toLine = toLine
        this.editType = editType;
        this.lineContent = lineContent
        this.text = `    ${this.lineContent.trim()}`;
        this.confidence = confidence;

        this.tooltip = `Line ${this.fromLine + 1}`;
        this.description = this.text;
        this.command = {
            command: 'coEdPilot.openFileAtLine',
            title: '',
            arguments: editType === "add" ? [
                this.fileItem.filePath,
                this.toLine,
                this.toLine  // edit of type "add" will only place the cursor at the starting of line
            ] : [
                this.fileItem.filePath,
                this.fromLine,
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
    sortCriterion = undefined;
    sortOrder = undefined;

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
                this.sort();
                if (!this.treeView.visible) {
                    await vscode.commands.executeCommand('editLocations.focus');
                }
                await this.treeView.reveal(this.provider.modTree[0], { expand: 2 });
            }, this),
            // NOTE if register for each edit location view, may cause config
            registerCommand('coEdPilot.setLocationSortByLineNumber', async () => {
                this.setSortCriterion("lineNumber");
            }),
            registerCommand('coEdPilot.setLocationSortByConfidence', async () => {
                this.setSortCriterion("confidence");
            }),
            registerCommand('coEdPilot.setLocationSortAsc', async () => {
                this.setSortOrder("asc");
            }),
            registerCommand('coEdPilot.setLocationSortDesc', async () => {
                this.setSortOrder("desc");
            })
        );

        // `sortCriterion` and `order` is initially undefined, so they will be switched to 'lineNumber' and 'asc' first
        this.switchSortCriterion();
        // this.switchSortOrder();  // switch order will switch the original order of "lineNumber" criterion to "desc"
    }
    
    sort() {
        this.provider?.sort(this.sortCriterion, this.sortOrder);
    }

    setSortCriterion(criterion) {
        this.sortCriterion = criterion;
        this.provider.sort(this.sortCriterion, this.sortOrder);
        vscode.commands.executeCommand('setContext', 'coEdPilot.locationSortCriterion', criterion);
    }

    setSortOrder(order) {
        this.sortOrder = order;
        this.provider.sort(this.sortCriterion, this.sortOrder);
        vscode.commands.executeCommand('setContext', 'coEdPilot.locationSortOrder', order);
    }

    switchSortCriterion() {
        if (this.sortCriterion === 'lineNumber') {
            this.setSortCriterion('confidence');
            this.setSortOrder('desc');
        } else {
            this.setSortCriterion('lineNumber');
            this.setSortOrder('asc');
        }
    }

    switchSortOrder() {
        if (this.sortOrder === 'asc') {
            this.setSortOrder('desc');
        } else {
            this.setSortOrder('asc');
        }
    }
}

export const editLocationView = new EditLocationView();
