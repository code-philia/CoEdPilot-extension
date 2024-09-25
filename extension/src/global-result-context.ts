import os from "os";
import vscode from "vscode";
import { DisposableComponent } from "./utils/base-component";
import { LineBreak, BackendApiEditLocation } from "./utils/base-types";
import { LocationResultDecoration } from "./ui/location-decoration";
import { globalLocationViewManager } from "./views/location-tree-view";

// TODO consider using/transfering to `async-lock` for this
class EditLock {
    isLocked: boolean = false;

    async tryWithLock(asyncCallback: (...args: any[]) => any) {
        if (this.isLocked) return undefined;
        this.isLocked = true;
        try {
            return await Promise.resolve(asyncCallback());
        } catch (err: any) {
            console.error(`Error occured when running in edit lock (async): \n${err.stack}`);
            // throw err;
        } finally {
            this.isLocked = false;
        }
    }
}

class QuerySettings {
    private commitMessage?: string;

    async requireCommitMessage() {
        if (this.commitMessage) {
            return this.commitMessage;
        }

        return await this.inputCommitMessage();
    }

    async inputCommitMessage() {
        const userInput = await vscode.window.showInputBox({
            prompt: 'Enter a description of edits you want to make.',
            placeHolder: 'Add a feature...',
            ignoreFocusOut: true,
            value: this.commitMessage ?? '',
            title: "✍️ Edit Description"
        });
        
        if (userInput) {
            this.commitMessage = userInput;
        }
        return userInput;   // returns undefined if canceled
    }
}

/**
 * This class manages an successfully produced location result,
 * i.e., its data, ui, and lifecycle
 */
class LocationResult{
    private readonly locations: BackendApiEditLocation[] = [];
    
    private decoration: LocationResultDecoration;

    constructor(locations: BackendApiEditLocation[]) {
        this.locations = locations;
        this.decoration = new LocationResultDecoration(this.locations);
        this.decoration.show();
        globalLocationViewManager.reloadLocations(this.locations);
    }

    getLocations() {
        return this.locations;
    }

    dispose() {
        this.decoration.dispose();
        // TODO there could be multiple sets of locations, use a manager class for each
        globalLocationViewManager.reloadLocations([]);
    }
}

class QueryContext extends DisposableComponent{
    readonly querySettings: QuerySettings = new QuerySettings();
    private activeLocationResult?: LocationResult;

    constructor() {
        super();
        this.register(
            vscode.commands.registerCommand('coEdPilot.inputMessage', () => {
                this.querySettings.inputCommitMessage();
            })
        );
    }

    getLocations() {
        return this.activeLocationResult?.getLocations();
    }

    updateLocations(locations?: BackendApiEditLocation[]) {
        // cannot use destructor() here due to JavaScript nature
        this.activeLocationResult?.dispose();
        if (locations) { 
            this.activeLocationResult = new LocationResult(locations);
        }
    }
}

export const globalEditLock = new EditLock();
export const globalQueryContext = new QueryContext();

export const supportedOSTypes = ['Windows_NT', 'Darwin', 'Linux'];
export const osType = os.type();

if (!supportedOSTypes.includes(osType)) {
    throw RangeError(`Operating system (node detected: ${osType}) is not supported yet.`);
}

export const defaultLineBreaks: { [key: string]: LineBreak } = {
    'Windows_NT': '\r\n',
    'Darwin': '\r',
    'Linux': '\n'
};
export const defaultLineBreak: string = defaultLineBreaks[osType] ?? '\n';
