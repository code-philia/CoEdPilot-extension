import { isLanguageSupported, SimpleEdit } from "./utils/base-types";
import { DisposableComponent } from "./utils/base-component";

class EditorState extends DisposableComponent {
    prevCursorAtLine: number;
    currCursorAtLine: number;
    prevSnapshot?: string;
    currSnapshot?: string;
    prevEdits: SimpleEdit[];
    inDiffEditor: boolean;
    language: string;
    toPredictLocation: boolean = false;

    constructor() {
        super();
        this.prevCursorAtLine = 0;
        this.currCursorAtLine = 0;
        this.prevSnapshot = undefined;
        this.currSnapshot = undefined;
        this.prevEdits = [];
        this.inDiffEditor = false;
        this.language= "unknown";
    }

    isActiveEditorLanguageSupported() {
        return isLanguageSupported(this.language);
    }
}

export const globalEditorState = new EditorState();
