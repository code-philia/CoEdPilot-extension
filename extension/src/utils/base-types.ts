import vscode from 'vscode';

export type EditType = "add" | "replace" | "remove";

export type LineBreak = "\r" | "\n" | "\r\n";

export type SimpleEdit = {
    afterEdit: string,
    beforeEdit: string
}

export type BackendApiEditLocation = {
    targetFilePath: string;
    editType: EditType;
    lineBreak: LineBreak;
    atLines: number[];
    lineInfo: {
        range: vscode.Range,
        text: string
    }
};

export type BackendApiEdit = {
    atLines: number[],
    editType: EditType,
    endPos: number,
    lineBreak: LineBreak,
    prevEdits: SimpleEdit[],
    startPos: number,
    targetFilePath: string,
    toBeReplaced: string,
    lineInfo: {
        range: vscode.Range,
        text: string
    }
};

export type SingleLineEdit = {
    location: vscode.Location,  // always the beginning of the line
    beforeContent: string, 
    afterContent: string
};
// export type RangeEdit = {
//     location: vscode.Location,
//     afterContent: string
// }
export type FileEdits = [vscode.Uri, vscode.TextEdit[]];

export type Edit = {
    path: string; // the file path
    line: number; // starting line
    rmLine: number; // number of removed lines
    rmText: string | null; // removed text, could be null
    addLine: number; // number of added lines
    addText: string | null; // added text, could be null
};
export const supportedLanguages = [
    "go",
    "python",
    "typescript",
    "javascript",
    "java"
];

export function isLanguageSupported(lang: string) {
    return supportedLanguages.includes(lang);
}

