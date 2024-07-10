import vscode from 'vscode';

export type EditType = "add" | "replace" | "remove";

export type LineBreak = "\r" | "\n" | "\r\n";

export type SimpleEdit = {
    afterEdit: string,
    beforeEdit: string
}

export type NativeEditLocation = {
    targetFilePath: string;
    toBeReplaced: string;
    editType: EditType;
    lineBreak: LineBreak;
    atLines: number[];
};

export type NativeEdit = {
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

export type Edit = {
    path: string; // the file path
    s: number; // starting line
    rmLine: number; // number of removed lines
    rmText: string | null; // removed text, could be null
    addLine: number; // number of added lines
    addText: string | null; // added text, could be null
};

