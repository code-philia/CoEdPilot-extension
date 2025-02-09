import { EditType, LineBreak } from "../utils/base-types";
import * as vscode from 'vscode';
import * as t from 'io-ts';

const User = t.type({
    userId: t.number,
    name: t.string
});

function headOf(s: string, l: number = 30000): string {
    return s.length > l ? s.slice(0, l) + "..." : s;
}

class InvalidJsonTypeError extends Error { }

abstract class JsonType<T> {
    abstract desc(): string;
    abstract accept(obj: any): obj is T;

    assert(obj: any): asserts obj is T {
        if (!this.accept(obj)) 
            throw new InvalidJsonTypeError(`This JSON object is not of type ${this.desc()}: ${headOf(JSON.stringify(obj))}`);
    }
    resolve(str: string): T | undefined {
        try {
            const obj = JSON.parse(str);
            return this.accept(obj) ? obj : undefined;
        } catch (e) {
            return undefined;
        }
    }
}

export type BackendApiEditRefactor = {
    file: string,
    line: number,
    beforeText: string,
    afterText: string
}

export class BackendApiEditRefactorJsonType extends JsonType<BackendApiEditRefactor> {
    desc(): string {
        return "BackendApiEditRefactor";
    }
    accept(obj: any): obj is BackendApiEditRefactor {
        return obj.file && obj.line && obj.beforeText && obj.afterText;
    }
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

export class BackendApiEditLocationJsonType extends JsonType<BackendApiEditLocation> {
    desc(): string {
        return "BackendApiEditLocationJsonType";
    }
    accept(obj: any): obj is BackendApiEditLocation {
        return obj.targetFilePath && obj.editType && obj.lineBreak && obj.atLines && obj.lineInfo;
    }
}

export type BackendApiEditGeneration = {
    editType: string,
    replacement: string[]
}

export class BackendApiEditGenerationJsonType extends JsonType<BackendApiEditGeneration> {
    desc(): string {
        return "BackendApiEditGeneration";
    }

    accept(obj: any): obj is BackendApiEditGeneration {
        return obj.editType && typeof (obj.editType) === "string"
            && obj.replacement instanceof Array
            && obj.replacement.every((r: any) => typeof (r) === "string");
    }
}
