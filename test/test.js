/*---------------------------------------------------------
 * Copyright (C) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------*/

const vscode = require('vscode');

const COMMAND = 'code-actions-sample.command';

function activate(context) {
	console.log('ha')
	context.subscriptions.push(
		vscode.languages.registerCodeActionsProvider('javascript', new Emojizer(), {
			providedCodeActionKinds: Emojizer.providedCodeActionKinds
		})
	);
}

/**
 * Provides code actions for converting :) to a smiley emoji.
 */
class Emojizer {
	static get providedCodeActionKinds() {
		return [
			vscode.CodeActionKind.QuickFix
		];
	}

	provideCodeActions(document, range) {
		if (!this.isAtStartOfSmiley(document, range)) {
			return;
		}

		const replaceWithSmileyCatFix = this.createFix(document, range, '😺');
		const replaceWithSmileyFix = this.createFix(document, range, '😀');
		const replaceWithSmileyHankyFix = this.createFix(document, range, '💩');
		return [
			replaceWithSmileyCatFix,
			replaceWithSmileyFix,
			replaceWithSmileyHankyFix,
		];
	}

	isAtStartOfSmiley(document, range) {
		const start = range.start;
		const line = document.lineAt(start.line);
		return line.text[start.character] === ':' && line.text[start.character + 1] === ')';
	}

	createFix(document, range, emoji) {
		const fix = new vscode.CodeAction(`Convert to ${emoji}`, vscode.CodeActionKind.QuickFix);
		fix.edit = new vscode.WorkspaceEdit();
		fix.edit.replace(document.uri, new vscode.Range(range.start, range.start.translate(0, 2)), emoji);
		return fix;
	}

	createCommand() {
		const action = new vscode.CodeAction('Learn more...', vscode.CodeActionKind.Empty);
		action.command = { command: COMMAND, title: 'Learn more about emojis', tooltip: 'This will open the unicode emoji page.' };
		return action;
	}
}


module.exports = {
	Emojizer
};
