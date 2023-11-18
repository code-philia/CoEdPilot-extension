const vscode = require('vscode');

async function createModifiedDocumentAndShowDiff(uri, lineNumber=0, newContent="Hello Compare") {
    // Ensure the document is opened and get its content
    const document = await vscode.workspace.openTextDocument(uri);
    const originalContent = document.getText();
    const originalLines = originalContent.split("\n");

    // Modify the specified line
    if (lineNumber < originalLines.length) {
        originalLines[lineNumber] = newContent;
    } else {
        vscode.window.showErrorMessage("Line number exceeds the document length.");
        return;
    }

    const modifiedContent = originalLines.join("\n");

    // Create a new, unsaved document with the modified content
    // const newDocument = await vscode.workspace.openTextDocument({
    //     language: document.languageId,
    //     content: modifiedContent
    // });

    // Open a diff view to compare the original and the modified document
    vscode.commands.executeCommand('vscode.diff', 
        document.uri, 
        vscode.Uri.file("c:/Users/aaa/Desktop/hello.txt"),
        `Original vs. Modified: ${document.fileName}`
    );
    // hello, new
    // return newDocument;
    
}

async function hello() {
    // heloo
    
}
module.exports = {
    createModifiedDocumentAndShowDiff
};
