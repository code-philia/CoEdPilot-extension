/*  new WysiHat.Selection(editor)
*  - editor (WysiHat.Editor): the editor object that you want to bind to
**/
function initialize(window) {
 this.window = window;
 this.document = window.document;
}

/**
* WysiHat.Selection#getSelection() -> Selection
*  Get selected text.
**/
function getSelection() {
 return this.window.getSelection ? this.window.getSelection() : this.window.document.selection;
}
