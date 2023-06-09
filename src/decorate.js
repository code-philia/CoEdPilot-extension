const vscode = require('vscode');

const decorator = vscode.window.createTextEditorDecorationType({
    overviewRulerLane: vscode.OverviewRulerLane.Center,
    borderRadius: '2px',
    color: '#000',
    backgroundColor: '#ffffff',
});

function replaceAt(s,index, replacement) {
	console.log(replacement)
	return s.substr(0, index) + replacement+ s.substr(index + 1);
}

function replaceRange(s,start,end, replacement) {
	console.log(replacement)
	return s.substr(0, start) + replacement+ s.substr(end);
}

const cancelDecorator=()=>{
	var dec=decorator;
	const visibleTextEditors = vscode.window.visibleTextEditors;
  	visibleTextEditors.forEach((editor) => {
		const { document } = editor;
		editor.setDecorations(dec, [new vscode.Range(document.positionAt(0),document.positionAt(0))]);
	})
}

function handle(poslist,text) {
	//console.log(Poslist)
	cancelDecorator();
	highlightTargetPos(arr1(poslist),text,decorator)
}

function handleRange(rangelist,editor) {
	cancelDecorator();
	highlightTargetRange(rangelist,decorator,editor)
}

function highlightTargetPos(poslist=[],text,dec=decorator) {
    var editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage("No editor opened.");
        return;
    }
    var { document } = editor;
    var strlist=[];
    for(let i=0;i<poslist.length;i++) {
        console.log(poslist[i],text[poslist[i]]);
        let startPos = document.positionAt(poslist[i]);
        let endPos = document.positionAt(poslist[i] + 1);
        console.log(startPos,endPos)
        strlist.push(new vscode.Range(startPos, endPos));
    }
    console.log(strlist)
    editor.setDecorations(dec, strlist);
};

function highlightTargetRange(rangelist=[],dec=decorator,editor) {
    var { document } = editor;
    var strlist=[];
    for(let i=0;i<rangelist.length;i++) {
        let startPos = document.positionAt(rangelist[i][0]);
        let endPos = document.positionAt(rangelist[i][1]);
        strlist.push(new vscode.Range(startPos, endPos));
    }
    editor.setDecorations(dec, strlist);
};

module.exports = {
	handleRange,
    replaceRange
};