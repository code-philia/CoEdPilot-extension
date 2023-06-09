const vscode = require('vscode');
var exec = require('child_process').exec;
const fs = require('fs')
const {handleRange,replaceRange}=require('./decorate');
const {getWebviewContent}=require('./sidebar')

class Edit {
    constructor() {
		this.rangeList=[];
		this.strList=[];
		this.vList=[];
		this.data={};
		this.text="";
		this.activeEditor;
	}

	getChange(idx) {
		if (idx >= this.vList.length || this.vList[idx]) {
		  	return Promise.resolve(); // 已经应用过变更，直接解析Promise
		}
	  
		this.vList[idx] = 1;
		let newcontent = replaceRange(this.text, this.rangeList[idx][0], this.rangeList[idx][1], this.data[idx][2]);
		this.rangeList[idx][1] = this.rangeList[idx][0] + this.data[idx][2].length;
		for (let i = idx + 1; i < this.data.length; i++) {
			this.rangeList[i][0] += this.data[idx][2].length - this.data[idx][1] + this.data[idx][0];
			this.rangeList[i][1] += this.data[idx][2].length - this.data[idx][1] + this.data[idx][0];
		}
	  
		return new Promise((resolve, reject) => {
			this.activeEditor.edit(text => text.replace(new vscode.Range(0, 0, this.activeEditor.document.lineCount, 0), newcontent)).then(() => {
				handleRange(this.rangeList, this.activeEditor);
				this.text = newcontent;
				resolve(); // 解析Promise
			}).catch(reject);
		});
	}
	  
	relateChange(idx,changeall=false) {
		const self = this; 
	  
		function executeNext(i) {
			if (i < self.strList.length) {
				if ((changeall)||(self.strList[i][0] === self.strList[idx][0] && self.strList[i][1] === self.strList[idx][1])) 
					return self.getChange(i).then(() => executeNext(i + 1));
				else 
					return executeNext(i + 1);
			} else {
				return Promise.resolve();
			}
		}
	  
		return executeNext(0);
	}  

	getUndo(idx,change=true) {
		console.log(idx);
		if(idx>=this.vList.length || !this.vList[idx])
			return;

		this.vList[idx]=0;
		let newcontent=replaceRange(this.text,this.rangeList[idx][0],this.rangeList[idx][0]+this.strList[idx][1].length,this.strList[idx][0]);
		this.rangeList[idx][1]=this.rangeList[idx][0]+this.strList[idx][0].length;
		console.log(newcontent)
		for(let i=idx+1;i<this.data.length;i++) {
			this.rangeList[i][0]-=this.data[idx][2].length-this.data[idx][1]+this.data[idx][0];
			this.rangeList[i][1]-=this.data[idx][2].length-this.data[idx][1]+this.data[idx][0];
		}
		this.activeEditor.edit(text => text.replace(new vscode.Range(0, 0, this.activeEditor.document.lineCount, 0),newcontent)).then(() => {
			console.log(this.activeEditor.document.getText());
			handleRange(this.rangeList,this.activeEditor);
			this.text=newcontent;
		});
	}
	
	getDisplay(activeEditor,data,text) {
		
		this.data=data;
		this.activeEditor=activeEditor;
		for(let i=0;i<data.length;i++) {
			this.vList.push(0);
			let tmp=[data[i][0],data[i][1]];
			console.log(tmp);
			this.rangeList.push(tmp);
			tmp=[text.substr(data[i][0],data[i][1]-data[i][0]),data[i][2]]
			console.log(tmp)
			this.strList.push(tmp)
		}
		handleRange(this.rangeList,activeEditor);
		this.text=text;
		console.log('ha')
		//webview
		let code_panel = vscode.window.createWebviewPanel(
			"suggestion",
			"code suggestions",
			vscode.ViewColumn.Two,
			{ enableScripts: true }
		);
		code_panel.webview.html = getWebviewContent(this.strList);
		code_panel.webview.onDidReceiveMessage(
			message => {
				if(message.applyAll) {
					this.relateChange(0,true);
				}
				else {
					var idx = message.idx;
					if(!message.undo)
						this.getChange(message.idx);
					else 
						this.getUndo(message.idx);
				}
				
			}
		);
	}

	static get providedCodeActionKinds() {
		return [
			vscode.CodeActionKind.QuickFix
		];
	}

	provideCodeActions(document, range) {
		console.log("code action ",range);
		let idx=this.isInEditlist(document, range);
		
		if (idx<0) 
			return [];
        var text=this.strList[idx][1];
        console.log(text)
		const replaceWithEdit = this.createFix(document, range, text, idx);
		replaceWithEdit.command = {
            command: 'extension.applyFix',
            title: '',
            arguments: [idx],
        };
		return [
			replaceWithEdit
		];
	}

	isInEditlist(document, range) {
		const start = document.offsetAt(range.start);
    	const end = document.offsetAt(range.end);
		console.log(start,end);
		for (let i = 0; i < this.rangeList.length; i++) {
			const [rangeStart, rangeEnd] = this.rangeList[i];
			if (rangeStart === start && rangeEnd === end) 
				return i;
		}

    	return -1;
	}

	isInRange(document,range) {
		const start = document.offsetAt(range.start);
    	const end = document.offsetAt(range.end);
		console.log(start,end);
		for (let i = 0; i < this.rangeList.length; i++) {
			const [rangeStart, rangeEnd] = this.rangeList[i];
			if (rangeStart <= start && rangeEnd >= end) 
				return i;
		}

    	return -1;
	}

	selectionControl() {
		const selectionDecorator = vscode.window.createTextEditorDecorationType({
			backgroundColor: 'gray',
			color: '#ffffff'
		});
		
		vscode.window.onDidChangeTextEditorSelection(event => {
			console.log('ghh')
			const selection = this.activeEditor.selection;
			const selectedRange = new vscode.Range(selection.start, selection.end);
			let idx=this.isInRange(this.activeEditor.document, selectedRange);
			console.log("selected",selectedRange,idx)
			if(idx>=0)
				this.activeEditor.setDecorations(selectionDecorator, [selectedRange]);
			else 
				this.activeEditor.setDecorations(selectionDecorator, []);
		});
	}

	createFix(document, range, text, idx) {
		const self = this;
		const fix = new vscode.CodeAction(`${text}`, vscode.CodeActionKind.QuickFix);
		fix.edit = new vscode.WorkspaceEdit();
		return fix;
	}
	
}

module.exports = {
	Edit
};