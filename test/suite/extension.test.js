import assert from "assert";

// You can import and use all API from the "vscode" module
// as well as import your extension to test it
import vscode from "vscode";
// import myExtension from "../extension";

import { join } from "path";
import { EditDetector } from "../../src/file";
import { readFileSync } from "fs";

function testEditDetectorBasic() {
	const versionFiles = ["file1.txt", "file2.txt", "file3.txt"];
	const versions = versionFiles.map((fileName) => {
		const filePath = join(__dirname, "files", fileName);
		return {
			path: filePath,
			text: readFileSync()
		};
	});

	const detector = new EditDetector();
	const baseVersion = versions[0];
	
	// the first time open the file
	detector.updateSnapshot(baseVersion.path, baseVersion.text);
	// make some edits
	detector.updateEdits(baseVersion.path, versions[1].text);
	
	let edit = detector.editList[0];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 4);
	assert.equal(edit.rmLine, 3);
	assert.equal(edit.rmText, `of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
`);
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);

	edit = detector.editList[1];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 7);
	assert.equal(edit.rmLine, 0);
	assert.equal(edit.rmText, null);
	assert.equal(edit.addLine, 1);
	assert.equal(edit.addText, "aaa\r\n");

	edit = detector.editList[2];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 15);
	assert.equal(edit.rmLine, 2);
	assert.equal(edit.rmText, `FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
`);
	assert.equal(edit.addLine, 1);
	assert.equal(edit.addText, "bbb\r\n");

	edit = detector.editList[3];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 18);
	assert.equal(edit.rmLine, 1);
	assert.equal(edit.rmText, "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\r\n");
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);

	// make some edits again
	detector.updateEdits(baseVersion.path, versions[2].text);

	edit = detector.editList[0];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 4);
	assert.equal(edit.rmLine, 1);
	assert.equal(edit.rmText, "of this software and associated documentation files (the 'Software'), to deal\r\n");
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);

	edit = detector.editList[1];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 11);
	assert.equal(edit.rmLine, 0);
	assert.equal(edit.rmText, null);
	assert.equal(edit.addLine, 2);
	assert.equal(edit.addText, "aaa\r\nbbb\r\n");

	edit = detector.editList[2];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 19);
	assert.equal(edit.rmLine, 0);
	assert.equal(edit.rmText, null);
	assert.equal(edit.addLine, 1);
	assert.equal(edit.addText, "ccc\r\n");

	edit = detector.editList[3];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 6);
	assert.equal(edit.rmLine, 1);
	assert.equal(edit.rmText, "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\r\n");
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);
}

suite("Edit Detector Basic Test", testEditDetectorBasic);

suite("Extension Test Suite", () => {
	vscode.window.showInformationMessage("Start all tests.");

	test("Sample test", () => {
		assert.strictEqual(-1, [1, 2, 3].indexOf(5));
		assert.strictEqual(-1, [1, 2, 3].indexOf(0));
	});
});
