const { join } = require("path");
const { readFileSync } = require("fs");
const { diffLines } = require("diff");
const fs = require("fs");
const assert = require("assert");

class EditDetector {
    constructor() {
        this.editLimit = 10;
        this.textBaseSnapshots = new Map();

        /**
         * Edit list, in which all edits are based on `textBaseSnapshots`, is like:
         * [
         * 		{
         * 			"path": string, the file path,
         * 			"s": int, starting line,
         * 			"rmLine": int, number of removed lines, 
         * 			"rmText": string or null, removed text, could be null
         * 			"addLine": int, number of added lines,
         * 			"addText": string or null, number of added text, could be null
         * 		},
         * 		...
         * ]
         */
        this.editList = [];
    }

    hasSnapshot(path) {
        return this.textBaseSnapshots.has(path);
    }

    addSnapshot(path, text) {
        if (!this.hasSnapshot(path)) {
            this.textBaseSnapshots.set(path, text);
        }
    }

    async updateAllSnapshotsFromDocument() {
        for (const [path,] of this.textBaseSnapshots) {
            try {
                const text = fs.readFileSync(path, "utf-8");
                this.updateEdits(path, text);
            } catch (err) {
                console.log("Cannot update snapshot on ${path}$");
            }
        }
    }

    updateEdits(path, text) {
        // Compare old `editList` with new diff on a document
        // All new diffs should be added to edit list, but merge the overlapped/adjoined to the old ones of them 
        // Merge "-" (removed) diff into an overlapped/adjoined old edit
        // Merge "+" (added) diff into an old edit only if its precedented "-" hunk (a zero-line "-" hunk if there's no) wraps the old edit's "-" hunk
        // By default, there could only be zero or one "+" hunk following a "-" hunk
        const newDiffs = diffLines(
            this.textBaseSnapshots.get(path),
            text
        );
        const oldEditsWithIdx = [];
        const oldEditIndices = new Set();
        this.editList.forEach((edit, idx) => {
            if (edit.path === path) {
                oldEditsWithIdx.push({
                    idx: idx,
                    edit: edit
                });
                oldEditIndices.add(idx);
            }
        });
        
        oldEditsWithIdx.sort((edit1, edit2) => edit1.edit.s - edit2.edit.s);	// sort in starting line order

        const oldAdjustedEditsWithIdx = new Map();
        const newEdits = [];
        let lastLine = 1;
        let oldEditIdx = 0;

        function mergeDiff(rmDiff, addDiff) {
            const fromLine = lastLine;
            const toLine = lastLine + (rmDiff?.count ?? 0);

            // construct new edit
            const newEdit = {
                "path": path,
                "s": fromLine,
                "rmLine": rmDiff?.count ?? 0,
                "rmText": rmDiff?.value ?? null,
                "addLine": addDiff?.count ?? 0,
                "addText": addDiff?.value ?? null,
            };

            // skip all old edits between this diff and the last diff
            while (
                oldEditIdx < oldEditsWithIdx.length &&
				oldEditsWithIdx[oldEditIdx].edit.s + oldEditsWithIdx[oldEditIdx].edit.rmLine < fromLine
            ) {
                ++oldEditIdx;
                // oldAdjustedEditsWithIdx.push(oldEditsWithIdx[oldEditIdx]);
            }

            // if current edit is overlapped/adjoined with this diff
			if (
				oldEditIdx < oldEditsWithIdx.length &&
				oldEditsWithIdx[oldEditIdx].edit.s <= toLine
			) {
                // replace the all the overlapped/adjoined old edits with the new edit
                const fromIdx = oldEditIdx;
                while (
                    oldEditIdx < oldEditsWithIdx.length &&
                    oldEditsWithIdx[oldEditIdx].edit.s <= toLine
                ) {
                    ++oldEditIdx;
                }
                // use the maximum index of the overlapped/adjoined old edits	---------->  Is it necessary?
                const minIdx = Math.max.apply(
                    null,
                    oldEditsWithIdx.slice(fromIdx, oldEditIdx).map((edit) => edit.idx)
                );
                oldAdjustedEditsWithIdx.set(minIdx, newEdit);
				// // skip the edit
				// ++oldEditIdx;
            } else {
                // simply add the edit as a new edit
                newEdits.push(newEdit);
            }
        }

        for (let i = 0; i < newDiffs.length; ++i) {
            const diff = newDiffs[i];

            if (diff.removed) {
                // unite the following "+" (added) diff
                if (i + 1 < newDiffs.length && newDiffs[i + 1].added) {
                    mergeDiff(diff, newDiffs[i + 1]);
                    ++i;
                } else {
                    mergeDiff(diff, null);
                }
            } else if (diff.added) {
                // deal with a "+" diff not following a "-" diff
                mergeDiff(null, diff);
            }

			if (!(diff.added)) {
				lastLine += diff.count;
			}
        }

        const oldAdjustedEdits = [];
		this.editList.forEach((edit, idx) => {
			if (oldEditIndices.has(idx)) {
				if (oldAdjustedEditsWithIdx.has(idx)) {
					oldAdjustedEdits.push(oldAdjustedEditsWithIdx.get(idx));
				}
			} else {
				oldAdjustedEdits.push(edit);
			}
		});

		this.editList = oldAdjustedEdits.concat(newEdits);
    }

    // Shift editList if out of capacity
    // For every overflown edit, apply it and update the document snapshots on which the edits base
    shiftEdits(numShifted) {
        // filter all removed edits
        const numRemovedEdits = numShifted ?? this.editList.length - this.editLimit;
        if (numRemovedEdits <= 0) {
            return;
        }
        const removedEdits = new Set(this.editList.slice(
            0,
            numRemovedEdits
        ));
		
		function performEdits(doc, edits) {
			const lines = doc.match(/[^\r\n]*(\r?\n|\r\n|$)/g);
			const addedLines = Array(lines.length).fill("");
			for (const edit of edits) {
				const s = edit.s - 1;  // zero-based starting line
				for (let i = s; i < s + edit.rmLine; ++i) {
					lines[i] = "";
				}
				addedLines[s] = edit.addText ?? "";
			}
			return lines
			.map((x, i) => addedLines[i] + x)
			.join("");
		}
		
		// for each file involved in the removed edits
        const affectedPaths = new Set(
			[...removedEdits].map((edit) => edit.path)
			);
		for (const filePath of affectedPaths) {
			const doc = this.textBaseSnapshots.get(filePath);
			const editsOnPath = this.editList
				.filter((edit) => edit.path === filePath)
				.sort((edit1, edit2) => edit1.s - edit2.s);
				
			// execute removed edits
			const removedEditsOnPath = editsOnPath.filter((edit) => removedEdits.has(edit));
			this.textBaseSnapshots.set(filePath, performEdits(doc, removedEditsOnPath));
			
			// rebase other edits in file
			let offsetLines = 0;
			for (let edit of editsOnPath) {
				if (removedEdits.has(edit)) {
					offsetLines = offsetLines - edit.rmLine + edit.addLine;
				} else {
					edit.s += offsetLines;
				}
			}
        }

        this.editList.splice(0, numRemovedEdits);
    }

    /**
     * Return edit list in such format:
     * [
     * 		{
     * 			"beforeEdit": string, the deleted hunk, could be null;
     * 			"afterEdit": string, the added hunk, could be null;
     * 		},
     * 		...
     * ]
     */
    async getUpdatedEditList() {
        await this.updateAllSnapshotsFromDocument();
        return this.editList.map((edit) => ({
			"beforeEdit": edit.rmText?.trim() ?? "",
            "afterEdit": edit.addText?.trim() ?? ""
        }));
    }
}

function testEditDetectorBasic() {
	const versionFiles = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"];
	const versions = versionFiles.map((fileName) => {
		const filePath = join(__dirname, "files", fileName);
		return {
			path: filePath,
			text: readFileSync(filePath, "utf-8")
		};
	});

	const detector = new EditDetector();
	const baseVersion = versions[0];
	
	// the first time open the file
	detector.addSnapshot(baseVersion.path, baseVersion.text);
	// make some edits
	detector.updateEdits(baseVersion.path, versions[1].text);
	
	let edit = detector.editList[0];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 4);
	assert.equal(edit.rmLine, 3);
	assert.equal(edit.rmText, 'of this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\r\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n');
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);

	edit = detector.editList[1];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 10);
	assert.equal(edit.rmLine, 0);
	assert.equal(edit.rmText, null);
	assert.equal(edit.addLine, 1);
	assert.equal(edit.addText, "aaa\n");

	edit = detector.editList[2];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 15);
	assert.equal(edit.rmLine, 2);
	assert.equal(edit.rmText, "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\r\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\r\n");
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
	assert.equal(edit.rmText, 'of this software and associated documentation files (the "Software"), to deal\r\n');
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);

	edit = detector.editList[1];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 10);
	assert.equal(edit.rmLine, 0);
	assert.equal(edit.rmText, null);
	assert.equal(edit.addLine, 2);
	assert.equal(edit.addText, "aaa\r\nbbb\r\n");

	edit = detector.editList[2];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 18);
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

	// shift edits
	detector.shiftEdits(2);
	assert.equal(detector.textBaseSnapshots.get(baseVersion.path), versions[3].text);

	edit = detector.editList[0];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 19);
	assert.equal(edit.rmLine, 0);
	assert.equal(edit.rmText, null);
	assert.equal(edit.addLine, 1);
	assert.equal(edit.addText, "ccc\r\n");

	edit = detector.editList[1];
	assert.equal(edit.path, baseVersion.path);
	assert.equal(edit.s, 5);
	assert.equal(edit.rmLine, 1);
	assert.equal(edit.rmText, "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\r\n");
	assert.equal(edit.addLine, 0);
	assert.equal(edit.addText, null);

}

testEditDetectorBasic();

// suite("Edit Detector Basic Test", testEditDetectorBasic);

// suite("Extension Test Suite", () => {
// 	vscode.window.showInformationMessage("Start all tests.");

// 	test("Sample test", () => {
// 		assert.strictEqual(-1, [1, 2, 3].indexOf(5));
// 		assert.strictEqual(-1, [1, 2, 3].indexOf(0));
// 	});
// });
