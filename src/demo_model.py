import re
import sys
import json
import random

def main(input):
    # read input from vscode extension
    dict = json.loads(input)
    files = dict["files"]
    beforeEdit = dict["beforeEdit"]
    afterEdit = dict["afterEdit"]
    filePath = files[0][0]

    # open json for next edit
    with open('/Users/russell/Downloads/Code-Edit-main/src/edit.json') as f:
        edits = json.load(f)['edit']

    results = []
    if beforeEdit.strip() == '''_FlagNoScan = 1 << 0 // GC doesn't have to scan object''':
        for edit in edits:
            if edit[3] == 0:
                results.append([filePath, edit[0], edit[1], edit[2]])
    elif beforeEdit.strip() == '''_FlagNoZero = 1 << 1 // don't zero memory''':
        for edit in edits:
            if edit[3] == 1:
                results.append([filePath, edit[0], edit[1], edit[2]])

    return {'data':results}


# 读取从 Node.js 传递的文本
input = sys.stdin.read()

# 处理文本并生成修改三元组
modifications = main(input)

# 将修改三元组作为输出发送给 Node.js
print(modifications)
sys.stdout.flush()

