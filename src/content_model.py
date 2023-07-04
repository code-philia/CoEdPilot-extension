import sys
import json

codeWindowLength = 10

def ContentModel(codeWindow, commitMessage=None, prevEdit=None):
    # extract the edit line from codeWindow
    editLine = codeWindow.split('<s>')[1][1:]
    
    replace_dict = {
        'happy': 'sad',
        'good': 'bad',
        'car': 'bike',
        'fly': 'dive'
    }
    for keyword, replacement in replace_dict.items():
        editLine = editLine.replace(keyword, replacement)

    return editLine

def main(input):
    # read input from vscode extension
    dict = json.loads(input)
    files = dict["files"]
    targetFilePath = dict["targetFilePath"]
    commitMessage = dict["commitMessage"]
    editType = dict["editType"]
    beforeEdit = dict["beforeEdit"]
    afterEdit = dict["afterEdit"]
    startPos = dict["startPos"]
    endPos = dict["endPos"]
    
    result = {
        "targetFilePath": targetFilePath,
        "startPos": startPos,
        "endPos": endPos
    }

    # convert beforeEdit & afterEdit to prevEdit
    prevEdit = f'<s> Delete {beforeEdit.strip()} Add {afterEdit.strip()}'

    # get targetFileContent
    for file in files:
        if file[0] == targetFilePath:
            targetFileContent = file[1]
            break

    # the the line number of the edit range from startPos ~ endPos
    targetFileLines = targetFileContent.split('\n')
    targetFileLineNum = len(targetFileLines)
    text = ''
    for i in range(targetFileLineNum):
        if len(text) == startPos and len(text)+len(targetFileLines[i]) == endPos:
            editLineIdx = i # editLineIdx count from 0
            break
        text += targetFileLines[i] + '\n'

    # get the the line range of codeWindow
    if targetFileLineNum <= codeWindowLength:
        startLineIdx = 0
        endLineIdx = targetFileLineNum-1
    else:
        startLineIdx = max(0, editLineIdx-(codeWindowLength//2-1))
        endLineIdx = min(targetFileLineNum-1, editLineIdx+(codeWindowLength//2))
        if endLineIdx - startLineIdx != codeWindowLength:
            if startLineIdx == 0:
                endLineIdx = codeWindowLength
            elif endLineIdx == targetFileLineNum-1:
                startLineIdx = endLineIdx - codeWindowLength
    
    # get codeWindow, with the edit line marked by <s>
    codeWindow = ''
    for lineIdx in range(startLineIdx, endLineIdx):
        if lineIdx == editLineIdx:
            codeWindow += f'<s> {targetFileLines[lineIdx]} <s>\n'
        elif lineIdx != editLineIdx and lineIdx != endLineIdx-1:
            try:
                codeWindow += f'{targetFileLines[lineIdx]}\n'
            except:
                raise ValueError(len(targetFileLines), lineIdx)
        else:
            codeWindow += f'{targetFileLines[lineIdx]}'
    
    replacement = ContentModel(codeWindow, commitMessage, prevEdit)

    if editType == 'add':
        replacement = targetFileLines[editLineIdx] + '\n' + replacement

    result["replacement"] = replacement

    return json.dumps({"data": result})  

# 读取从 Node.js 传递的文本
# 输入 Python 脚本的内容为字典格式: { "files": list, [[filePath, fileContent], ...],
#                                 "targetFilePath": string filePath,
#                                 "commitMessage": string, commit message,
#                                 "editType": str, the type of edit,
#                                 "beforeEdit": string, before edit content in previous edit,
#                                 "afterEdit": string, after edit content in previous edit,
#                                 "startPos": int, start position,
#                                 "endPos": int, end position}
input = sys.stdin.read()
output = main(input)

# 将修改三元组作为输出发送给 Node.js
# 输出 Python 脚本的内容为字典格式: {"data": 
#                                       { "targetFilePath": string, filePath of target file,
#                                         "startPos": int, start position,
#                                         "endPos": int, end position,
#                                         "replacement": string, replacement content   
#                                       }
#                               }
print(output)
sys.stdout.flush()