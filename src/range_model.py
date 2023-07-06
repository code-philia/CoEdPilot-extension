import sys
import math
import json

codeWindowLength = 10

def normalize_string(s):
    if type(s) != str:
        return s
    # 当检测到 s 含有 ' 时，进行转义
    if s.find("'") != -1:
        s = s.replace("'", "\'")
    return s

def RangeModel(codeWindow, commitMessage = None, prevEdit = None):
    lines = codeWindow.split('\n')
    words = ['fly', 'happy', 'good', 'car']
    preds = []
    for line in lines:
        remain = True
        for word in words:
            if word in line:
                preds.append('remove')
                remain = False
                break
        if remain:
            preds.append('remain')
    return ' '.join(preds)

def main(input):
    dict = json.loads(input)
    files = dict["files"]
    commitMessage = dict["commitMessage"]
    beforeEdit = dict["beforeEdit"]
    afterEdit = dict["afterEdit"]
    results = []

    # get targetFile content
    for file in files:
        targetFilePath = file[0]
        targetFileContent = file[1]
    
        # convert beforeEdit & afterEdit to prevEdit
        prevEdit = f'<s> Delete {beforeEdit.strip()} Add {afterEdit.strip()}'

        # get the number of lines in targetFile
        targetFileLines = targetFileContent.split('\n')
        targetFileLineNum = len(targetFileLines)

        preds = []
        for i in range(math.ceil(targetFileLineNum/codeWindowLength)):
            # get code window 
            if i == math.ceil(targetFileLineNum/codeWindowLength)-1:
                codeWindow = '\n'.join(targetFileLines[i*codeWindowLength:])
            else:
                codeWindow = '\n'.join(targetFileLines[i*codeWindowLength:(i+1)*codeWindowLength])

            # feed into the edit range suggestion model
            # the output of rangeModel: '<editType> <editType> ... <editType>'
            preds.extend(RangeModel(codeWindow, commitMessage, prevEdit).split(' '))

        if len(preds) != targetFileLineNum:
            raise ValueError(f'The number of lines ({targetFileLineNum}) in the target file is not equal to the number of predictions ({len(preds)}).')
        
        # get the edit range
        text = ''
        for i in range(targetFileLineNum):
            if preds[i] != 'remain':
                results.append({
                    "targetFilePath": targetFilePath,
                    "beforeEdit": normalize_string(beforeEdit),
                    "afterEdit": normalize_string(afterEdit),
                    "toBeReplaced": normalize_string(targetFileLines[i]),
                    "startPos": len(text),
                    "endPos": len(text)+len(targetFileLines[i]),
                    "editType": preds[i]
                })
            
            text += targetFileLines[i] + '\n'

    return json.dumps({"data": results})
        

# 读取从 Node.js 传递的文本
# 输入 Python 脚本的内容为字典格式: {"files": list, [[filePath, fileContent], ...],
#                                "targetFilePath": str, filePath,
#                                "commitMessage": str, commit message,
#								 "beforeEdit": str, content before edit,
# 								 "afterEdit": str, content after edit}
input = sys.stdin.read()
output = main(input)

# 将修改三元组作为输出发送给 Node.js
# 输出 Python 脚本的内容为字典格式: {"data": , [ { "targetFilePath": str, filePath,
#                                              "beforeEdit", str, the content before edit for previous edit,
#                                              "afterEdit", str, the content after edit for previous edit,
#                                              "toBeReplaced": str, the content to be replaced,
#                                              "startPos": int, start position of the word,
#                                              "endPos": int, end position of the word,
#                                              "editType": str, the type of edit, add or remove}, ...]}
print(output)
sys.stdout.flush()