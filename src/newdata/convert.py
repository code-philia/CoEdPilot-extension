import json
import sys
import os
import Levenshtein

def move_code():
    data={}
    for i in range(8):
        file=os.path.join('E:\CodeGeneration-demo\src\data','code'+str(i)+'.js')
        with open(file,'r') as f:
            code=f.read()
        file=os.path.join('E:\CodeGeneration-demo\src\data','truth'+str(i)+'.js')
        with open(file,'r') as f:
            truth=f.read()
        res = Levenshtein.editops(code, truth)
        edit=[list(i) for i in res]
        edit=handle_edit(code,truth,edit)
        
        data[str(i)]={'code':code,'new':truth,'edit':edit}
    with open('E:\CodeGeneration-demo\src\data\code-edit.json','w') as f:
        json.dump(data,f,indent=4)

def handle_edit(code,truth,edit):
    for i in range(len(edit)):
        if edit[i][0]=='replace':
            edit[i][1]+=code.count('\n',0,edit[i][1])
            edit[i][2]=truth[edit[i][2]]
        elif edit[i][0]=='delete':
            edit[i][1]+=code.count('\n',0,edit[i][1])
        elif edit[i][0]=='insert':
            edit[i][1]+=code.count('\n',0,edit[i][1])
            edit[i][2]=truth[edit[i][2]]
        
    return edit
    
def change_edit(idx,edit):
    with open('E:\CodeGeneration-demo\src\data\code-edit.json','r') as f:
        data=json.load(f)
    data[str(idx)]['edit']=edit
    with open('E:\CodeGeneration-demo\src\data\code-edit.json','w') as f:
        json.dump(data,f,indent=4)

def genEdits():
    data={}
    for i in range(8):
        file=os.path.join('E:\CodeGeneration-demo\src\data','code'+str(i)+'.js')
        with open(file,'r') as f:
            code=f.read()
        file=os.path.join('E:\CodeGeneration-demo\src\data','truth'+str(i)+'.js')
        with open(file,'r') as f:
            truth=f.read()
        


        data[str(i)]={'code':code,'new':truth,'edit':''}
    with open('E:\CodeGeneration-demo\src\data\code-edit.json','w') as f:
        json.dump(data,f,indent=4)

data = [
    ['insert', 130, 130],
    ['delete', 253, 254],
    ['delete', 254, 254],
    ['delete', 285, 284],
    ['insert', 532, 530]

]

move_code()
