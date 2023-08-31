# -*- coding: utf-8 -*-
import os
import sys
import math
import json
import torch
import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

current_file_path = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_file_path, 'discriminator_pytorch_model.pth')

def main(input):
    dict = json.loads(input)
    rootPath = dict["rootPath"]
    files = dict["files"]
    targetFilePath = dict["targetFilePath"]

    model_inputs = []
    relativeTargetFilePath = os.path.relpath(targetFilePath, rootPath) # 将路径转换为相对路径
    for idx, (filePath, _) in enumerate(files):
        if filePath == targetFilePath:
            targetFilePathIdx = idx
        filePath = os.path.relpath(filePath, rootPath) # 将路径转换为相对路径
        if filePath <= relativeTargetFilePath:
            model_inputs.append(filePath+" <s> "+relativeTargetFilePath)
        else:
            model_inputs.append(relativeTargetFilePath+" <s> "+filePath)
    
    # 初始化tokenizer和模型
    max_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("vishnun/codenlbert-sm")
    model = AutoModelForSequenceClassification.from_pretrained("vishnun/codenlbert-sm")
    model.to(device)

    # 对输入进行编码
    encoded_data = []
    attention_mask = []
    for model_input in model_inputs:
        encoded_code = tokenizer.tokenize(model_input)[:max_length-2]
        encoded_code =[tokenizer.cls_token]+encoded_code+[tokenizer.sep_token]
        encoded_code =  tokenizer.convert_tokens_to_ids(encoded_code)
        source_mask = [1] * (len(encoded_code))
        padding_length = max_length - len(encoded_code)
        encoded_code+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        # 添加到encoded_data列表和labels列表
        encoded_data.append(encoded_code)
        attention_mask.append(source_mask)
    
    # 组成 batch
    code_batch_tensor = torch.tensor(encoded_data, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    # 创建TensorDataset
    batch_size = 128
    dataset = TensorDataset(code_batch_tensor, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 加载模型
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 预测
    predictions = []
    logits = [] # save all samples' logits
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            code_batch_tensor, attention_mask = batch
            outputs = model(code_batch_tensor, attention_mask=attention_mask).logits
            logits.extend(outputs.detach().cpu().numpy())
            batch_predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            predictions.extend(batch_predictions.detach().cpu().numpy())
    
    # 对 sample 进行排名
    probs = F.softmax(torch.tensor(logits), dim=1)
    class_1_probs = probs[:, 1] # 提取 prediction 为 1 的概率
    sorted_indices = torch.argsort(class_1_probs, descending=True).numpy() # 从大到小排列
    filtered_indices = [index for index in sorted_indices if predictions[index] == 1] # 提取 prediction 为 1 的 sample 的索引

    # 提取 prediction 为 1 的文件
    results = []
    if len(predictions) != len(files):
        raise Exception("The number of predictions is not equal to the number of files.")
    for i in filtered_indices: 
        results.append(files[i][0])
    if targetFilePathIdx not in filtered_indices:
        results.append(targetFilePath)

    # it seems that returning the content of the file would somehow cause the program to crash
    # maybe related to json.dumps, so we only return the file path instead
    return json.dumps({"data": results})


# 读取从 Node.js 传递的文本
# 输入 Python 脚本的内容为字典格式: {   "rootPath": str, "rootPath",
#                                   "files": list, [[filePath, fileContent], ...],
#                                   "targetFilePath": str, filePath,
#                                   "commitMessage": str, commit message,
#								    "prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""}}
input = sys.stdin.read()
output = main(input)

# 输出 Python 脚本的内容为字典格式: {"data": list, [[filePath, fileContent], ...]}
print(output)
sys.stdout.flush()