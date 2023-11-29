import os
import traceback
import torch
import torch.nn as nn
import re
from tqdm import tqdm
from retriv import SearchEngine
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from perf import Stopwatch

current_file_path = os.path.dirname(os.path.abspath(__file__))
# model_name = os.path.join(current_file_path, 'pytorch_model.bin')
model_name = r"C:\Users\aaa\Desktop\models\discriminator\pytorch_model.bin"

model = None
tokenizer = None
device = None

def is_model_cached():
    global tokenizer, model, device
    return not (tokenizer == None or model == None or device == None)

class CombinedModel(nn.Module):
    def __init__(self, model_name):
        super(CombinedModel, self).__init__()
        
        # Define the layers
        self.lm = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.linear = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        out = self.lm(input_ids, attention_mask).logits # (B,2)
        out = self.linear(out)
        return out.squeeze(1)
    
def load_data(inputs, tokenizer, max_length=128):
    global batch_size
    encoded_data = []
    attention_mask = []

    for idx, sample in enumerate(inputs):
        code_tokens = ' '.join(sample).replace('\n',' ')
        
        # 使用tokenizer进行编码
        # encoded_code = tokenizer.encode(code_tokens, add_special_tokens=True, padding=True, max_length=max_length)
        encoded_code = tokenizer.tokenize(code_tokens)[:max_length-2]
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
    dataset = TensorDataset(code_batch_tensor, attention_mask)

    return dataset

def load_model():
    global model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('vishnun/codenlbert-sm', model_max_length=512)
    model = CombinedModel('vishnun/codenlbert-sm')
    model.to(device)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer, device

def load_model_cache():
    global model, tokenizer, device
    model, tokenizer, device = load_model()

def predict(json_input):
    '''
    Function: this is the interface between discriminator and VSCode extension
    Args:
        input: dictionary
            {
                "files":            list, [[relativeFilePath, fileContent], ...]
                "targetFilePath":   string, the relative path of the file to be edited
                "commitMessage":    string, the commit message
                "prevEdits":        list, [{"beforeEdit": string, "afterEdit": string}, ...]
            }
    Return:
        output: dictionary, contains chosen files' path and content
            {
                "data": [string], relative file paths that are probably related to target file
            }
    '''
    global model, tokenizer, device
    stopwatch = Stopwatch()

    stopwatch.start()
    # check model cache
    if not is_model_cached():
        print('+++ loading discriminator model')
        load_model_cache()
    stopwatch.lap('load model')

    # 0. remove targetFilePath from input["files"]
    for i in range(len(json_input["files"])):
        if json_input["files"][i][0] == json_input["targetFilePath"]:
            json_input["files"].pop(i)
            break

    # 1. make code database
    stopwatch.lap('build code collection')
    
    # 2. Detech which file contain similar content to previous edits
    # store files that exist code clone into codeCloneFilePaths
    codeCloneFilePaths = set()
    edits = []
    for edit in json_input["prevEdits"]:
        edits.extend([edit["beforeEdit"], edit["afterEdit"]])

    search_re = re.compile('|'.join(map(lambda x: re.escape(x), edits)))
    for filePath, fileContent in json_input["files"]:
        found = False
        for x in search_re.finditer(fileContent):
            found = True
            break
        if found:
            codeCloneFilePaths.add(filePath)
    stopwatch.lap('find clone file paths')

    # 3. prepare input (string format)
    model_inputs = []
    for filePath, fileContent in json_input["files"]:
        if filePath in codeCloneFilePaths:
            cloneBoolean = "True"
        else:
            cloneBoolean = "False"
        model_input = ' </s> '.join([cloneBoolean, json_input["targetFilePath"], filePath, json_input["commitMessage"]])
        model_inputs.append(model_input)
    stopwatch.lap('assemble input text')

    # 4. load model

    # 5. prepare input (tensor format)
    batch_size = 128
    test_set = load_data(model_inputs, tokenizer)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    stopwatch.lap('prepare data loader')

    # 6. inference
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        batch_input, attention_mask = [item.to(device) for item in batch]
        outputs = model(input_ids=batch_input, attention_mask=attention_mask)
        preds.append(outputs.detach().cpu())
    model_outputs = (torch.cat(preds, dim=0) >= 0.5).numpy()
    stopwatch.lap('infer result')

    # 7. prepare output
    output = {"data": []}
    for idx, model_output in enumerate(model_outputs):
        if model_output == 1:
            output["data"].append(json_input["files"][idx][0])
    stopwatch.lap('post-process result')
    print("+++ Discriminator profiling:")
    stopwatch.print_result()

    return output