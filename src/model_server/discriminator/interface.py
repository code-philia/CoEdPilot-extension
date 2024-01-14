import os
import torch
import torch.nn as nn
import re
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from perf import Stopwatch
from model_manager import load_model_with_cache

MODEL_ROLE = "discriminator"
OUTPUT_MAX = 100
WORD_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
LANGUAGE_KEYWORDS = {
    "go": set(["break", "default", "func", "interface", "select", "case", "defer", "go", "map", "struct",
           "chan", "else", "goto", "package", "switch", "const", "fallthrough", "if", "range", "type",
           "continue", "for", "import", "return", "var"]),
    "python": set(["False", "await", "else", "import", "pass", "None", "break", "except", "in", "raise",
               "True", "class", "finally", "is", "return", "and", "continue", "for", "lambda", "try",
               "as", "def", "from", "nonlocal", "while", "assert", "del", "global", "not", "with",
               "async", "elif", "if", "or", "yield"]),
    "javascript": set(["break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete",
                   "do", "else", "export", "extends", "finally", "for", "function", "if", "import",
                   "in", "instanceof", "new", "return", "super", "switch", "this", "throw", "try",
                   "typeof", "var", "void", "while", "with", "yield"]),
    "typescript": set(["break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete",
                   "do", "else", "enum", "export", "extends", "finally", "for", "function", "if", "implements",
                   "import", "in", "instanceof", "interface", "let", "new", "package", "private", "protected",
                   "public", "return", "static", "super", "switch", "this", "throw", "try", "typeof",
                   "var", "void", "while", "with", "yield", "as", "asserts", "any", "async", "await",
                   "boolean", "constructor", "declare", "get", "infer", "is", "keyof", "module", "namespace",
                   "never", "readonly", "require", "number", "object", "set", "string", "symbol", "type",
                   "undefined", "unique", "unknown"]),
    "java": set(["abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
             "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
             "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
             "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
             "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile",
             "while"])
}

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
    token_cnt = 0

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
        
        token_cnt += len(encoded_code)

    # 组成 batch
    code_batch_tensor = torch.tensor(encoded_data, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    
    # 创建TensorDataset
    dataset = TensorDataset(code_batch_tensor, attention_mask)

    return dataset, token_cnt

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('vishnun/codenlbert-sm', model_max_length=512)
    model = CombinedModel('vishnun/codenlbert-sm')
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer, device


def predict(json_input, language):
    '''
    Function: this is the interface between discriminator and VSCode extension
    Args:
        input: dictionary
            {
                "files":            list, [[relativeFilePath, fileContent], ...]
                "targetFilePath":   string, the relative path of the file to be edited
                "commitMessage":    string, the edit description
                "prevEdits":        list, [{"beforeEdit": string, "afterEdit": string}, ...]
            }
    Return:
        output: dictionary, contains chosen files' path and content
            {
                "data": [string], relative file paths that are probably related to target file
            }
    '''
    stopwatch = Stopwatch()

    stopwatch.start()
    # check model cache
    model, tokenizer, device = load_model_with_cache(MODEL_ROLE, language, load_model)
    stopwatch.lap('load model')

    # 0. remove targetFilePath from input["files"]
    # for i in range(len(json_input["files"])):
    #     if json_input["files"][i][0] == json_input["targetFilePath"]:
    #         json_input["files"].pop(i)
    #         break

    # 1. make code database
    stopwatch.lap('build code collection')
    
    # 2. Detech which file contain similar content to previous edits
    # store files that exist code clone into codeCloneFilePaths
    codeCloneFilePaths = set()
    edits = []
    for edit in json_input["prevEdits"]:
        edits.extend([edit["beforeEdit"], edit["afterEdit"]])

    keyword_set = LANGUAGE_KEYWORDS[language]
    edit_words_list = np.array([np.array(
        [y for y in WORD_PATTERN.findall(x) if y not in keyword_set]
        ) for x in edits], dtype=object)
    edit_words = []
    if len(edit_words_list):
        edit_words = np.concatenate(edit_words_list)
    search_re = re.compile('|'.join(map(lambda x: re.escape(x), edit_words)))
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
    test_set, token_cnt = load_data(model_inputs, tokenizer)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    stopwatch.lap('prepare data loader', token_cnt)

    # 6. inference
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        batch_input, attention_mask = [item.to(device) for item in batch]
        outputs = model(input_ids=batch_input, attention_mask=attention_mask)
        preds.append(outputs.detach().cpu())
    model_outputs = (torch.cat(preds, dim=0) >= 0.5).numpy()
    stopwatch.lap('infer result', token_cnt)

    # 7. prepare output
    output = {"data": []}
    for idx, model_output in enumerate(model_outputs):
        if model_output == 1:
            output["data"].append(json_input["files"][idx][0])
        if len(output["data"]) >= OUTPUT_MAX:
            break
    stopwatch.lap('post-process result')
    print("+++ Discriminator profiling:")
    stopwatch.print_result(len(json_input["files"]), 'disc_stat.txt')

    return output    

load_model_with_cache(MODEL_ROLE, "python", load_model)
