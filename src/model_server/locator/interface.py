# import json
import torch
import math

from .model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from perf import Stopwatch
from model_manager import load_model_with_cache

CODE_WINDOW_LENGTH = 10
MODEL_ROLE = "locator"

model = None
tokenizer = None
device = None

def is_model_cached():
    global tokenizer, model, device
    return not (tokenizer == None or model == None or device == None)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask  
        
def read_examples(raw_inputs):
    examples = []
    for idx, sample in enumerate(raw_inputs):
        code=sample
        label=''       
        examples.append(
            Example(
                    idx = idx,
                    source=code,
                    target=label
                    ) 
        )
    return examples

def convert_examples_to_features(examples, tokenizer, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples)):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:512-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        original_source_len = len(source_ids)
        source_mask = [1] * (len(source_tokens))
        padding_length = 512 - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.source)[:512-2]
        label_idx = 0
        # replace mask token with label token
        for i in range(len(target_tokens)):
            if target_tokens[i] == tokenizer.mask_token:
                target_tokens[i]=example.target[label_idx]
                label_idx+=1
        
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        original_target_len = len(target_ids)
        target_mask = [1] *len(target_ids)
        padding_length = 512 - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        # if original_source_len != original_target_len:
            # print(example.source)
            # print(example.target)
            # print('source length: ', original_source_len)
            # print('target length: ', original_target_len)
            # break

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_class, model_class, tokenizer_class = (RobertaConfig, RobertaModel, RobertaTokenizer)
    config = config_class.from_pretrained("microsoft/codebert-base")
    tokenizer = tokenizer_class.from_pretrained("microsoft/codebert-base")
    encoder = model_class.from_pretrained("microsoft/codebert-base", config=config)
    model = Seq2Seq(encoder=encoder,config=config,
                    beam_size=10,max_length=512,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id,mask_id=tokenizer.mask_token_id)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model, tokenizer, device

def load_model_cache():
    global model, tokenizer, device
    model, tokenizer, device = load_model()

def normalize_string(s):
    if type(s) != str:
        return s
    # 当检测到 s 含有 ' 时，进行转义
    if s.find("'") != -1:
        s = s.replace("'", "\'")
    return s

def merge_adjacent_removals(results):
    sorted_results = sorted(results, key=lambda x: (x["targetFilePath"], x["atLines"][0]))  # 按照目标文件路径和起始位置对元素进行排序
    merged_results = []

    def can_merge(last_result, this_result):
        return last_result and \
            last_result["atLines"][-1] == this_result["atLines"][0] - 1 and \
            last_result["editType"] == this_result["editType"]

    for mod in sorted_results:
        if len(merged_results) > 0 and can_merge(merged_results[-1], mod):
            merged_results[-1]["atLines"].append(mod["atLines"][0])
        else:
            merged_results.append(mod)

    return merged_results

def predict(json_input, language):
    '''
    Function: interface between locator and VScode extension
    Args:
        input: dictionary
            {
                "files":            list, [[filePath, fileContent], ...],
                "targetFilePath":   str, filePath,
                "commitMessage":    str, commit message,
                "prevEdits":        list, of previous edits, each in format: {"beforeEdit": string, "afterEdit":string}
            }
    Returns:
        output: dictionary
            {
                "data": [ 
                    {   
                        "targetFilePath":   str, filePath,
                        "editType":         str, the type of edit, add or replace,
                        "lineBreak":        str, '\n', '\r' or '\r\n',
                        "atLines":           list, numbers of the line indices of to be replaced code 
                    }, 
                    ...
                ]
            }
    '''
    stopwatch = Stopwatch()

    stopwatch.start()
    # check model cache
    model, tokenizer, device = load_model_with_cache(MODEL_ROLE, language, load_model)
    stopwatch.lap('load model')

    # 提取从 JavaScript 传入的参数
    files = json_input["files"]
    commitMessage = json_input["commitMessage"]
    prevEdits = json_input["prevEdits"]
    results = []
    # print("+++ Prev Edits:")
    # print(json.dumps(prevEdits, indent=4))

    window_token_cnt = 0
    window_line_cnt = 0
    window_text = ""
    def try_feed_in_window(text):
        nonlocal window_token_cnt, window_line_cnt, window_text
        masked_line = " <mask> " + text
        masked_line_token_cnt = len(tokenizer.tokenize(masked_line))
        if window_token_cnt + masked_line_token_cnt < 508 and window_line_cnt < 10: # a conservative number for token number
            window_token_cnt += masked_line_token_cnt
            window_line_cnt += 1
            window_text += masked_line
            return True
        else:
            return False
    def end_window(input_list):
        nonlocal prevEdits, commitMessage, window_token_cnt, window_line_cnt, window_text
        model_input = window_text + ' </s> '  + commitMessage 
        for prevEdit in prevEdits:
            model_input +=' </s> replace ' + prevEdit["beforeEdit"] + ' add ' + prevEdit["afterEdit"]
        input_list.append(model_input)
        # with open(r"C:\Users\aaa\Desktop\edit-pilot\mark.txt", "a+", newline='') as f:
        #     f.write(model_input)
        #     f.write("\n")
        window_token_cnt = 0
        window_line_cnt = 0
        window_text = ""

    # 获取每个文件的内容
    for file in files:
        targetFilePath = file[0] 
        targetFileContent = file[1]
        # 获取文件行数
        targetFileLines = targetFileContent.splitlines(True) # 保留每行的换行符
        targetFileLineNum = len(targetFileLines)

        model_inputs = []
        # for windowIdx in range(math.ceil(targetFileLineNum/codeWindowLength)):
        #     # 按照 codeWindowLength 将文件内容分割成 codeWindow
        #     if windowIdx == math.ceil(targetFileLineNum/codeWindowLength)-1:
        #         codeWindowLines = targetFileLines[windowIdx*codeWindowLength:]
        #     else:
        #         codeWindowLines = targetFileLines[windowIdx*codeWindowLength:(windowIdx+1)*codeWindowLength]
        #     codeWindowLines = [" <mask> " + line for line in codeWindowLines]
        #     codeWindow = ''.join(codeWindowLines)

        #     # 将 CodeWindow， CommitMessage 和 prevEdit 合并为一个字符串，作为模型的输入
        #     model_input = codeWindow + ' </s> '  + commitMessage 
        #     for prevEdit in prevEdits:
        #         model_input +=' </s> replace ' + prevEdit["beforeEdit"] + ' add ' + prevEdit["afterEdit"]
        #     model_inputs.append(model_input)

        i = 0
        while i < targetFileLineNum:
            cur_line = targetFileLines[i]
            if try_feed_in_window(cur_line):
                i += 1
            else:
                if window_line_cnt == 0:    # the first line is longer than window limit 
                    while True:
                        cur_line = cur_line[:len(cur_line)/2]
                        if try_feed_in_window(cur_line):
                            break
                else:
                    end_window(model_inputs)
        if len(window_text) > 0:
            end_window(model_inputs)
        stopwatch.lap_by_task('assemble input text')

        # prepare model input (tensor format)
        examples=read_examples(model_inputs)
        eval_features=convert_examples_to_features(examples, tokenizer, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)  
        all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids,all_source_mask, all_target_ids,all_target_mask)   

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=10, shuffle=False)

        # run model
        model.eval() 
        preds = []
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask= batch                  
            with torch.no_grad():
                lm_logits = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask, train=False).to('cpu')
                # extract masked edit operations
                for i in range(lm_logits.shape[0]): # for sample within batch
                    output = []
                    for j in range(lm_logits.shape[1]): # for every token
                        if source_ids[i][j]==tokenizer.mask_token_id: # if is masked
                            output.append(tokenizer.decode(torch.argmax(lm_logits[i][j]),clean_up_tokenization_spaces=False))
                    preds.extend(output)

        if len(preds) != targetFileLineNum:
            raise ValueError(f'The number of lines ({targetFileLineNum}) in the target file is not equal to the number of predictions ({len(preds)}).') # TODO: solve this problem when some lines are too long
        stopwatch.lap_by_task('infer result')

        # print(f"+++ Target file lines:\n{''.join([preds[i] + '    ' + targetFileLines[i] for i in range(targetFileLineNum)])}")
        # get the edit range
        # text = ''
        for i in range(targetFileLineNum):
            if preds[i] != 'keep': # 如果模型输出的 editType 不是 keep，则该行需要被修改
                if targetFileLines[i].endswith('\r\n'):
                    lineBreak = '\r\n'
                elif targetFileLines[i].endswith('\n'):
                    lineBreak = '\n'
                elif targetFileLines[i].endswith('\r'):
                    lineBreak = '\r'
                else:
                    lineBreak = ''
                
                results.append({
                    "targetFilePath": targetFilePath,
                    # "prevEdits": prevEdits,
                    # "toBeReplaced": normalize_string(targetFileLines[i].rstrip("\n\r")), # 高亮的部分不包括行尾的换行符
                    # "startPos": len(text),
                    # "endPos": len(text)+len(targetFileLines[i].rstrip("\n\r")), # 高亮的部分不包括行尾的换行符
                    "editType": preds[i],
                    "lineBreak": lineBreak,
                    "atLines": [i] # 行数从 0 开始
                })
            
            # text += targetFileLines[i]
        stopwatch.lap_by_task('prepare result')

    results = merge_adjacent_removals(results)
    stopwatch.lap('post-process result')
    print("+++ Locator profiling:")
    stopwatch.print_result()

    return {"data": results}