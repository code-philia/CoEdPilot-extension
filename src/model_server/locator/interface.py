# import json
import torch
import json

from .model import Locator
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
from perf import Stopwatch
from model_manager import load_model_with_cache

CODE_WINDOW_LENGTH = 10
MODEL_ROLE = "locator"

model = None
tokenizer = None
device = None


def is_model_cached():
    global tokenizer, model, device
    return not (tokenizer is None or model is None or device == None)


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
        code = sample
        label = ''
        examples.append(
            Example(
                idx=idx,
                source=code,
                target=label
                )
        )
    return examples


def convert_examples_to_features(examples, tokenizer, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:512 - 2]
        source_tokens = [tokenizer.cls_token] + \
            source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        original_source_len = len(source_ids)
        source_mask = [1] * (len(source_tokens))
        padding_length = 512 - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.source)[:512 - 2]
        label_idx = 0
        # replace mask token with label token
        for i in range(len(target_tokens)):
            if target_tokens[i] == tokenizer.mask_token:
                target_tokens[i] = example.target[label_idx]
                label_idx += 1

        target_tokens = [tokenizer.cls_token] + \
            target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        original_target_len = len(target_ids)
        target_mask = [1] * len(target_ids)
        padding_length = 512 - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

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


def load_model(model_path, device):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # M chip acceleration
    else:
        device = torch.device("cpu")
    # Load pre-training models
    config_class, model_class, tokenizer_class = (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
    config = config_class.from_pretrained("salesforce/codet5-large")
    tokenizer = tokenizer_class.from_pretrained("salesforce/codet5-large")
    codeT5 = model_class.from_pretrained("salesforce/codet5-large") 
    encoder = codeT5.encoder
    
    # add customization tokens
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "</insert>","<replace-by>", "</replace-by>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    encoder.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    
    model = Locator(encoder=encoder,config=config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, tokenizer, device


def normalize_string(s):
    if not isinstance(s, str):
        return s
    # 当检测到 s 含有 ' 时，进行转义
    if s.find("'") != -1:
        s = s.replace("'", "\'")
    return s


def merge_adjacent_removals(results):
    sorted_results = sorted(
        results,
        key=lambda x: (
        x["targetFilePath"],
        x["atLines"][0]))  # 按照目标文件路径和起始位置对元素进行排序
    merged_results = []

    def can_merge(last_result, this_result):
        return last_result and \
            last_result["atLines"][-1] == this_result["atLines"][0] - 1 and \
            last_result["editType"] == this_result["editType"]

    for mod in sorted_results:
        if len(merged_results) > 0 and can_merge(merged_results[-1], mod):
            merged_length = len(merged_results[-1]["atLines"])
            mod_length = len(mod["atLines"])
            merged_results[-1]["confidence"] = (merged_length * merged_results[-1]["confidence"] + mod_length * mod["confidence"]) / (merged_length + mod_length)
            merged_results[-1]["atLines"].append(mod["atLines"][0])
        else:
            merged_results.append(mod)

    return merged_results

async def predict(json_input):
    '''
    Function: interface between locator and VScode extension
    Args:
        input: dictionary
            {
                "files":            list, [[filePath, fileContent], ...],
                "targetFilePath":   str, filePath,
                "commitMessage":    str, edit description,
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
    language = "multilingual"
    model, tokenizer, device = await load_model_with_cache(MODEL_ROLE, language, load_model)
    stopwatch.lap('load model')

    # 提取从 JavaScript 传入的参数
    files = json_input["files"]
    commitMessage = json_input["commitMessage"]
    prevEdits = json_input["prevEdits"][::-1]
    results = []

    window_token_cnt = 0
    window_line_cnt = 0
    window_text = ""

    def security_checked(text, tokenizer):
        return text.replace(tokenizer.mask_token, "\<mask\>")

    def try_feed_in_window(text):
        nonlocal window_token_cnt, window_line_cnt, window_text
        masked_line = f"{tokenizer.mask_token}" + security_checked(text, tokenizer)
        masked_line_token_cnt = len(tokenizer.tokenize(masked_line))
        # a conservative number for token number
        if window_token_cnt + masked_line_token_cnt < 508 and window_line_cnt < 10:
            window_token_cnt += masked_line_token_cnt
            window_line_cnt += 1
            window_text += masked_line
            return True
        else:
            return False

    def end_window(input_list):
        nonlocal prevEdits, commitMessage, window_token_cnt, window_line_cnt, window_text

        model_input = f"<code_window>{window_text}</code_window>" + "<prompt>" + commitMessage + "</prompt>"
        model_input += "<prior_edits>"
        if prevEdits:
            for prevEdit in prevEdits[:3]:
                codeAbove = prevEdit["codeAbove"].splitlines(keepends=True)
                codeAbove = "".join(["<keep>" + loc for loc in codeAbove])
                beforeEdit = prevEdit["beforeEdit"].splitlines(keepends=True)
                if len(beforeEdit) == 0:
                    editType = "insert"
                else:
                    editType = "replace"
                beforeEdit = "".join(["<replace>" + loc for loc in beforeEdit])
                codeBelow = prevEdit["codeBelow"].splitlines(keepends=True)
                codeBelow = "".join(["<keep>" + loc for loc in codeBelow])
                
                if editType == "replace":
                    model_input += "<edit>" + codeAbove + "<replace-by>" + prevEdit["afterEdit"] +"</replace-by>" + beforeEdit + codeBelow + "</edit>"
                else:
                    model_input += "<edit>" + codeAbove + "<insert>" + prevEdit["afterEdit"] + "</insert>" + codeBelow + "</edit>"
        model_input += "</prior_edits>"
        
        input_list.append(model_input)
        window_token_cnt = 0
        window_line_cnt = 0
        window_text = ""

    # 获取每个文件的内容
    for file in files:
        targetFilePath = file[0]
        targetFileContent = file[1]
        # 获取文件行数
        targetFileLines = targetFileContent.splitlines(True)  # 保留每行的换行符
        targetFileLineNum = len(targetFileLines)

        model_inputs = []

        i = 0
        while i < targetFileLineNum:
            cur_line = targetFileLines[i]
            if try_feed_in_window(cur_line):
                i += 1
            else:
                if window_line_cnt == 0:    # the first line is longer than window limit
                    while True:
                        cur_line = cur_line[:len(cur_line) // 2]
                        if try_feed_in_window(cur_line):
                            i += 1
                            break
                else:
                    end_window(model_inputs)
        if len(window_text) > 0:
            end_window(model_inputs) 
        # print("Locator input text:")
        # print(json.dumps(model_inputs, indent=4))
        stopwatch.lap_by_task('assemble input text')

        # prepare model input (tensor format)
        examples=read_examples(model_inputs)
        eval_features=convert_examples_to_features(examples, tokenizer, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)  
        all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids,all_source_mask, all_target_ids)   

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data,
            sampler=eval_sampler,
            batch_size=10,
            shuffle=False)

        # run model
        model.eval()
        preds = []
        confidences = []
        softmax = torch.nn.Softmax(dim=-1)

        for batch in tqdm(eval_dataloader,total=len(eval_dataloader), desc=targetFilePath):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids = batch                  
            with torch.no_grad():
                lm_logits = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids).to('cpu')
                # extract masked edit operations
                for i in range(lm_logits.shape[0]):  # for sample within batch
                    output = []
                    confidence = []
                    for j in range(lm_logits.shape[1]): # for every token
                        softmax_output = softmax(lm_logits[i][j])
                        if source_ids[i][j]==tokenizer.mask_token_id: # decode masked edit operation token
                            max_confidence_token_idx = torch.argmax(softmax_output)
                            max_confidence_token = tokenizer.decode(max_confidence_token_idx, clean_up_tokenization_spaces=False)
                            if max_confidence_token == "<replace>" and torch.max(softmax_output) < 0.90:
                                output.append("<keep>")
                                confidence.append(1.0)
                            elif max_confidence_token == "<insert>" and torch.max(softmax_output) < 0.98:
                                output.append("<keep>")
                                confidence.append(1.0)
                            else:
                                output.append(max_confidence_token)
                                confidence.append(softmax_output[max_confidence_token_idx].item())
                    preds.extend(output)
                    confidences.extend(confidence)

        if len(preds) != targetFileLineNum:
            raise ValueError(f'The number of lines ({targetFileLineNum}) in the target file is not equal to the number of predictions ({len(preds)}).') # TODO: solve this problem when some lines are too long
        if len(confidences) != targetFileLineNum:
            raise ValueError(f'The number of lines ({targetFileLineNum}) in the target file is not equal to the number of confidences ({len(confidences)}).')
        stopwatch.lap_by_task('infer result')

        # print(json.dumps(preds, indent=4))
        for i, (pred, conf) in enumerate(zip(preds, confidences)):
            if pred != '<keep>': # 如果模型输出的 editType 不是 keep，则该行需要被修改
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
                    "editType": "replace" if pred == "<replace>" else "add",
                    "lineBreak": lineBreak,
                    "atLines": [i], # 行数从 0 开始
                    "confidence": conf
                })
            
        stopwatch.lap_by_task('prepare result')

    results = merge_adjacent_removals(results)
    stopwatch.lap('post-process result')
    print("+++ Locator profiling:")
    stopwatch.print_result()
    return {"data": results}

# load model when backend starts to run
import asyncio
asyncio.run(load_model_with_cache(MODEL_ROLE, "multilingual", load_model))