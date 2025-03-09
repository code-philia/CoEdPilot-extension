import json
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
from perf import Stopwatch
from model_manager import load_model_with_cache

CONTEXT_LENGTH = 5
MODEL_ROLE = "generator"


def is_model_cached():
    global tokenizer, model, device
    return not (tokenizer is None or model is None or device is None)

def load_model(model_path, device):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # M chip acceleration
    else:
        device = torch.device("cpu")
        
    config_class, model_class, tokenizer_class = (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
    config = config_class.from_pretrained("salesforce/codet5-base")
    tokenizer = tokenizer_class.from_pretrained("salesforce/codet5-base")
    model = model_class.from_pretrained("salesforce/codet5-base",config=config)
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "<replace-by>", "</replace-by>",
                          "<feedback>", "</feedback>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    config.vocab_size = len(tokenizer)
    # if directly using model.encoder.resize_token_embeddings(), in some cases, will also change the shape of decoder embedding
    new_encoder_embedding = nn.Embedding(config.vocab_size, config.d_model)
    model.encoder.embed_tokens = new_encoder_embedding
    
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, tokenizer, device

async def predict(json_input):
    '''
    Function: interface between generator and VScode extension
    Args: input, dictionary
        {
            "targetFileContent":    string, the whole content fo target file
            "commitMessage":        string, edit description,
            "editType":             str, the type of edit,
            "prevEdits":            list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
            "atLines":               list, of edit line indices
        }
    Return: dictionary
        {
            "data":
            {
                "editType":     string, "replace" or "add",
                "replacement":  [string], list of replacing candidates,
            }
        }
    '''
    stopwatch = Stopwatch()

    stopwatch.start()
    # check model cache
    language = "multilingual"
    model, tokenizer, device = await load_model_with_cache(MODEL_ROLE, language, load_model)
    stopwatch.lap('load model')

    # 提取从 JavaScript 传入的参数
    targetFileContent = json_input["targetFileContent"]
    commitMessage = json_input["commitMessage"]
    editType = json_input["editType"]
    prevEdits = json_input["prevEdits"]
    editLineIdx = json_input["atLines"]

    result = {  # 提前记录返回的部分参数
        "editType": editType,
    }

    # 获取文本的行数
    targetFileLines = targetFileContent.splitlines(False) # 保留每行的换行符
    targetFileLineNum = len(targetFileLines)

    # 获取 editRange 的上下文
    startLineIdx = max(0, editLineIdx[0] - CONTEXT_LENGTH)
    endLineIdx = min(targetFileLineNum, editLineIdx[-1] + CONTEXT_LENGTH + 1)
    stopwatch.lap('pre-process arguments')

    # 把 editRange 的上下文和 editRange 的内容拼接成 codeWindow
    codeWindow = ""
    for lineIdx in range(startLineIdx, endLineIdx):
        if lineIdx in editLineIdx:
            if editType == "add":
                label = "<insert>"
            elif editType == "replace":
                label = "<replace>"
            else:
                raise ValueError(f"Unsupported edit type: {editType}")
            codeWindow += f"{label}{targetFileLines[lineIdx]}"
        else:
            codeWindow += f"<keep>{targetFileLines[lineIdx]}"
    
    model_input = f"<code_window>{codeWindow}</code_window><prompt>{commitMessage}</prompt><prior_edits>"
    for prevEdit in prevEdits:
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
    
    print("Generator input:")
    print(model_input)
    encoded_source_seq = tokenizer(model_input, padding="max_length", truncation=True, max_length=512)
    source_ids = encoded_source_seq["input_ids"]
    stopwatch.lap('assemble input text')

    # prepare model input (tensor format)
    batch_size=1
    beam_size=10
    all_source_ids = torch.tensor([source_ids], dtype=torch.long)  
    eval_data = TensorDataset(all_source_ids)  

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=batch_size)
    stopwatch.lap('prepare data loader')

    # run model
    replacements = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids = batch[0]
        # print(source_ids.shape)
        source_mask = source_ids.ne(tokenizer.pad_token_id)         
        with torch.no_grad():
            preds = model.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    num_beams=beam_size,
                                    max_length=256,
                                    num_return_sequences=beam_size)
            preds = preds.reshape(source_ids.size(0), beam_size, -1)
            preds = preds.cpu().numpy()
            replacements=[]
            for pred in preds[0]: # batch_size=1
                replacements.append(tokenizer.decode(pred, skip_special_tokens=True,clean_up_tokenization_spaces=False))
    # remove the line break at the end of each replacement
    replacements = [s.strip("\n\r") for s in replacements]
    stopwatch.lap('infer result')

    result["replacement"] = replacements
    print(f"Generator output: \n{replacements[0]}")
    stopwatch.lap('post-process result')
    print("+++ Generator profiling:")
    stopwatch.print_result()
    return {"data": result}
