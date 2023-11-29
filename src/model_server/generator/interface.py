import bleu
import torch
import logging
import warnings
import torch.nn as nn
from .model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from perf import Stopwatch

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

contextLength = 5
# current_file_path = os.path.dirname(os.path.abspath(__file__))
# model_name = os.path.join(current_file_path, 'pytorch_model.bin')
model_name = r"C:\Users\aaa\Desktop\models\generator\pytorch_model.bin"

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
                 edit_ops
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.edit_ops = edit_ops

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
        
def read_examples(input, labels):
    examples=[]

    code = input
    nl=''
    label_window=labels
                
    examples.append(
        Example(
                idx = 0,
                source=code,
                target = nl,
                edit_ops = label_window
                ) 
    )
    return examples

def convert_examples_to_features(examples, tokenizer, prev_preds=None, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, desc='convert examples to features')):
        #source
        # 1. add previous rejected prediction to input
        example_elements = example.source.split(' </s> ')
        if prev_preds is not None:
            prev_pred = prev_preds[example_index]
            (goldMap, predictionMap) = bleu.direct_computeMaps(prev_pred, example.target)
            bleu_score = bleu.bleuFromMaps(goldMap, predictionMap)[0]
            if bleu_score < 20: 
                example_elements.insert(2, prev_pred)
            else:
                example_elements.insert(2, ' ')
        else:
            example_elements.insert(2, ' ')
        new_example_source = ' </s> '.join(example_elements)
        source_tokens = tokenizer.tokenize(new_example_source)[:512-2]
        
        # 2. replace mask token with edit operation token
        # doing this is because sometimes the tokenizer will not split the edit operation label into single token
        edit_op_idx = 0
        for i in range(len(source_tokens)):
            if source_tokens[i] == tokenizer.mask_token:
                source_tokens[i] = example.edit_ops[edit_op_idx]
                edit_op_idx += 1
    
        # the reset is the same as original code
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = 512 - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:128-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = 128 - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
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

def load_model():
    global model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained("microsoft/codebert-base")
    tokenizer = tokenizer_class.from_pretrained("microsoft/codebert-base")
    encoder = model_class.from_pretrained("microsoft/codebert-base",config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                    beam_size=10,max_length=128,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    return model, tokenizer, device

def load_model_cache():
    global model, tokenizer, device
    model, tokenizer, device = load_model()

def predict(json_input):
    '''
    Function: interface between generator and VScode extension
    Args: input, dictionary
        { 
            "targetFileContent":    string, the whole content fo target file
            "commitMessage":        string, commit message,
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
    global model, tokenizer, device
    stopwatch = Stopwatch()

    stopwatch.start()
    # check model cache
    if not is_model_cached():
        print('+++ loading generator model')
        load_model_cache()
    stopwatch.lap('load model')

    # 提取从 JavaScript 传入的参数
    targetFileContent = json_input["targetFileContent"]
    commitMessage = json_input["commitMessage"]
    editType = json_input["editType"]
    prevEdits = json_input["prevEdits"]
    editLineIdx = json_input["atLines"]
    
    result = { # 提前记录返回的部分参数
        "editType": editType,
    }

    # 获取文本的行数
    targetFileLines = targetFileContent.splitlines(True) # 保留每行的换行符
    targetFileLineNum = len(targetFileLines)

    # 获取 editRange 的上下文
    startLineIdx = max(0, editLineIdx[0]-contextLength)
    endLineIdx = min(targetFileLineNum, editLineIdx[-1]+contextLength+1)
    stopwatch.lap('pre-process arguments')

    # 把 editRange 的上下文和 editRange 的内容拼接成 codeWindow
    codeWindow = ''
    labels = []
    for lineIdx in range(startLineIdx, endLineIdx):
        codeWindow += ' <mask> ' + targetFileLines[lineIdx]
        if lineIdx in editLineIdx:
            labels.append(editType)
        else:
            labels.append('keep')
    
    model_input = codeWindow + ' </s> '  + commitMessage 
    for prevEdit in prevEdits:
        model_input += ' </s> remove ' + prevEdit["beforeEdit"] + ' add ' + prevEdit["afterEdit"]
    stopwatch.lap('assemble input text')

    # prepare model input (tensor format)
    batch_size=1
    eval_examples = read_examples(model_input, labels)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    eval_data = TensorDataset(all_source_ids,all_source_mask)  

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    stopwatch.lap('prepare data loader')

    # run model
    replacements=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask = batch                  
        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)  
            for pred in preds[0]: # batch_size=1
                t=pred.cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                replacements.append(text)
    stopwatch.lap('infer result')

    if editType == 'add':
        replacements = [targetFileLines[editLineIdx[0]] + replacement for replacement in replacements]

    result["replacement"] = replacements
    stopwatch.lap('post-process result')
    print("+++ Generator profiling:")
    stopwatch.print_result()
    return {"data": result}