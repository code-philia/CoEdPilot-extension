import sys
import json
import torch
import logging
import warnings
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

contextLength = 5
model_name = '/Users/russell/Downloads/Code-Edit-main/src/generator_pytorch_model.bin'
run_real_model = True # 为了 debug 添加的参数

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
    def encoder_output(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        return outputs[0]
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero=torch.LongTensor(1).fill_(0)     
            for i in range(source_ids.shape[0]):
                context=encoder_output[:,i:i+1]
                context_mask=source_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)                
            return preds   

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda if torch.cuda.is_available() else torch
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
 
def load_model():
    pretrained_model_name = "microsoft/graphcodebert-base"
    base_model_dir = '.'
    # model configuration
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(pretrained_model_name)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

    # build pretrained model
    encoder = model_class.from_pretrained(pretrained_model_name, config=config)
    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)
    pretrained_model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                            beam_size=10, max_length=64,
                            sos_id=tokenizer.cls_token_id, 
                            eos_id=tokenizer.sep_token_id)

    # load the parameter(fined tuned on the training dataset)
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model_path = os.path.join(base_model_dir, model_name)
    pretrained_model.load_state_dict(torch.load(load_model_path, map_location='cpu'))
    finetuned_model = pretrained_model.to(device)
    return finetuned_model, tokenizer, device

def predict(example, model, tokenizer, device):
    max_source_length=256
    source_tokens = tokenizer.tokenize(example)[:max_source_length]
    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    source_mask = [1] * (len(source_tokens))
    padding_length = max_source_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length
    # feed to the model and predict the result
    preds = model(source_ids=torch.tensor([source_ids]).to(device), source_mask=torch.tensor([source_mask]).to(device))
    result = []
    for pred in preds:
        multiple_results = []
        for candidate in pred:
            t = candidate.cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
            multiple_results.append(text)
        result.append(multiple_results)
    return result[0]

def ContentModel(codeWindow, commitMessage, prevEdits):
    # extract the edit line from codeWindow
    editLine = codeWindow.split('<s>')[1][1:-1].rstrip("\n\r")

    replace_dict = {
        'happy': 'sad',
        'good': 'bad',
        'car': 'bike',
        'fly': 'dive'
    }
    for keyword, replacement in replace_dict.items():
        editLine = editLine.replace(keyword, replacement)

    return [editLine, editLine+'(2nd)', editLine+'(3rd)']

def main(input):
    if run_real_model:
        finetuned_model, tokenizer, device = load_model()

    # 提取从 JavaScript 传入的参数
    dict = json.loads(input)
    files = dict["files"]
    targetFilePath = dict["targetFilePath"]
    commitMessage = dict["commitMessage"]
    editType = dict["editType"]
    prevEdits = dict["prevEdits"]
    startPos = dict["startPos"]
    endPos = dict["endPos"]
    
    result = { # 提前记录返回的部分参数
        "targetFilePath": targetFilePath,
        "editType": editType,
        "startPos": startPos,
        "endPos": endPos
    }

    # 获得 targetFile 的文本内容
    for file in files:
        if file[0] == targetFilePath:
            targetFileContent = file[1]
            break

    # 获取文本的行数
    targetFileLines = targetFileContent.splitlines(True) # 保留每行的换行符
    targetFileLineNum = len(targetFileLines)
    
    text = ''
    editLineIdx = []
    # 获取 startPos ~ endPos 所对应的行数，可能是多行
    # 在匹配 endPos 时，需要去掉每行的换行符
    for i in range(targetFileLineNum): # editLineIdx 从 0 计数
        if len(text) == startPos and len(text)+len(targetFileLines[i].rstrip("\n\r")) == endPos: # 如果 startPos ~ endPos 匹配到唯一一行
            editLineIdx.append(i) 
            break
        else: # 否则若 startPos ~ endPos 对应多行
            if len(text) == startPos: # 如果是 editRange 的第一行
                editLineIdx.append(i) 
            elif len(text) > startPos and len(text)+len(targetFileLines[i].rstrip("\n\r")) == endPos: # 如果是 editRange 的最后一行
                editLineIdx.append(i)
                break
            elif len(text) > startPos and len(text)+len(targetFileLines[i].rstrip("\n\r")) < endPos: # 如果是 editRange 的最后一行
                # 如果是 editRange 的中间行
                editLineIdx.append(i)
            elif len(text) > startPos and len(text)+len(targetFileLines[i].rstrip("\n\r")) > endPos:
                # 如果超过了 endPos，说明 editRange 的最后一行在当前行的一半位置
                break
        text += targetFileLines[i]

    # 获取 editRange 的上下文
    startLineIdx = max(0, editLineIdx[0]-contextLength)
    endLineIdx = min(targetFileLineNum, editLineIdx[-1]+contextLength+1)
    
    # 把 editRange 的上下文和 editRange 的内容拼接成 codeWindow
    codeWindow = ''
    if len(editLineIdx) == 1: # 如果 editRange 只有一行
        for lineIdx in range(startLineIdx, endLineIdx):
            if lineIdx == editLineIdx[0]:
                codeWindow += f'<s> {targetFileLines[lineIdx]} <s>'
            else:
                codeWindow += f'{targetFileLines[lineIdx]}'
    elif len(editLineIdx) > 1: # 如果 editRange 有多行
        for lineIdx in range(startLineIdx, endLineIdx):
            if lineIdx == editLineIdx[0]:
                codeWindow += f'<s> {targetFileLines[lineIdx]}'
            elif lineIdx > editLineIdx[0] and lineIdx < editLineIdx[-1]:
                codeWindow += f'{targetFileLines[lineIdx]}'
            elif lineIdx == editLineIdx[-1]:
                codeWindow += f'{targetFileLines[lineIdx]} <s>'
            else:
                codeWindow += f'{targetFileLines[lineIdx]}'
    else:
        raise ValueError('editLineIdx is empty')

    # 将 codeWindow， commitMessage， prevEdits 拼接成 example
    if run_real_model:
        example = codeWindow + ' </s> '  + commitMessage + ' </s>'
        for prevEdit in prevEdits:
            example += ' Delete ' + prevEdit["beforeEdit"].strip() + ' Add ' + prevEdit["afterEdit"].strip() + ' </s>'
        replacements = predict(example, finetuned_model, tokenizer, device)
    else:
        replacements = ContentModel(codeWindow, commitMessage, prevEdits)

    if editType == 'add':
        replacements = [targetFileLines[editLineIdx[0]] + replacement for replacement in replacements]

    result["replacement"] = replacements
    return json.dumps({"data": result})  

# 读取从 Node.js 传递的文本
# 输入 Python 脚本的内容为字典格式: { "files": list, [[filePath, fileContent], ...],
#                                 "targetFilePath": string filePath,
#                                 "commitMessage": string, commit message,
#                                 "editType": str, the type of edit,
#                                 "prevEdits": list, of previous edits, each in format: {"beforeEdit":"", "afterEdit":""},
#                                 "startPos": int, start position,
#                                 "endPos": int, end position}
input = sys.stdin.read()
output = main(input)

# 将修改三元组作为输出发送给 Node.js
# 输出 Python 脚本的内容为字典格式: {"data": 
#                                       { "targetFilePath": string, filePath of target file,
#                                         "editType": str, the type of edit,
#                                         "startPos": int, start position,
#                                         "endPos": int, end position,
#                                         "replacement": list of strings, replacement content   
#                                       }
#                               }
print(output)
sys.stdout.flush()