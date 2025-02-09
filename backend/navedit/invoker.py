import torch
import numpy as np
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from .rich_semantic import finer_grain_window

class Invoker(nn.Module):
    """
        Parameters:

        * `encoder`- encoder. e.g. roberta
        * `config`- configuration of encoder model
    """
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, 3, bias=True)
        
        self.criterion = nn.BCEWithLogitsLoss()
                                   
    def forward(self, source_ids=None, source_mask=None, labels=None, train=True):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        hidden_states = torch.tanh(self.dense(encoder_output)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states).contiguous()
        cls_logits = lm_logits[:,0,:]
        if train:
            loss = self.criterion(cls_logits, labels)
            return loss
        else:
            return cls_logits
         
def load_model_invoker(invoker_file, device):
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
    config = config_class.from_pretrained("microsoft/codebert-base")
    tokenizer = tokenizer_class.from_pretrained("microsoft/codebert-base")
    
    # build expert model
    encoder = model_class.from_pretrained("microsoft/codebert-base")
    # add special tokens
    new_special_tokens = ["<lastest_edit>", "</lastest_edit>",
                          "<prefix>", "</prefix>",
                          "<before>", "</before>",
                          "<after>", "</after>",
                          "<suffix>", "</suffix>",
                          "<previous_edit>", "</previous_edit>",
                          "<rename>", "<def&ref>", "<clone>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    encoder.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    invoker = Invoker(encoder, config)
    
    invoker.load_state_dict(torch.load(invoker_file))
    invoker.to(device)
    return invoker, tokenizer

def ask_invoker(prior_edit_hunks, invoker, invoker_tokenizer, prior_edit_type, device, lang, logger):
    """
    Func:
        Given a list of prior edit hunks, ask the expert to decide which LSP service to use
    Return:
        str: the service name
    """
    prior_edit_hunk_set = prior_edit_hunks[-min(3, len(prior_edit_hunks)):]
    prior_edit_hunk_set.reverse()
    
    code_blocks = finer_grain_window(prior_edit_hunk_set[0]['before_edit'], prior_edit_hunk_set[0]['after_edit'], lang)
    input_seqs = []
    
    common_seq = ""
    for previous_edit in prior_edit_hunk_set[1:]:
        common_seq += "<previous_edit>"
        common_seq += f"<before>{''.join(previous_edit['before_edit'])}</before>"
        common_seq += f"<after>{''.join(previous_edit['after_edit'])}</after>"
        common_seq += "</previous_edit>"
    if prior_edit_type != "clone":
        for block in code_blocks:
            if block["before"] == [] or block["after"] == []:
                continue
            input_seq = f"<{prior_edit_type}><latest_edit>"
            input_seq += f"<before>{''.join(block['before'])}</before>"
            input_seq += f"<after>{''.join(block['after'])}</after>"
            input_seq += "</latest_edit>"
            input_seq += common_seq
            input_seqs.append(input_seq)
    else:
        input_seq = f"<{prior_edit_type}><latest_edit>"
        input_seq += f"<before>{''.join(prior_edit_hunk_set[0]['before_edit'])}</before>"
        input_seq += f"<after>{''.join(prior_edit_hunk_set[0]['after_edit'])}</after>"
        input_seq += "</latest_edit>"
        input_seq += common_seq
        input_seqs.append(input_seq)
    
    if input_seqs == []:
        return "normal", None
    input = invoker_tokenizer(input_seqs, padding="max_length", truncation=True, max_length=512)
    source_ids = torch.tensor(input["input_ids"]).to(device)
    source_masks = torch.tensor(input["attention_mask"]).to(device)

    threshold = np.array([0.754, 0.90, 0.89])
    with torch.no_grad():
        logits = invoker(source_ids=source_ids,source_mask=source_masks,labels=None, train=False)
        probability = torch.sigmoid(logits).detach().cpu().numpy()
        logger.info(f"Probability: {probability}")
        binary_predictions = (probability >= threshold).astype(int)
    
    results = []
    for prediction in binary_predictions:
        if prediction[0] == 1:
            results.append("rename")
        elif prediction[1] == 1:
            results.append("def&ref")
        elif prediction[2] == 1:
            results.append("clone")

    if len(set(results)) == 0:
        return "normal", None
    elif len(set(results)) == 1:
        service = list(results)[0]
        if service == "rename":
            return "rename", max(probability[:,0])
        elif service == "def&ref":
            return "def&ref", max(probability[:,1])
        elif service == "clone":
            return "clone", max(probability[:,2])
    else:
        raise ValueError(f"Invalid results: {results}")
