import torch
import torch.nn as nn
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

class Locator(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder. e.g. roberta
        * `config`- configuration of encoder model. 
        * `mask_id`- the id of mask token. e.g. 50264
    """
    def __init__(self, encoder, config, 
                 inline_mask_id=None, inter_mask_id=None, 
                 keep_token_id=None, delete_token_id=None, replace_token_id=None, 
                 null_token_id=None, insert_token_id=None, block_split_token_id=None):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.model_type = "codet5"
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.inline_mask_id=inline_mask_id
        self.inter_mask_id=inter_mask_id
        self.keep_token_id=keep_token_id
        self.delete_token_id=delete_token_id
        self.replace_token_id=replace_token_id
        self.null_token_id=null_token_id
        self.insert_token_id=insert_token_id
        self.block_split_token_id=block_split_token_id
        self.label_weight = torch.ones(config.vocab_size) * 1e-3
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=self.label_weight)
        
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
        if self.model_type == "codet5":
            # T5 encoder has different embedding module
            self._tie_or_clone_weights(self.lm_head,
                                    self.encoder.embed_tokens)
        else:
            self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)  
                                   
    def forward(self, source_ids=None, source_mask=None, target_ids=None, train=True):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        hidden_states = torch.tanh(self.dense(encoder_output)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states).contiguous()
        if train:
            # Flatten the tokens
            active_loss = ((source_ids == self.inter_mask_id) | (source_ids == self.inline_mask_id)).contiguous().view(-1) # find which tokens are masked
            labels = target_ids.contiguous().view(-1)[active_loss] # get the labels of the masked tokens
            filtered_logits = lm_logits.contiguous().view(-1, self.config.vocab_size)[active_loss] # get the logits of the masked tokens

            loss = self.criterion(filtered_logits, labels)
            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            return lm_logits
   

def load_model_locator(model_path,device):
    config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
    locator_config = config_class.from_pretrained('salesforce/codet5-large')
    locator_tokenizer = tokenizer_class.from_pretrained('salesforce/codet5-large')
    encoder = model_class.from_pretrained('salesforce/codet5-large').encoder

    # add special tokens
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "</insert>","<replace-by>", "</replace-by>"]
    locator_tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    encoder.resize_token_embeddings(len(locator_tokenizer))
    locator_config.vocab_size = len(locator_tokenizer)
    
    locator=Locator(encoder=encoder,config=locator_config,
                    inline_mask_id=locator_tokenizer.mask_token_id,
                    inter_mask_id=locator_tokenizer.convert_tokens_to_ids("<inter-mask>"),
                    keep_token_id=locator_tokenizer.convert_tokens_to_ids("<keep>"),
                    delete_token_id=locator_tokenizer.convert_tokens_to_ids("<delete>"),
                    replace_token_id=locator_tokenizer.convert_tokens_to_ids("<replace>"),
                    null_token_id=locator_tokenizer.convert_tokens_to_ids("<null>"),
                    insert_token_id=locator_tokenizer.convert_tokens_to_ids("<insert>"),
                    block_split_token_id=locator_tokenizer.convert_tokens_to_ids("<block-split>"))
    locator.load_state_dict(torch.load(model_path, map_location = device), strict = False)
    locator.to(device)
    return locator, locator_tokenizer
