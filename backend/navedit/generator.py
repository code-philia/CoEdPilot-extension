import torch
import time
from rank_bm25 import BM25Okapi
from transformers import (RobertaTokenizer, T5Config, T5ForConditionalGeneration)
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from .code_window import CodeWindow

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}

CONTEXT_LENGTH = 5
MODEL_ROLE = "generator"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_generator(model_path: str, device: torch.device):
    config_class, model_class, tokenizer_class = MODEL_CLASSES["codet5"]
    
    config = config_class.from_pretrained("salesforce/codet5-base")
    tokenizer = tokenizer_class.from_pretrained("salesforce/codet5-base")
    model = model_class.from_pretrained("salesforce/codet5-base")
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
    model.encoder.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model, tokenizer

def generate_edit(matched_file_path: int,edit_idx: int, raw_location_preds, static_msg, commit_content: dict, commit_msg: str, prev_edits: list[dict], 
                  generator: T5ForConditionalGeneration, generator_tokenizer: RobertaTokenizer, device: torch.device):
    """
    Func:
        Given location_results, commit_content, commit_msg, prev_edits
        predict code after edit using Generator
    Args:
        matched_file_path: the file path that contain the matched edit or selected edit by Locator
        edit_idx: int, index of the matched/selected edit within the file
        commit_content: dict,commit content
        commit_msg: str,commit msg
        prev_edits: list,previous edits
        generator: generator model
        generator_tokenizer: generator tokenizer
        device: device defined in main.py
    Return:
        edit: list of strings, [candidate1, candidate2, ..., candidate10]
        record: dict, {"prepare_time": float, "predict_time": float}
    """
    def retrieve_elements(data_list, index_list):
        return [data_list[i] for i in index_list if i < len(data_list)]
    global code_window_prior_edits
    global input

    record = {
        "prepare_time": 0,
        "predict_time": 0
    }
    current_edit = commit_content[matched_file_path]['edits'][edit_idx].copy()

    targetFileLines,targetFileContent = snapshot_to_file(commit_content[matched_file_path]['snapshot'])
    targetFileLineNum = len(targetFileLines)

    # get context of editRange
    start_line = 0
    end_line = 0
    current_line = 0
    editLineIdx = []
    for hunk in commit_content[matched_file_path]['snapshot']:
        if type(hunk) == list:
            current_line += len(hunk)

        elif type(hunk) == dict:
            if hunk['id'] == current_edit['id'] and (hunk['type'] == 'replace' or hunk['type'] == 'delete'):
                editLineIdx = [x for x in range(current_line,current_line+len(hunk['before']))]
                start_line = max(0,editLineIdx[0]-CONTEXT_LENGTH)
                end_line = min(targetFileLineNum-1,editLineIdx[-1]+CONTEXT_LENGTH)
                break
            elif hunk['id'] == current_edit['id'] and hunk['type'] == 'insert':
                editLineIdx = [max(0, current_line-1)]
                start_line = max(0,current_line-CONTEXT_LENGTH)
                end_line = min(targetFileLineNum-1,current_line+CONTEXT_LENGTH-1)
                break
            else:
                if hunk["state"] == 1:
                    current_line+=len(hunk['after'])
                else:
                    current_line+=len(hunk['before'])
    
    # prepare model input (string format)
    whole_file_inter_golds = raw_location_preds[matched_file_path]["inter_labels"]
    whole_file_inline_golds = raw_location_preds[matched_file_path]["inline_labels"]
    
    inline_labels = whole_file_inline_golds[start_line:end_line+1]
    inter_labels = whole_file_inter_golds[start_line:end_line+2]
    
    # print(f"inline_labels: {inline_labels}")
    # print(f"inter_labels: {inter_labels}")
    code_window_lst = targetFileLines[start_line:end_line+1]
    
    current_edit["code_window"] = code_window_lst
    current_edit["edit_start_line_idx"] = editLineIdx[0]
    current_edit["inline_labels"] = [label[1:-1] for label in inline_labels] # remove < and >
    current_edit["inter_labels"] = [label[1:-1] for label in inter_labels] # remove < and >
    # rename key after to after_edit
    current_edit["after_edit"] = current_edit.pop("after")
    
    """
    current_edit = {
        "id": int, id,
        "file_path": str, file_path,
        "type": str, add/replace,
        "before": list[str], lines of code before edit,
        "after": list[str], lines of code after edit,
        "state": int, 1/0, if this edit has been executed,
        "code_window": list[str], lines of code in the window, including context,
        "edit_start_line_idx": int, the start line index of the edit in the window,
        "inline_labels": list[str], labels for each line in the window, including context
        "inter_labels": list[str], labels for each line in the window, including context
    """
    start = time.time()
    selected_prev_edits = select_hunk(current_edit, prev_edits, generator_tokenizer)
    code_window_prior_edits = selected_prev_edits.copy()

    all_source_ids = formalize_generator_input(current_edit, commit_msg, static_msg, selected_prev_edits, generator_tokenizer)
    sampler = SequentialSampler(all_source_ids)
    eval_dataloader = DataLoader(all_source_ids,sampler=sampler, batch_size=1)

    # run model
    start = time.time()
    generator.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)

        source_ids = batch[0]
        source_mask = source_ids.ne(generator_tokenizer.pad_token_id)
        with torch.no_grad():
            preds = generator.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    num_beams=10,
                                    max_length=512,
                                    num_return_sequences=10)
            preds = preds.reshape(source_ids.size(0), 10, -1)
            preds = preds.cpu().numpy()
            for idx in range(preds.shape[0]):
                replacements = []
                for candidate in preds[idx]:
                    replacements.append(generator_tokenizer.decode(candidate, skip_special_tokens=True,clean_up_tokenization_spaces=False))

    return replacements

def snapshot_to_file(snapshot):
    """
    Fucn:
        replace the current edit with edit['before'] to get the current version or current version
    Return:
        file_lines: the file content after edit (list of lines)
        file_content: the file content after edit (as a whole string)
    """
    file_lines = []
    for window in snapshot:
        if type(window) == dict: # edit
            file_lines.extend(window['before'] if window['state']==0 else window['after'])
        else:
            file_lines.extend(window)
    return file_lines, ''.join(file_lines)


def select_hunk(tgt_hunk: dict, prev_eidt_hunks: list[dict], tokenizer: RobertaTokenizer) -> 'list[dict]':
    """
    Func: 
        Given a target hunk and a list of other hunks, select the prior edits from the other hunks
    Args:
        tgt_hunk: dict, the target hunk
        other_hunks: list[dict], the other hunks
    Return:
        prior_edits: list[dict], the prior edits
    """
    non_overlap_hunks = [CodeWindow(edit, "hunk") for edit in prev_eidt_hunks]
    choosen_hunk_ids = [hunk.id for hunk in non_overlap_hunks] # index to hunk id
    tokenized_corpus = [tokenizer.tokenize("".join(hunk.before_edit_region()+hunk.after_edit_region())) for hunk in non_overlap_hunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer.tokenize("".join(tgt_hunk["code_window"]))
    retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3) 
    retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
    prior_edit_id = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk id
    prior_edits = []
    for id in prior_edit_id: # preserve the order
        prior_edits.append([hunk for hunk in prev_eidt_hunks if hunk["id"] == id][0])
    
    return prior_edits

def formalize_generator_input(sliding_window: dict, prompt: str, static_msg: str,
                            prior_edits: 'list[dict]', tokenizer) -> 'tuple[str, str]':
    sliding_window = CodeWindow(sliding_window, "hunk")
    source_seq = f"<feedback>{static_msg}</feedback>"
    source_seq += sliding_window.formalize_as_generator_target_window(beautify=False, label_num=6)
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    source_seq += f"</code_window><prompt>{truncated_prompt}</prompt><prior_edits>"
    common_seq_len = len(tokenizer.encode(source_seq, add_special_tokens=False))
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit = CodeWindow(prior_edit, "hunk")
        prior_edit_seq = prior_edit.formalize_as_prior_edit(beautify=False, label_num=6)
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        source_seq += prior_edit_seq
        common_seq_len += prior_edit_seq_len
        if common_seq_len + prior_edit_seq_len > 512 - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    source_seq += "</prior_edits>"
    target_seq = "".join(sliding_window.after_edit)
    
    encoded_source_seq = tokenizer(source_seq, padding="max_length", truncation=True, max_length=512)
    source_ids = torch.tensor([encoded_source_seq["input_ids"]], dtype=torch.long)
    data = TensorDataset(source_ids)
    return data
    

