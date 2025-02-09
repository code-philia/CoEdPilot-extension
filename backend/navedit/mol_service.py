import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer
from rank_bm25 import BM25Okapi

from model_cache import load_model_with_cache
from .logic_gate import logic_gate
from .rich_semantic import finer_grain_window
from .code_window import CodeWindow
from .invoker import ask_invoker, load_model_invoker
from .locator import load_model_locator
from .generator import load_model_generator

logger = logging.getLogger(__name__)

def load_invoker(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    invoker_model, invoker_tokenizer = load_model_invoker(checkpoint_path, device)
    return invoker_model, invoker_tokenizer, device

def load_locator(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator_model, locator_tokenizer = load_model_locator(checkpoint_path, device)
    return locator_model, locator_tokenizer, device

def load_generator(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_model, generator_tokenizer = load_model_generator(checkpoint_path, device)
    return generator_model, generator_tokenizer, device

def predict_navedit_service(data):
    lang = data["language"]
    modified_files = get_file_snapshot_for_edited_files(data["files"], data["prevEdits"])
    prev_edit_hunks = [construct_prev_edit_hunk(file["snapshots"], file["edit"], lang) \
                       for file in modified_files]

    invoker, invoker_tokenizer, device = load_model_with_cache("invoker_model", lang, load_invoker)

    ### STARTING PHASE: judges the type of primitive edit
    # (NOTE logic gate only discriminates the last edit!)
    # (prior_edit_type = "rename" | "def&ref" | "clone" | "normal")
    # (gate_info contains refactor information like rename) 
    if len(prev_edit_hunks) == 0:
        return {
            "type": "normal"
        }

    prior_edit_type, gate_info = logic_gate(prev_edit_hunks, "python")
    
    if prior_edit_type != "normal":
        ### SECOND PHASE: discriminate the type of on-going edits
        service, service_confidence = ask_invoker(prev_edit_hunks, invoker, invoker_tokenizer, prior_edit_type, device, lang, logger)
        if service == prior_edit_type:
            return {
                "type": service,
                "info": gate_info
            }
    return {
        "type": "normal"
    }

def range_to_sliding_windows(diagnostics, file_content):
    """
    Func:
        Convert lsp diagnostics to sliding windows
    Input:
        diagnostics: list of dict:
            {
                "range": {
                    "start": {
                        "line": 21,
                        "character": 0
                    },
                    "end": {
                        "line": 21,
                        "character": 10
                    }
                },
                "file_path": "/fs/absolute/path/to/file",
                "file_content": 
            }
    """
    sliding_windows = []
    for diagnostic in diagnostics:
        sliding_window = {}
        start_line_idx = max(0, diagnostic["range"]["start"]["line"] - 3)
        end_line_idx = min(len(file_content), diagnostic["range"]["end"]["line"] + 5)
        sliding_window["code_window"] = file_content[start_line_idx:end_line_idx]
        sliding_window["file_path"] = diagnostic["file_path"]
        sliding_window["start_line_idx"] = start_line_idx
        sliding_window["file_lines"] = len(file_content)
        sliding_windows.append(sliding_window)
    
    return sliding_windows

def get_file_snapshot_for_edited_files(files: list[tuple[str, str]], edits: list[dict]):
    '''For each file only consider its last edit and construct the snapshot.'''

    modified_files = []

    files_by_path = {p: f for p, f in files}

    edits_by_path = {}
    for edit in edits:
        if edit['path'] not in edits_by_path:
            edits_by_path[edit['path']] = []
        edits_by_path[edit['path']].append(edit)

    edit_id_cnt = 0
    for file_path, edits in edits_by_path.items():
        snapshots = []
        file_lines = files_by_path[file_path].replace('\r\n', '\n').replace('\r', '\n').split('\n')
        curr_line = 0

        last_edit = edits[-1]
        # FIXME do conversion here, not addition
        last_edit["id"] = edit_id_cnt
        last_edit["before"] = last_edit["rmText"]
        last_edit["after"] = last_edit["addText"]
        edit_id_cnt += 1
        
        if last_edit['line'] > curr_line:
            snapshots.append(file_lines[curr_line:last_edit['line']])
        snapshots.append(last_edit)
        curr_line = last_edit['line'] + last_edit['rmLine']
        if len(file_lines) > curr_line:
            snapshots.append(file_lines[curr_line:])

        modified_files.append({
            "path": file_path,
            "snapshots": snapshots,
            "edit": last_edit
        })

    return modified_files

def get_sliding_window_for_files(files: list[tuple[str, str]]):
    max_sliding_size = 8
    sliding_windows = []
    for file_path, file_content in files:
        file_lines = file_content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        for i in range(0, len(file_lines), max_sliding_size):
            sliding_window = {
                "code_window": file_lines[i:i+max_sliding_size],
                "file_path": file_path,
                "start_line_idx": i,
                "file_lines": len(file_lines)
            }
            sliding_windows.append(sliding_window)
    return sliding_windows

def predict_files(data):
    lang = data["language"]
    modified_files = get_file_snapshot_for_edited_files(data["files"], data["prevEdits"])
    prev_edit_hunks = [construct_prev_edit_hunk(file["snapshots"], file["edit"], lang) \
                       for file in modified_files]

    locator, locator_tokenizer, device = load_model_with_cache("locator_model", lang, load_locator)
    all_files_sliding_windows = get_sliding_window_for_files(data["files"])
    result = predict_sliding_windows(prev_edit_hunks, locator, locator_tokenizer, data["commitMsg"], device, all_files_sliding_windows, "normal", "positive")
    return result

def predict_sliding_windows(prev_edit_hunks, locator, locator_tokenizer, commit_msg, device, sliding_windows, service_name, invoker_service_status=None):
    """
    Func:
        Given a list of sliding windows, predict the label of these sliding windows.
        The pad the rest of the file with either <keep> or <null>
    Return:
        None. The result is stored in raw_preds, the time cost is saved in record    
    """
    raw_preds = {}
    locator_dateset_one_file = make_locator_dataset(sliding_windows, prev_edit_hunks,locator_tokenizer,commit_msg)
    locator_dataloader = DataLoader(locator_dateset_one_file, batch_size=20, shuffle=False)
            
    # predict locations
    locator.eval()
    
    all_preds, all_confidences = locator_predict(locator, locator_tokenizer, device, "multiple files", locator_dataloader)

            
    # process the prediction result
    for preds, confidences, sliding_window in zip(all_preds, all_confidences, sliding_windows):
        # if only <keep> and <null> in preds, we ignore this prediction
        if "<replace>" not in preds and "<delete>" not in preds and "<insert>" not in preds and "<block-split>" not in preds:
            continue
        # restore the prediction for the entire file
        inline_preds = ["<keep>"] * sliding_window["start_line_idx"] # code in file before the sliding window
        inter_preds = ["<null>"] * sliding_window["start_line_idx"] # code in file before the sliding window
        inline_confidences = [1] * sliding_window["start_line_idx"] # code in file before the sliding window
        inter_confidences = [1] * sliding_window["start_line_idx"] # code in file before the sliding window
                
        inline_preds_in_window = [preds[i] for i in range(1, len(preds), 2)]
        inter_preds_in_window = [preds[i] for i in range(0, len(preds), 2)]
        inline_confidences_in_window = [confidences[i] for i in range(1, len(confidences), 2)]
        inter_confidences_in_window = [confidences[i] for i in range(0, len(confidences), 2)]
                
        inline_preds.extend(inline_preds_in_window)
        inter_preds.extend(inter_preds_in_window)
        inline_confidences.extend(inline_confidences_in_window)
        inter_confidences.extend(inter_confidences_in_window)
                
        file_lines_after_window = sliding_window["file_lines"] - len(inline_preds)
        inline_preds.extend(["<keep>"] * file_lines_after_window)
        inter_preds.extend(["<null>"] * file_lines_after_window)
        inline_confidences.extend([1] * file_lines_after_window)
        inter_confidences.extend([1] * file_lines_after_window)
                
        assert len(inline_preds) == len(inline_confidences) == sliding_window["file_lines"]
        assert len(inter_preds) == len(inter_confidences) == sliding_window["file_lines"] + 1
                
        file_path = sliding_window["file_path"]
        if sliding_window["file_path"] in raw_preds: # merge the prediction
            exist_inline_preds = raw_preds[file_path]["inline_preds"]
            exist_inter_preds = raw_preds[file_path]["inter_preds"]
            exist_inline_confidences = raw_preds[file_path]["inline_confidences"]
            exist_inter_confidences = raw_preds[file_path]["inter_confidences"]
                    
            for idx, (exist_inline_pred, inline_pred, exist_inline_confidence, inline_confidence) in enumerate(zip(exist_inline_preds, inline_preds, exist_inline_confidences, inline_confidences)):
                if exist_inline_pred != inline_pred and inline_pred == "<keep>": # follow the exist labels
                    inline_preds[idx] = exist_inline_pred
                    inline_confidences[idx] = exist_inline_confidence
                elif exist_inline_pred != inline_pred and inline_pred != "<keep>" and exist_inline_pred != "<keep>": 
                            # both are not <keep>, we need to consider the confidence
                    if inline_confidence < exist_inline_confidence:
                        inline_preds[idx] = exist_inline_pred
                        inline_confidences[idx] = exist_inline_confidence
                    
            for idx, (exist_inter_pred, inter_pred, exist_inter_confidence, inter_confidence) in enumerate(zip(exist_inter_preds, inter_preds, exist_inter_confidences, exist_inter_confidences)):
                if exist_inter_pred != inter_pred and inter_pred == "<null>":
                    inter_preds[idx] = exist_inter_pred
                    inter_confidences[idx] = exist_inter_confidence
                elif exist_inter_pred != inter_pred and inter_pred != "<null>" and exist_inter_pred != "<null>":
                            # both are not <null>, we need to consider the confidence
                    if inter_confidence < exist_inter_confidence:
                        inter_preds[idx] = exist_inter_pred
                        inter_confidences[idx] = exist_inter_confidence
                        
        
        raw_preds[file_path] = {
            "inline_preds": inline_preds,
            "inter_preds": inter_preds,
            "inline_confidences": inline_confidences,
            "inter_confidences": inter_confidences
        }
    for file_path, raw_pred in raw_preds.items():
        inline_service = [None] * len(raw_pred["inline_preds"])
        inter_service = [None] * len(raw_pred["inter_preds"])

        if invoker_service_status is not None:
            for idx in range(len(raw_pred["inline_preds"])):
                if raw_pred["inline_preds"][idx] != "<keep>":
                    inline_service[idx] = {
                        "service": service_name,
                        "status": invoker_service_status
                    }
            for idx in range(len(raw_pred["inter_preds"])):
                if raw_pred["inter_preds"][idx] != "<null>":
                    inter_service[idx] = {
                        "service": service_name,
                        "status": invoker_service_status
                    }
        raw_preds[file_path]["inline_service"] = inline_service
        raw_preds[file_path]["inter_service"] = inter_service
        
    return raw_preds

def locator_predict(locator, locator_tokenizer, device, file_path, locator_dataloader):
    all_preds = []
    all_confidences = []
    for batch in tqdm(locator_dataloader,desc=f"predicting locations on {file_path}",leave=False):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask = batch                  
        with torch.no_grad():
            lm_logits = locator(source_ids=source_ids,source_mask=source_mask, train=False).to(device)
            lm_logits = torch.nn.functional.softmax(lm_logits, dim=-1)
                # extract masked edit operations
            for i in range(lm_logits.shape[0]): # for sample within batch
                output = []
                confidences = []
                for j in range(lm_logits.shape[1]): # for every token
                    if source_ids[i][j] == locator.inline_mask_id or source_ids[i][j] == locator.inter_mask_id: # if is masked
                        pred_label = locator_tokenizer.decode(torch.argmax(lm_logits[i][j]),clean_up_tokenization_spaces=False)
                        if not pred_label.startswith("<") or not pred_label.endswith(">"):
                            pred_label = f"<{pred_label}>"
                        confidence = torch.max(lm_logits[i][j]).item() # Get the confidence value (0-1)
                        if pred_label == "<insert>" and confidence < 0.5: # debug
                            pred_label = "<null>"
                            confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<null>")].item()
                        elif pred_label == "<replace>" and confidence < 0.5: # debug
                            pred_label = "<keep>"
                            confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<keep>")].item()
                        elif pred_label == "<delete>" and confidence < 0.5: # debug
                            pred_label = "<keep>"
                            confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<keep>")].item()
                        elif pred_label == "<block-split>" and confidence < 0.5: #debug
                            pred_label = "<null>"
                            confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<null>")].item()
                        output.append(pred_label)
                        confidences.append(confidence)
                all_preds.append(output)
                all_confidences.append(confidences)
    return all_preds,all_confidences

def construct_prev_edit_hunk(snapshots: list[list|dict], edit: dict, lang: str):
    """
    Func:
        construct prior edit hunk for a given edit
        edit:{
            "id": int,
            "before": list of str, []
            "after": list of str, []
        }
        edit hunk:{
            "id": int,
            "code_window": list of str, code snippet
            "labels": list of str, label of each line in code_window
            "before_edit": str, the code before edit
            "after_edit": str, the code after edit
            "file_path": str, file path of the edit
            "type": str, edit type, add or replace
            "edit_start_line_idx": int, the start line idx of the edit in the file"
        }
    """
    edit_start_line_idx = 0

    # find edit idx in snapshot
    idx = 0
    while idx < len(snapshots):
        snapshot = snapshots[idx]
        if type(snapshot) is dict and snapshot['id'] == edit['id']:
            break
        elif type(snapshot) is dict:
            edit_start_line_idx+= len(snapshot['before'])
        else:
            edit_start_line_idx+= len(snapshot)
        idx = idx + 1
    
    if idx == len(snapshots):
        raise IndexError('Edit not found!')

    # find context
    code_pre = []
    code_suf = []
    if idx > 0:
        snapshot = snapshots[idx-1]
        if type(snapshot) is dict:
            code_pre.extend(snapshot['before'])
        else:
            code_pre.extend(snapshot)
    if idx < len(snapshots)-1:
        snapshot = snapshots[idx+1]
        if type(snapshot) is dict:
            code_suf.extend(snapshot['before'])
        else:
            code_suf.extend(snapshot)
    
    code_pre = code_pre if len(code_pre)<3 else code_pre[-3:]
    code_suf = code_suf if len(code_suf)<3 else code_suf[:3]

    
    if edit["before"] == [] and edit["after"] != []: # insert type
        hunk = {
            "id": edit["id"],
            "type": "insert",
            "code_window": code_pre + edit["before"] + code_suf,
            "inline_labels": ["keep"]*len(code_pre)+ ["keep"]*len(code_suf),
            "inter_labels": ["null"] * len(code_pre) + ["insert"] + ["null"] * len(code_suf),
            "before_edit": edit['before'],
            "after_edit": edit['after'],
            "edit_start_line_idx": max(0,edit_start_line_idx)
        }
        
    elif edit["before"] != [] and edit["after"] == []: # delete type
        hunk = {
            "id": edit["id"],
            "type": "delete",
            "code_window": code_pre + edit["before"] + code_suf,
            "inline_labels": ["keep"]*len(code_pre)+ ["delete"]* len(edit["before"]) + ["keep"]*len(code_suf),
            "inter_labels": ["null"] * (len(code_pre) + len(edit["before"]) + len(code_suf)),
            "before_edit": edit['before'],
            "after_edit": edit['after'],
            "edit_start_line_idx": max(0,edit_start_line_idx)
        }
    
    else:
        code_blocks = finer_grain_window(edit["before"], edit["after"], lang)
        
        inline_labels = ["keep"] * len(code_pre)
        inter_labels = ["null"] * len(code_pre)
        inter = "null"
        for block in code_blocks:
            if block["block_type"] == "insert":
                inter = "insert"
            elif block["block_type"] == "delete":
                if inter == "block-split":
                    inter_labels.append("null")
                else: # only allow insert or null
                    inter_labels.append(inter)
                inter = "null"
                inline_labels += ["delete"] * len(block["before"])
                inter_labels += ["null"] * (len(block["before"]) - 1)
            elif block["block_type"] == "modify":
                inter_labels.append(inter)
                inter = "block-split"
                inline_labels += ["replace"] * len(block["before"])
                inter_labels += ["null"] * (len(block["before"]) - 1)
        if inter == "block-split":
            inter_labels.append("null")
        else:
            inter_labels.append(inter)
            
        inline_labels += ["keep"] * len(code_suf)
        inter_labels += ["null"] * len(code_suf)
        assert len(inline_labels) + 1 == len(inter_labels)
        hunk = {
            "id": edit["id"],
            "type": "replace",
            "code_window": code_pre + code_blocks + code_suf,
            "inline_labels": inline_labels,
            "inter_labels": inter_labels,
            "before_edit": edit['before'],
            "after_edit": edit['after'],
            "edit_start_line_idx": max(0,edit_start_line_idx)
        }
    return hunk

def make_locator_dataset(sliding_windows: list, prev_eidt_hunks: list,
                         locator_tokenizer: RobertaTokenizer, commit_msg: str)-> TensorDataset:
    """
    Func:
        Given a fixed prior edit estimator, select most relevant hunk as prior edit 
        and construct the dataset for locator to infer
    Args:
        
    """
    source_seqs = []
    hunks = [CodeWindow(edit, "hunk") for edit in prev_eidt_hunks]
    for sliding_window in sliding_windows:
        non_overlap_hunks = hunks
        choosen_hunk_ids = [hunk.id for hunk in hunks] # index to hunk id
        tokenized_corpus = [locator_tokenizer.tokenize("".join(hunk.before_edit_region()+hunk.after_edit_region())) for hunk in non_overlap_hunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = locator_tokenizer.tokenize("".join(sliding_window["code_window"]))
        retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3) 
        retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
        prior_edit_id = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk id
        prior_edits = []
        for id in prior_edit_id: # preserve the order
            prior_edits.append([hunk for hunk in hunks if hunk.id == id][0])
            
        source_seq = formalize_locator_input(sliding_window, commit_msg, prior_edits, locator_tokenizer)
        source_seqs.append(source_seq)
        
    encoded_source_seq = locator_tokenizer(source_seqs, padding="max_length", truncation=True, max_length=512)
    
    source_ids = torch.tensor(encoded_source_seq["input_ids"])
    source_mask = torch.tensor(encoded_source_seq["attention_mask"])
    dataset = TensorDataset(source_ids, source_mask)

    return dataset

def formalize_locator_input(sliding_window: dict, prompt: str, 
                            prior_edits: list[dict], tokenizer: RobertaTokenizer) -> tuple[str, str]:
    """
    Func:
        Given a sliding window, prior edits, and prompt, form the input sequence for locator
    Args:
        sliding_window: one sliding window
        prior_edits: the prior edit hunks selected
    """
    source_seq = "<code_window><inter-mask>"
    for line in sliding_window["code_window"]:
        source_seq += f"<mask>{line}<inter-mask>"
    source_seq += f"<prompt>{prompt}</prompt><prior_edits>"
    source_seq_len = len(tokenizer.encode(source_seq, add_special_tokens=False))
    
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit_seq = prior_edit.formalize_as_prior_edit(beautify=False, label_num=6)
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        source_seq += prior_edit_seq
        source_seq_len += prior_edit_seq_len
        if source_seq_len + prior_edit_seq_len > 512 - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    source_seq += "</prior_edits>"
    
    return source_seq


