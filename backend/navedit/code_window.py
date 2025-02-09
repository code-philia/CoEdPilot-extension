import nltk
import difflib

class CodeWindow():
    def __init__(self, info_dict: dict, type: str):
        assert type in ["hunk", "sliding_window"]
        self.window_type = type
        if self.window_type == "hunk":
            self.id = info_dict["id"]
            self.code_window = info_dict["code_window"]
            self.inline_labels = info_dict["inline_labels"]
            self.inter_labels = info_dict["inter_labels"]
            self.after_edit = info_dict["after_edit"]
            self.edit_type = info_dict["type"] # "replace", "insert", "delete"
            # self.file_path = info_dict["file_path"]
            self.edit_start_line_idx = info_dict["edit_start_line_idx"]
        else:
            self.code_window = info_dict["code_window"]
            self.sliding_window_type = info_dict["sliding_window_type"]
            self.inline_labels = info_dict["inline_labels"]
            self.inter_labels = info_dict["inter_labels"]
            self.overlap_hunk_ids = info_dict["overlap_hunk_ids"]
            # self.file_path = info_dict["file_path"]
            self.to_insert = info_dict["to_insert"]
            self.edit_start_line_idx = info_dict["edit_start_line_idx"]
    
    def before_edit_window(self, split_by_line: bool = True):
        """
        What the entire window looks like before edit
        """
        window = []
        if self.window_type == "hunk":
            window = []
            for line in self.code_window:
                if type(line) is str:
                    window.append(line)
                else:
                    window.extend(line["before"])
        else:
            window = self.code_window
        
        if split_by_line:
            return window
        else:
            return "".join(window)
        
    def after_edit_window(self, split_by_line: bool = True):
        """
        What the entire window looks like after edit
        """
        if self.window_type == "sliding_window":
            raise ValueError("This is a sliding window, no after edit window")
        window = []
        if self.edit_type == "replace":
            for line in self.code_window:
                if type(line) is str:
                    window.append(line)
                else:
                    window.extend(line["after"])
        elif self.edit_type == "delete":
            for line, inline_label in zip(self.code_window, self.inline_labels):
                if inline_label != "delete":
                    window.append(line)
        elif self.edit_type == "insert":
            insert_idx = self.inter_labels.index("insert")
            window = self.code_window[:insert_idx]
            window.extend(self.after_edit)
            window.extend(self.code_window[insert_idx:])
        if split_by_line:
            return window
        else:
            return "".join(window)
    
    def before_edit_region(self, split_by_line: bool = True, allow_fuzzy: bool = True):
        """
        What the to edit region looks like before edit
        """
        if self.window_type == "sliding_window":
            raise ValueError("This is a sliding window, no after edit window")
        window = []
        if allow_fuzzy:
            if self.edit_type == "replace":
                for line in self.code_window:
                    if type(line) is not str:
                        window.extend(line["before"])
            elif self.edit_type == "delete":
                for line, inline_label in zip(self.code_window, self.inline_labels):
                    if inline_label == "delete":
                        window.append(line)
            elif self.edit_type == "insert":
                insert_idx = self.inter_labels.index("insert")
                try:
                    window.append(self.code_window[insert_idx-1])
                except:
                    pass
                try:
                    window.append(self.code_window[insert_idx])
                except:
                    pass
            if split_by_line:
                return window
            else:
                return "".join(window)
        else:
            if self.edit_type == "replace":
                for line in self.code_window:
                    if type(line) is not str:
                        window.extend(line["before"])
            elif self.edit_type == "delete":
                for line, inline_label in zip(self.code_window, self.inline_labels):
                    if inline_label == "delete":
                        window.append(line)
            elif self.edit_type == "insert":
                pass
            if split_by_line:
                return window
            else:
                return "".join(window)
        
    def after_edit_region(self, split_by_line: bool = True):
        """
        What the to edit region looks like after edit
        """
        if self.window_type == "sliding_window":
            raise ValueError("This is a sliding window, no after edit window")
        window = []
        if self.edit_type == "replace":
            for line in self.code_window:
                if type(line) is not str:
                    window.extend(line["after"])
        elif self.edit_type == "delete":
            pass
        elif self.edit_type == "insert":
            window = self.after_edit
        
        if split_by_line:
            return window
        else:
            return "".join(window)
        
    def formalize_as_locator_target_window(self, beautify: bool = False, label_num: int = 6):
        if self.window_type == "hunk":
            raise ValueError("Hunk cannot be formalized as locator target window")
        labels = []
        if label_num == 6:
            if beautify:
                target_window_target = f"<code_window>\n\t<{self.inter_labels[0]}>"
                target_window_source = f"<code_window>\n\t<inter-mask>"
            else:
                target_window_target = f"<code_window><{self.inter_labels[0]}>"
                target_window_source = f"<code_window><inter-mask>"
        else:
            target_window_target = f"<code_window>"
            target_window_source = f"<code_window>"
        labels.append(self.inter_labels[0])
        for inline_label, code_line, inter_label in zip(self.inline_labels, self.code_window, self.inter_labels[1:]):
            if label_num == 6:
                if beautify:
                    code_line = code_line.replace("\n", "")
                    target_window_source += f"\n\t\t<mask>{code_line}\n\t<inter-mask>"
                    target_window_target += f"\n\t\t<{inline_label}>{code_line}\n\t<{inter_label}>"
                else:
                    target_window_source += f"<mask>{code_line}<inter-mask>"
                    target_window_target += f"<{inline_label}>{code_line}<{inter_label}>"
            else:
                if beautify:
                    code_line = code_line.replace("\n", "")
                    target_window_source += f"\n\t\t<mask>{code_line}"
                    target_window_target += f"\n\t\t<{inline_label}>{code_line}"
                else:
                    target_window_source += f"<mask>{code_line}"
                    target_window_target += f"<{inline_label}>{code_line}"
            labels.append(inline_label)
            labels.append(inter_label)
        if beautify:
            target_window_source += "\n</code_window>"
            target_window_target += "\n</code_window>"
        else:
            target_window_source += "</code_window>"
            target_window_target += "</code_window>"
        return target_window_source, target_window_target

    def formalize_as_generator_target_window(self, beautify: bool = False, label_num: int = 6):
        if self.window_type == "sliding_window":
            raise ValueError("Sliding window cannot be formalized as generator target window")
        if label_num == 6:
            if beautify:
                target_window = f"<code_window>\n\t<{self.inter_labels[0]}>"
            else:
                target_window = f"<code_window><{self.inter_labels[0]}>"
        else:
            target_window = f"<code_window>"
        before_edit_window = self.before_edit_window()
        assert len(before_edit_window) == len(self.inline_labels) == len(self.inter_labels) - 1
        if label_num == 6:
            for inline_label, code_line, inter_label in zip(self.inline_labels, before_edit_window, self.inter_labels[1:]):
                if beautify:
                    code_line = code_line.replace("\n", "")
                    target_window += f"\n\t\t<{inline_label}>{code_line}\n\t<{inter_label}>"
                else:
                    target_window += f"<{inline_label}>{code_line}<{inter_label}>"
        else:
            from utils import label_conversion
            label = label_conversion(self.inline_labels, self.inter_labels)
            for label, code_line in zip(label, before_edit_window):
                if beautify:
                    code_line = code_line.replace("\n", "")
                    target_window += f"\n\t<{label}>{code_line}"
                else:
                    target_window += f"<{label}>{code_line}"
        if beautify:
            target_window += "\n</code_window>"
        else:
            target_window += "</code_window>"
        return target_window
    
    def word_level_diff(self, before: list[str], after: list[str]):
        delete_start = "<block-delete>"
        delete_end = "</block-delete>"
        insert_start = "<block-insert>"
        insert_end = "</block-insert>"
        before = "".join(before)
        after = "".join(after)
        
        # Split the strings into words
        chars1 = nltk.word_tokenize(before)
        chars2 = nltk.word_tokenize(after)

        # Find the difference between the two strings
        differ = difflib.Differ()
        diff = list(differ.compare(chars1, chars2))

        # 生成差异报告并合并相邻的 ++ 或者 --
        result = []
        current_token = ""
        current_type = "="

        for line in diff:
            if line.startswith("  "):  # 未更改的字符
                if current_type == "=":
                    current_token += line[2:]
                elif current_type == "+":
                    result += f"{insert_start}{current_token}{insert_end}"
                    current_token = line[2:]
                    current_type = "="
                elif current_type == "-":
                    result += f"{delete_start}{current_token}{delete_end}"
                    current_token = line[2:]
                    current_type = "="
            elif line.startswith("- "):  # 仅在第一个文本中的字符
                if current_type == "=":
                    result += f"{current_token}"
                    current_token = line[2:]
                    current_type = "-"
                elif current_type == "+":
                    result += f"{insert_start}{current_token}{insert_end}"
                    current_token = line[2:]
                    current_type = "-"
                elif current_type == "-":
                    current_token += line[2:]
            elif line.startswith("+ "):  # 仅在第二个文本中的字符
                if current_type == "=":
                    result += f"{current_token}"
                    current_token = line[2:]
                    current_type = "+"
                elif current_type == "+":
                    current_token += line[2:]
                elif current_type == "-":
                    result += f"{delete_start}{current_token}{delete_end}"
                    current_token = line[2:]
                    current_type = "+"
                
        # 添加最后一个 token
        if current_token:
            if current_type == "=":
                result.append(current_token)
            elif current_type == "-":
                result.append(f"[-{current_token}-]")
            elif current_type == "+":
                result.append(f"[+{current_token}+]")
        
        return "".join(result)
    
    def formalize_as_prior_edit(self, beautify: bool = False, label_num: int = 6):
        if self.window_type == "sliding_window":
            raise ValueError("Sliding window cannot be formalized as prior edit")
        if label_num == 6:
            if self.edit_type == "insert":
                if self.inter_labels[0] == "insert":
                    if beautify:
                        prior_edit = f"<edit>\n\t<insert>{''.join(self.after_edit)}</insert>"
                    else:
                        prior_edit = f"<edit><insert>{''.join(self.after_edit)}</insert>"
                else:
                    if beautify:
                        prior_edit = f"<edit>\n\t<{self.inter_labels[0]}>"
                    else:
                        prior_edit = f"<edit><{self.inter_labels[0]}>"
                for inline_label, code_line, inter_label in zip(self.inline_labels, self.code_window, self.inter_labels[1:]):
                    if beautify:
                        code_line = code_line.replace("\n", "")
                        prior_edit += f"\n\t\t<{inline_label}>{code_line}\n\t"
                    else:
                        prior_edit += f"<{inline_label}>{code_line}"
                    if inter_label == "insert":
                        prior_edit += f"<insert>{''.join(self.after_edit)}</insert>"
                    else:
                        prior_edit += f"<{inter_label}>"
                prior_edit += "\n</edit>"
            elif self.edit_type == "delete":
                if beautify:
                    prior_edit = f"<edit>\n\t<{self.inter_labels[0]}>"
                else:
                    prior_edit = f"<edit><{self.inter_labels[0]}>"
                for inline_label, code_line, inter_label in zip(self.inline_labels, self.code_window, self.inter_labels[1:]):
                    if beautify:
                        code_line = code_line.replace("\n", "")
                        prior_edit += f"\n\t\t<{inline_label}>{code_line}\n\t<{inter_label}>"
                    else:
                        prior_edit += f"<{inline_label}>{code_line}<{inter_label}>"
                prior_edit += "\n</edit>"
            elif self.edit_type == "replace":
                prior_edit = "<edit>"
                for block_idx, block in enumerate(self.code_window):
                    if block_idx == 0:
                        prev_block_type = "str"
                    else:
                        if type(self.code_window[block_idx - 1]) is str:
                            prev_block_type = "str"
                        else:
                            prev_block_type = self.code_window[block_idx - 1]["block_type"]

                    if prev_block_type == "str" or prev_block_type == "delete":
                        if type(block) is str:
                            block = block.replace("\n", "")
                            if beautify:
                                prior_edit += f"\n\t<null>\n\t\t<keep>{block}"
                            else:
                                prior_edit += f"<null><keep>{block}"
                        elif block["block_type"] == "insert":
                            code = ''.join(block['after']).replace("\n", "")
                            if beautify:
                                # remove last \n
                                prior_edit += f"\n\t<insert>{code}</insert>"
                            else:
                                prior_edit += f"<insert>{code}</insert>"
                        elif block["block_type"] == "delete":
                            for line in block["before"]:
                                line = line.replace("\n", "")
                                if beautify:
                                    prior_edit += f"\n\t<null>\n\t\t<delete>{line}"
                                else:
                                    prior_edit += f"<null><delete>{line}"
                        elif block["block_type"] == "modify":
                            for code in block["before"]:
                                if beautify:
                                    prior_edit += f"\n\t<null>\n\t\t<replace>{code}"
                                else:
                                    prior_edit += f"<null><replace>{code}"
                            if beautify:
                                prior_edit += f"\n\t\t<replace-by>{''.join(block['after'])}\t\t</replace-by>"
                            else:
                                prior_edit += f"<replace-by>{''.join(block['after'])}</replace-by>"
                    elif prev_block_type == "insert":
                        if type(block) is str:
                            block = block.replace("\n", "")
                            if beautify:
                                prior_edit += f"\n\t\t<keep>{block}"
                            else:
                                prior_edit += f"<keep>{block}"
                        elif block["block_type"] == "insert":
                            raise ValueError("Cannot have two consecutive insert blocks")
                        elif block["block_type"] == "delete":
                            for line in block["before"]:
                                line = line.replace("\n", "")
                                if beautify:
                                    prior_edit += f"\n\t\t<delete>{line}"
                                else:
                                    prior_edit += f"<delete>{line}"
                        elif block["block_type"] == "modify":
                            for code_idx, code in enumerate(block["before"]):
                                if code_idx == 0:
                                    if beautify:
                                        prior_edit += f"\n\t\t<replace>{code}"
                                    else:
                                        prior_edit += f"<replace>{code}"
                                else:
                                    if beautify:
                                        prior_edit += f"\n\t<null>\n\t\t<replace>{code}"
                                    else:
                                        prior_edit += f"<null><replace>{code}"
                            if beautify:
                                prior_edit += f"\n\t\t<replace-by>{''.join(block['after'])}\t\t</replace-by>"
                            else:
                                prior_edit += f"<replace-by>{''.join(block['after'])}</replace-by>"
                    elif prev_block_type == "modify":
                        if type(block) is str:
                            block = block.replace("\n", "")
                            if beautify:
                                prior_edit += f"\n\t<null>\n\t\t<keep>{block}"
                            else:
                                prior_edit += f"<null><keep>{block}"
                        elif block["block_type"] == "insert":
                            code = ''.join(block['after']).replace("\n", "")
                            if beautify:
                                prior_edit += f"\n\t<insert>{code}</insert>"
                            else:
                                prior_edit += f"<insert>{code}</insert>"
                        elif block["block_type"] == "delete":
                            for line in block["before"]:
                                line = line.replace("\n", "")
                                if beautify:
                                    prior_edit += f"\n\t<null>\n\t\t<delete>{line}"
                                else:
                                    prior_edit += f"<null><delete>{line}"
                        elif block["block_type"] == "modify":
                            for code_idx, code in enumerate(block["before"]):
                                if code_idx == 0:
                                    if beautify:
                                        prior_edit += f"\n\t<block-split>\n\t<replace>{code}"
                                    else:
                                        prior_edit += f"<block-split><replace>{code}"
                                else:
                                    if beautify:
                                        prior_edit += f"\n\t<null>\n\t\t<replace>{code}"
                                    else:
                                        prior_edit += f"\n\t<null><replace>{code}"
                            if beautify:
                                prior_edit += f"\n\t\t<replace-by>{''.join(block['after'])}\t\t</replace-by>"
                            else:
                                prior_edit += f"<replace-by>{''.join(block['after'])}</replace-by>"
                if beautify:
                    prior_edit += f"\n\t<{self.inter_labels[-1]}>\n</edit>"
                else:
                    prior_edit += f"<{self.inter_labels[-1]}></edit>"
                return prior_edit

        elif label_num == 3:
            from utils import label_conversion
            labels = label_conversion(self.inline_labels, self.inter_labels)
            code_window = []
            for line in self.code_window:
                if type(line) is str:
                    code_window.append(line)
                else:
                    code_window.extend(line["before"])
                    
            assert len(code_window) == len(labels)
            prior_edit = "<edit>"
            first_encounter = True
            for line, label in zip(code_window, labels):
                if label == "keep":
                    if beautify:
                        prior_edit += f"\n<keep>{line}"
                    else:
                        prior_edit += f"<keep>{line}"
                if label == "insert":
                    if beautify:
                        prior_edit += f"\n<insert>{line}{''.join(self.after_edit)}</insert>"
                    else:
                        prior_edit += f"<insert>{line}{''.join(self.after_edit)}</insert>"
                elif label == "replace" and first_encounter:
                    if beautify:
                        prior_edit += f"\n<replace-by>{''.join(self.after_edit)}</replace-by>\n<replace>{line}"
                    else:
                        prior_edit += f"<replace-by>{''.join(self.after_edit)}</replace-by><replace>{line}"
                    first_encounter = False
                elif label == "replace" and not first_encounter:
                    if beautify:
                        prior_edit += f"\n<replace>{line}"
                    else:
                        prior_edit += f"<replace>{line}"
            prior_edit += "\n</edit>" if beautify else "</edit>"
        
        return prior_edit
    