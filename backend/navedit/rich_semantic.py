import os
from tree_sitter import Language, Parser

# FIXME use a better way to get correct path
os.chdir(os.path.dirname(__file__))

def parse(code, language):
    assert language in ["go", "javascript", "typescript", "python", "java"]
    if not os.path.exists("tree-sitter/build/my-languages.so"):
        Language.build_library(
            # Store the library in the `build` directory
            "tree-sitter/build/my-languages.so",

            # Include one or more languages
            [
                "tree-sitter/tree-sitter-go",
                "tree-sitter/tree-sitter-javascript",
                "tree-sitter/tree-sitter-typescript/typescript",
                "tree-sitter/tree-sitter-python",
                "tree-sitter/tree-sitter-java",
            ]
        )
    parser = Parser()
    parser.set_language(Language("tree-sitter/build/my-languages.so", language))
    tree = parser.parse(bytes(code, "utf8"))
    return tree

def merge_matched_position(common_positions):
    """
    Func:
        Given the matched replace blocks, merge the overlapped blocks
    Args:
        common_positions: list, the list of matched replace blocks
    Return:
        merged_positions: list, the list of merged replace blocks
    """
    def is_consecutive(numbers):
        if len(numbers) < 2:
            return True  # 0或1个元素的列表被视为连贯的

        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i - 1] + 1:
                return False
        return True
    
    positions = [(position["before_at_line"], position["after_at_line"]) for position in common_positions]
    
    merged_positions = [positions[0]]
    for position in positions[1:]:
        to_merge_position_group_idx = []
        for mp_idx, mp in enumerate(merged_positions):
            if len(set(position[0]).intersection(set(mp[0]))) != 0 or len(set(position[1]).intersection(set(mp[1]))) != 0:
                to_merge_position_group_idx.append(mp_idx)
        if to_merge_position_group_idx == []:
            merged_positions.append((position[0], position[1]))
            continue
        to_merge_position = [merged_positions[idx] for idx in to_merge_position_group_idx] + [(position[0], position[1])]
        merged_old_position = list(set([line for lines in to_merge_position for line in lines[0]]))
        sorted_old_position = sorted(merged_old_position)
        merged_new_position = list(set([line for lines in to_merge_position for line in lines[1]]))
        sorted_new_position = sorted(merged_new_position)
        # if idx in sorted_old_position & sorted_new_position is continuous, then merge them
        if is_consecutive(sorted_old_position) and is_consecutive(sorted_new_position):
            merged_positions = [mp for idx, mp in enumerate(merged_positions) if idx not in to_merge_position_group_idx]
            merged_positions.append((sorted_old_position, sorted_new_position))
        else:
            return None # if the merged positions are not continuous, we believe the quality of the matched positions is not good, so we return None

    return merged_positions

def finer_grain_window(before: list[str], after: list[str], lang: str) -> list:
    def return_delete_insert_blocks(before,after):
        blocks = [
            {
                "block_type": "delete",
                "before": before,
                "after": []
            }
        ]
        if "".join(after).strip() != "":
            blocks.append({
                "block_type": "insert",
                "before": [],
                "after": after
            })
        return blocks
    
    new_window = []
    before_str = "".join(before)
    after_str = "".join(after)
    before_tree = parse(before_str, lang)
    after_tree = parse(after_str, lang)
    
    before_symbols = get_symbol_info(before_tree.root_node, before)
    after_symbols = get_symbol_info(after_tree.root_node, after)
    matched_symbols = lcs(before_symbols, after_symbols)
    to_pop_idx = []
    for m_idx, m in enumerate(matched_symbols):
        if m["before_at_line"] >= len(before) or \
           m["before_at_line"] < 0 or \
           m["after_at_line"] >= len(after) or \
           m["after_at_line"] < 0:
            to_pop_idx.append(m_idx)
    matched_symbols = [matched_symbols[m_idx] for m_idx in range(len(matched_symbols)) if m_idx not in to_pop_idx]
    
    if matched_symbols == []:
        return return_delete_insert_blocks(before, after)
    for ms in matched_symbols:
        ms["before_at_line"] = [ms["before_at_line"]]
        ms["after_at_line"] = [ms["after_at_line"]]
        
    merged_positions = merge_matched_position(matched_symbols)
    if merged_positions is None:
        return return_delete_insert_blocks(before, after)
    filtered_merged_positions = [merged_positions[0]]
    for match_pos_idx, match_pos in enumerate(merged_positions[1:]):
        match_pos_idx += 1
        if match_pos[0][0] > filtered_merged_positions[-1][0][-1] and \
        match_pos[1][0] > filtered_merged_positions[-1][1][-1]:
            filtered_merged_positions.append(match_pos)
    
    if len(filtered_merged_positions) == 0:
        return return_delete_insert_blocks(before, after)
    skip_emtpy_insertion = 0
    for match_pos_idx, match_pos in enumerate(filtered_merged_positions):
        if match_pos_idx == 0:
            prev_old_end_line_idx = -1
            prev_new_end_line_idx = -1
        else:
            prev_old_end_line_idx = filtered_merged_positions[match_pos_idx-1][0][-1]
            prev_new_end_line_idx = filtered_merged_positions[match_pos_idx-1][1][-1]
        # take care of unmatched positions before this matched position
        if prev_old_end_line_idx + 1 < match_pos[0][0] and prev_new_end_line_idx + 1 < match_pos[1][0]:
            new_window.append({
                "block_type": "delete",
                "before": before[prev_old_end_line_idx+1:match_pos[0][0]],
                "after": []
            })
            if "".join(after[prev_new_end_line_idx+1:match_pos[1][0]]).strip() != "":
                new_window.append({
                    "block_type": "insert",
                    "before": [],
                    "after": after[prev_new_end_line_idx+1:match_pos[1][0]]
                })
            else:
                skip_emtpy_insertion += 1
        elif prev_old_end_line_idx + 1 < match_pos[0][0]:
            new_window.append({
                "block_type": "delete",
                "before": before[prev_old_end_line_idx+1:match_pos[0][0]],
                "after": []
            })
        elif prev_new_end_line_idx + 1 < match_pos[1][0]:
            if "".join(after[prev_new_end_line_idx+1:match_pos[1][0]]).strip() != "":
                new_window.append({
                    "block_type": "insert",
                    "before": [],
                    "after": after[prev_new_end_line_idx+1:match_pos[1][0]]
                })
            else:
                skip_emtpy_insertion += 1
        # take care of matched positions
        new_window.append({
            "block_type": "modify",
            "before": before[match_pos[0][0]:match_pos[0][-1]+1],
            "after": after[match_pos[1][0]:match_pos[1][-1]+1]
        })
        # take care of unmatched positions after last matched position
        if match_pos_idx == len(filtered_merged_positions) - 1:   
            if match_pos[0][-1] != len(before) - 1 and match_pos[1][-1] != len(after) - 1:
                new_window.append({
                    "block_type": "delete",
                    "before": before[match_pos[0][-1]+1:],
                    "after": []
                })
                if "".join(after[match_pos[1][-1]+1:]).strip() != "":
                    new_window.append({
                        "block_type": "insert",
                        "before": [],
                        "after": after[match_pos[1][-1]+1:]
                    })
                else:
                    skip_emtpy_insertion += 1
            elif match_pos[0][-1] != len(before) - 1:
                new_window.append({
                    "block_type": "delete",
                    "before": before[match_pos[0][-1]+1:],
                    "after": []
                })
            elif match_pos[1][-1] != len(after) - 1:
                if "".join(after[match_pos[1][-1]+1:]).strip() != "":
                    new_window.append({
                        "block_type": "insert",
                        "before": [],
                        "after": after[match_pos[1][-1]+1:]
                    })
                else:
                    skip_emtpy_insertion += 1

    totoal_block_before = 0
    totoal_block_after = 0
    for block in new_window:
        totoal_block_before += len(block["before"])
        totoal_block_after += len(block["after"])
        assert not (block["before"] == [] and block["after"] == [])
    try:
        assert totoal_block_before == len(before)
        assert totoal_block_after + skip_emtpy_insertion == len(after)
    except:
        raise AssertionError
    
    return new_window

def get_symbol_info(node, code_window):
    symbol_list = []
    if len(node.children) == 0:
        if node.type not in [",", ":", ".", ";", "(", ")", "[", "]", "{", "}"]:
            return [
                {
                    "text": node.text.decode("utf-8"),
                    "type": node.type,
                    "at_line": node.start_point[0],
                    "at_column": node.start_point[1]
                }
            ]
        elif code_window[node.start_point[0]].strip() == node.type:
            # when it is trivial, but takes the whole line, we should consider it
            return [
                {
                    "text": node.text.decode("utf-8"),
                    "type": node.type,
                    "at_line": node.start_point[0],
                    "at_column": node.start_point[1]
                }
            ]
        else:
            return []
    else:
        for child in node.children:
            symbol_list.extend(get_symbol_info(child, code_window))
    return symbol_list

def lcs(list1, list2):
    # 获取列表的长度
    m, n = len(list1), len(list2)
    
    # 创建一个 (m+1) x (n+1) 的二维数组，初始化为 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 动态规划计算 LCS
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1]['text'] == list2[j - 1]['text'] and list1[i - 1]['type'] == list2[j - 1]['type']:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # 逆向回溯找到 LCS 并合并元素
    merged_list = []
    i, j = m, n
    while i > 0 and j > 0:
        if list1[i - 1]['text'] == list2[j - 1]['text'] and list1[i - 1]['type'] == list2[j - 1]['type']:
            merged_element = {
                'text': list1[i - 1]['text'],
                'type': list1[i - 1]['type'],
                'before_at_line': list1[i - 1].get('at_line'),
                'before_at_column': list1[i - 1].get('at_column'),
                'after_at_line': list2[j - 1].get('at_line'),
                'after_at_column': list2[j - 1].get('at_column')
            }
            merged_list.append(merged_element)
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    # 逆序输出 LCS
    merged_list.reverse()
    return merged_list
