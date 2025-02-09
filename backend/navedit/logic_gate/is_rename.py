import os
from tree_sitter import Language, Parser
from typing import Literal, TypedDict, List, Dict

def parse(code: str, language: str):
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

def parse_identifier(code: bytes, lang):
    def traverse_tree(node, results, lang, level=0):
        if len(node.children) == 0 and "identifier" in node.type:
            results.append({
                "name": node.text.decode("utf-8"),
                "type": node.type,
                "start": node.start_point,
                "end": node.end_point,
                "level": level #node.parent.text.decode("utf-8")
            })
        for child in node.children:
            traverse_tree(child, results, lang, level+1)
            
        return results
    
    LANGUAGE = Language("tree-sitter/build/my-languages.so", lang)

    parser = Parser()
    parser.set_language(LANGUAGE)
    
    tree = parser.parse(code)
    root_node = tree.root_node

    return traverse_tree(root_node, [], lang)

def get_symbols(node, code_window):
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
            symbol_list.extend(get_symbols(child, code_window))
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

def rest_is_same(code_before, deleted_identifiers, code_after, added_identifiers):
    def remove_ranges_from_document(document, locations):
        """
        Removes multiple ranges of characters from the document based on the given locations.

        :param document: A string representing the entire document.
        :param locations: A list of dictionaries, each containing 'start', 'end', and other information about the range to be removed.
        :return: The modified document with the specified ranges removed.
        """
        lines = document.splitlines()  # Split document into lines

        # Sort locations by start position in reverse order to avoid affecting indices while modifying
        locations = sorted(locations, key=lambda loc: (loc['start'][0], loc['start'][1]), reverse=True)

        for loc in locations:
            start_line, start_col = loc['start']
            end_line, end_col = loc['end']

            # If the start and end are on the same line
            if start_line == end_line:
                lines[start_line] = lines[start_line][:start_col] + lines[start_line][end_col:]
            else:
                # Handle the first line (partially)
                lines[start_line] = lines[start_line][:start_col]
                # Handle the last line (partially)
                lines[end_line] = lines[end_line][end_col:]
                # Remove the lines in between
                del lines[start_line + 1:end_line]

                # Concatenate the start line with the end line
                lines[start_line] += lines[end_line]
                # Finally, remove the last line since it has been merged with the start line
                del lines[end_line]

        # Join the lines back into a single string
        modified_document = "\n".join(lines)
        return modified_document

    remove_identifier_before = remove_ranges_from_document(code_before, deleted_identifiers)
    remove_identifier_after = remove_ranges_from_document(code_after, added_identifiers)
    if remove_identifier_before == remove_identifier_after:
        return True
    return False
    
def rename_edit_assertions(identifier_before: list, identifier_after: list, code_before: str, code_after: str):
    # Assertion 1: has the same number of identifiers
    old_identifier_names = [(identifier["name"],identifier["level"]) for identifier in identifier_before]
    new_identifier_names = [(identifier["name"],identifier["level"]) for identifier in identifier_after]
    if len(old_identifier_names) != len(new_identifier_names):
        return None
    set_old_identifier_names = set(old_identifier_names)
    set_new_identifier_names = set(new_identifier_names)
    if len(set_old_identifier_names) != len(set_old_identifier_names):
        return None

    # Assertion 2: set diff should be equal and not empty
    deleted_identifiers = list(set_old_identifier_names.difference(set_new_identifier_names))
    added_identifiers = list(set_new_identifier_names.difference(set_old_identifier_names))

    if len(deleted_identifiers) != len(added_identifiers):
        return None
    if len(deleted_identifiers) == 0:
        return None

    full_info_deleted = []
    for identifier in identifier_before:
        for name, level in deleted_identifiers:
            if identifier["name"] == name and identifier["level"] == level:
                full_info_deleted.append(identifier)
                break
    full_info_added = []
    for identifier in identifier_after:
        for name, level in added_identifiers:
            if identifier["name"] == name and identifier["level"] == level:
                full_info_added.append(identifier)
                break
    # Assertion 3: remove deleted identifiers and added identifiers, the rest of code should be the same
    if not rest_is_same(code_before, full_info_deleted, code_after, full_info_added):
        return None

    """
    Pass all assertions, return dict:
    {
        "deleted_identifiers": deleted_identifiers,
        "added_identifiers": added_identifiers
        "map": {
            "deleted_identifier_name": "added_identifier_name"
        }
    }
    """
    map = {}
    for deleted_identifier, new_identifier in zip(full_info_deleted, full_info_added):
        if deleted_identifier["name"] in map: # avoid the same identifier mapping to multiple identifiers
            try:
                assert map[deleted_identifier["name"]] == new_identifier["name"]
            except:
                return None
        map[deleted_identifier["name"]] = new_identifier["name"]
    return {
        "deleted_identifiers": full_info_deleted,
        "added_identifiers": full_info_added,
        "map": map
    }

class RenameEditResults(TypedDict):
    deleted_identifiers: List[str]
    added_identifiers: List[str]
    map: Dict[str, str]

rename_edit_results: RenameEditResults = {
    "deleted_identifiers": [],
    "added_identifiers": [],
    "map": {}
}

def is_rename_edit(code_before: str, code_after: str, lang: str) -> Literal[False] | RenameEditResults:
    # first match the lines between before & after
    tree_before = parse(code_before, lang)
    tree_after = parse(code_after, lang)
    
    # get the type of each node of code for matching
    symbols_before = get_symbols(tree_before.root_node, code_before)
    symbols_after = get_symbols(tree_after.root_node, code_after)
    
    # match the symbols
    matched_symbols = lcs(symbols_before, symbols_after)
    to_pop_idx = []
    for m_idx, m in enumerate(matched_symbols):
        if m["before_at_line"] >= len(code_before.splitlines(keepends=True)) or \
           m["before_at_line"] < 0 or \
           m["after_at_line"] >= len(code_after.splitlines(keepends=True)) or \
           m["after_at_line"] < 0:
            to_pop_idx.append(m_idx)
    matched_symbols = [matched_symbols[m_idx] for m_idx in range(len(matched_symbols)) if m_idx not in to_pop_idx]
    if len(matched_symbols) == 0:
        return False
    for ms in matched_symbols:
        ms["before_at_line"] = [ms["before_at_line"]]
        ms["after_at_line"] = [ms["after_at_line"]]
        
    merged_positions = merge_matched_position(matched_symbols)
    if merged_positions is None:
        return False
    
    # In each of the mimimum edit unit, check if this matches a rename edit
    code_before_in_line = code_before.splitlines(keepends=True)
    code_after_in_line = code_after.splitlines(keepends=True)
    
    rename_edit_results = {
        "deleted_identifiers": [],
        "added_identifiers": [],
        "map": {}
    }

    for mp in merged_positions[:1]:
        unit_before = "".join([code_before_in_line[line_idx] for line_idx in mp[0]])
        unit_after = "".join([code_after_in_line[line_idx] for line_idx in mp[1]])

        identifier_before = parse_identifier(unit_before.encode("utf-8"), lang)
        identifier_after = parse_identifier(unit_after.encode("utf-8"), lang)
        
        response = rename_edit_assertions(identifier_before, identifier_after, unit_before, unit_after)

        if response is None:
            continue
        
        # check if the map is consistent
        for deleted_identifer_name in response["map"].keys():
            if deleted_identifer_name in rename_edit_results["map"].keys():
                # identifier of same name should map to the same identifier
                assert rename_edit_results["map"][deleted_identifer_name] == response["map"][deleted_identifer_name]
            else:
                # if not exist, add it
                rename_edit_results["map"][deleted_identifer_name] = response["map"][deleted_identifer_name]

        rename_edit_results["deleted_identifiers"].extend(response["deleted_identifiers"])
        rename_edit_results["added_identifiers"].extend(response["added_identifiers"])
    
    if len(rename_edit_results["deleted_identifiers"]) == 0:
        return False
    return rename_edit_results
            
