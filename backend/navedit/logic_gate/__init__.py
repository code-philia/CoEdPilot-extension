from .is_rename import is_rename_edit
from .is_defref import is_defref_edit
from .is_clone import is_clone_edit
from ..code_window import CodeWindow
from tree_sitter import Language, Parser

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
    
    LANGUAGE = Language('build/my-languages.so', lang)

    parser = Parser()
    parser.set_language(LANGUAGE)
    
    tree = parser.parse(code)
    root_node = tree.root_node

    return traverse_tree(root_node, [], lang)

def logic_gate(prev_edit_hunks: list, lang: str):
    '''Returns ("edit-type", result)
    '''
    prior_edits = [CodeWindow(hunk, "hunk") for hunk in prev_edit_hunks]
    
    code_before = prior_edits[-1].before_edit_region(split_by_line=False, allow_fuzzy=False)
    code_after = prior_edits[-1].after_edit_region(split_by_line=False)

    rename_result = is_rename_edit(code_before, code_after, lang)
    if rename_result is not False:
        return "rename", rename_result

    refdef_result = is_defref_edit(code_before, code_after, lang)
    if refdef_result is not False:
        return "def&ref", refdef_result
    
    clone_result = is_clone_edit(prior_edits, lang)
    if clone_result is not False:
        if clone_result is None:
            raise ValueError("clone_result is None")
        return "clone", clone_result
    else:
        return "normal", None
