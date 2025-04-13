from tree_sitter import *
from .file_utils import safe_open

RUST_LANGUAGE = Language("utils/build_parser/language_rust.so", "rust")
MARKDOWN_LANG = Language("utils/build_parser/language_markdown.so", "markdown")
rust_parser = Parser()
rust_parser.set_language(RUST_LANGUAGE)
md_parser = Parser()
md_parser.set_language(MARKDOWN_LANG)


def get_query(pattern: str):
    # query of rust language
    return RUST_LANGUAGE.query(pattern)


def get_parser() -> Parser:
    # parser of rust language
    return rust_parser


def get_syntax_tree(rust_file: str) -> Tree:
    # get syntax tree of rust file
    fp = safe_open(rust_file, "r")
    content = fp.read().encode("utf-8")
    tree = rust_parser.parse(content)
    fp.close()
    return tree


def parse_md_doc(md_doc: str) -> list:
    # there might be some error when the block is not ended with a new line '\n'
    # ref: https://ikatyang.github.io/tree-sitter-markdown/
    if len(md_doc) == 0:
        return list()
    md_doc = md_doc + "\n"
    document = md_parser.parse(md_doc.encode("utf-8")).root_node
    doc = list()
    # the first section should be the description (have no heading)
    section_type = "description"
    content = ""
    if document.child_count == 0:
        return doc
    cur_section = document.child(0)
    while cur_section is not None:
        for paragraph in cur_section.children:
            if paragraph.type == "atx_heading":
                section_type = paragraph.child(1).text.decode("utf-8")
                section_type = section_type.lower().strip()
            else:
                content = content + paragraph.text.decode("utf-8")
        doc.append({"type": section_type, "content": content})
        section_type = ""
        content = ""
        cur_section = cur_section.next_sibling
    return doc


def parse_hover_info(hover_info: str) -> dict:
    """
    parse hover info
    hover info is in markdown format with following structure:
    1. a code block: item source (mod)
    2. a code block: item declaration
    3. markdown text: item documenation that might be empty or multiple sections
    for example:
    ```rust
    crossbeam_epoch::atomic::Shared
    ```

    ```rust
    pub unsafe fn into_owned(self) -> Owned<T>
    ```

    ---

    Takes ownership of the pointee.

    # Panics

    Panics if this pointer is null, but only in debug mode.

    # Safety


    This method may be called only if the pointer is valid and nobody else is holding a
    reference to the same object.

    # Examples

    ```rust
    use crossbeam_epoch::{self as epoch, Atomic};
    use std::sync::atomic::Ordering::SeqCst;

    let a = Atomic::new(1234);
    unsafe {
        let guard = &epoch::unprotected();
        let p = a.load(SeqCst, guard);
        drop(p.into_owned());
        }
    ```
    """
    # there might be some error when the block is not ended with a new line '\n'
    hover_info = hover_info + "\n"
    root = md_parser.parse(hover_info.encode("utf-8")).root_node
    section = root.child(0)
    kid = section.child(0)
    node_before_break = list()
    while kid is not None and kid.type != "thematic_break":
        node_before_break.append(kid)
        kid = kid.next_sibling
    # get src_mod and declaration from nodes before the thematic break
    if len(node_before_break) == 1:
        src_mod = ""
        declaration = node_before_break[0].child(3).text.decode("utf-8")
    elif len(node_before_break) == 2:
        src_mod = node_before_break[0].child(3).text.decode("utf-8")
        declaration = node_before_break[1].child(3).text.decode("utf-8")

    # get description from the node after the thematic break
    doc = list()
    if kid is not None:
        kid = kid.next_sibling
        if kid is not None:
            doc.append({"type": "description", "content": kid.text.decode("utf-8")})
    # get other section
    while section.next_sibling is not None:
        section = section.next_sibling
        content = ""
        kid = section.child(0).next_sibling
        while kid is not None:
            content = content + kid.text.decode("utf-8")
            kid = kid.next_sibling
        doc.append(
            {
                "type": section.child(0).child(1).text.decode("utf-8"),
                "content": content,
            }
        )
    parsed_info = {"src_mod": src_mod, "declaration": declaration, "doc": doc}
    return parsed_info


def get_fn_params(fn_declaration: str) -> list:
    param_query = get_query(
        "(parameter pattern: (_) @param) (self_parameter (self) @param)"
    )
    tree = rust_parser.parse(fn_declaration.encode("utf-8"))
    captures = param_query.captures(tree.root_node)
    params = list()
    for node, _ in captures:
        params.append(node.text.decode("utf-8"))
    return params


def parse_md_list(md_list: str) -> list:
    """
    parse markdown list
    """
    # handling condition that response is wrapped as xml format. <xxx>md_list</xxx>
    if md_list[0] == "<":
        l_idx = 0
        r_idx = len(md_list) - 1
        while l_idx < len(md_list) and md_list[l_idx] != ">":
            l_idx = l_idx + 1
        while r_idx >= 0 and md_list[r_idx] != "<":
            r_idx = r_idx - 1
        md_list = md_list[l_idx + 1 : r_idx]

    md_list = md_list + "\n"
    node = md_parser.parse(md_list.encode("utf-8")).root_node
    node = node.child(0)
    node = node.child(0)
    items = list()
    # skip the text before the first list item
    while node is not None and node.type != "list":
        node = node.next_sibling

    if node is None:
        return items
    else:
        node = node.child(0)

    while node is not None:
        # ignore the first child (list annotation like '1.' or '*')
        if node.child_count > 1:
            items.append(node.child(1).text.decode("utf-8"))
        node = node.next_sibling
    return items
