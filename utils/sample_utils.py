import logging

logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
)

from .parser_utils import get_query, parse_md_doc, get_parser
from tree_sitter import *

parser = get_parser()

unsafe_fn_pattern = '(function_item (function_modifiers ("unsafe"))) @unsafe_function'
unsafe_fn_query = get_query(unsafe_fn_pattern)
unsafe_block_pattern = "(unsafe_block) @unsafe_block"
unsafe_block_query = get_query(unsafe_block_pattern)
fn_call_pattern = """
(field_expression field: (_) @function_call)
(scoped_identifier name: (_) @function_call)
(call_expression function: (identifier) @function_call)
"""
fn_call_query = get_query(fn_call_pattern)
type_pattern = "(type_identifier) @type"
type_query = get_query(type_pattern)
self_pattern = "(self) @self"
self_query = get_query(self_pattern)
line_comment_query = get_query("(line_comment) @line_comment")


def find_parent_fn_node(node: Node):
    while node is not None:
        if node.type == "function_item":
            return node
        node = node.parent
    return None


def trait_node(fn_node: Node) -> bool:
    """
    Get the outer trait node of fn_node
    """
    cur_node = fn_node
    while cur_node is not None:
        if cur_node.type == "trait_item":
            return cur_node
        cur_node = cur_node.parent
    return None


def in_pub_trait(fn_node: Node) -> bool:
    """
    Check if the function is in a public trait
    """
    trait = trait_node(fn_node)
    if trait is None:
        return False
    return trait.text.decode().lstrip().startswith("pub")


def get_prev_comment(node: Node):
    cur_node = node.prev_sibling
    comment = ""
    while cur_node is not None:
        if (
            cur_node.type == "attribute_item"
            or cur_node.type == "inner_attribute_item"
            or cur_node.start_point[0] == node.start_point[0]
        ):
            cur_node = cur_node.prev_sibling
        elif cur_node.type == "line_comment":
            comment = cur_node.text.decode().strip() + "\n" + comment
            cur_node = cur_node.prev_sibling
        else:
            break
    return comment


def node_of_unsafe_func(fn_node: Node) -> bool:
    """
    Check if the function is unsafe, the node param should be the function node
    """
    node = fn_node.child_by_field_name("name")
    while node is not None:
        # ! the previous sibling - `function_modifiers` could be `const unsafe`, so use in instead of ==
        if "unsafe" in node.text.decode():
            return True
        node = node.prev_sibling
    return False


def node_of_pub_func(fn_node: Node) -> bool:
    """
    Check if the function is public, the node param should be the function node
    """
    node = fn_node.child_by_field_name("name")
    while node is not None:
        if node.text.decode().startswith("pub"):
            return True
        node = node.prev_sibling
    return False


def get_prev_block_comment(node: Node):
    while node.parent is not None and node.parent.start_point[0] == node.start_point[0]:
        node = node.parent
    return get_prev_comment(node)


def get_inner_block_comment(node: Node):
    """
    the inner comment of unsafe block is the comment of the first line
    child(0) is `unsafe`, child(1) is `block`
    """
    comment = ""
    cur_node = node.child(1)
    if node.child_count < 2:
        return comment
    cur_node = cur_node.child(1)
    while cur_node is not None:
        if cur_node.type == "line_comment":
            comment = comment + cur_node.text.decode().strip() + "\n"
        cur_node = cur_node.next_sibling
    return comment


class Sample:
    def __init__(self, sample: dict):
        self.sample_label = sample["sample_label"]
        self.raw_code = sample["raw_code"]
        self.code_with_hints = sample["code_with_hints"]
        self.fn_comment = sample["comment"]
        self.fn_name = sample["name"]
        self.params = sample["params"]
        self.fn_attributes = sample.get("attributes", "")
        # including declaration, description and safety
        self.unsafe_callees = list()
        # including info for type, type(struct, enum, generic), trait
        self.context_detail = list()
        # hover info for `self` token
        self.self_info = sample["self_info"]
        self.sample_detail_in_markdown = None
        self.resolve_context_detail(sample["detail_info"])

    def resolve_context_detail(self, detail_info):
        for item in detail_info:
            # if item["item_type"] not in ["struct", "enum", "function"]:
            #     continue
            doc = parse_md_doc(item["documentation"])
            if len(doc) > 0 and doc[0]["type"] == "description":
                description = doc[0]["content"]
            else:
                description = ""
            declaration = item["declaration"]
            if item["item_type"] == "function" and "unsafe " in item["declaration"]:
                safety = ""
                for section in doc:
                    if section["type"].lower() == "safety":
                        safety = section["content"]
                        break
                self.unsafe_callees.append(
                    {
                        "src_mod": item["src_mod"],
                        "name": item["name"],
                        "declaration": declaration,
                        "description": description,
                        "safety": safety,
                    }
                )
            else:
                self.context_detail.append(
                    {
                        "src_mod": item["src_mod"],
                        "name": item["name"],
                        "type": item["item_type"],
                        "declaration": declaration,
                        "description": description,
                    }
                )

    def has_unsafe_callee(self) -> bool:
        return len(self.unsafe_callees) > 0

    def has_safety_constraint(self) -> bool:
        # check if there is at least one unsafe function with safety section
        has_safety = False
        for item in self.unsafe_callees:
            if len(item["safety"]) > 0:
                has_safety = True
                break
        return has_safety

    def has_safety_comment(self) -> bool:
        # check whether all inner unsafe blocks have safety comment
        unsafe_block_query = get_query("(unsafe_block) @unsafe_block")
        fn_node = get_parser().parse(self.raw_code.encode("utf-8")).root_node
        block_nodes = unsafe_block_query.captures(fn_node)
        for block, _ in block_nodes:
            comment = get_prev_block_comment(block) + get_inner_block_comment(block)
            # if len(comment) == 0:
            # commtent not contain "Safety" or "SAFETY"
            if not ("Safety" in comment or "SAFETY" in comment):
                return False
        return True

    def has_other_parameter(self) -> bool:
        for p in self.params:
            if p != "self":
                return True
        return False

    def too_long_code(self) -> bool:
        return len(self.code_with_hints.split("\n")) > 64

    def nested_fn(self) -> bool:
        return self.code_with_hints.count(" fn ") > 1

    def to_markdown(self, context_of_reference:bool=True, add_hints:str="all") -> str:
        """
        serialize the sample to str in the style of markdown
        """
        if not context_of_reference:
            context_str = ""
        else:
            # self information
            declaration = add_prefix_to(f"self declaration", wrap_in_code_block(self.self_info["declaration"]))
            description = add_prefix_to(f"self description", wrap_in_text(get_description_from_doc(self.self_info["documentation"])))
            context_str = add_section_title(f"### `self` information - {wrap_in_inline_block(self.self_info['name'])}", declaration + description)

            # other context information
            for item in self.context_detail:
                item_type = item["type"]
                if item_type == "function" and item["description"] == "":
                    continue
                declaration = add_prefix_to(f"{item_type} declaration", wrap_in_code_block(item["declaration"]))
                description = add_prefix_to(f"{item_type} description", wrap_in_text(item["description"]))
                context_str = merge_content(context_str, add_section_title(f"### {item_type} - {wrap_in_inline_block(item['name'])}", declaration + description))

            # add unsafe callee information
            unsafe_fn_str = ""
            for item in self.unsafe_callees:
                declaration = add_prefix_to("function declaration", wrap_in_code_block(item["declaration"]))
                description = add_prefix_to("function description", wrap_in_text(item["description"]))
                unsafe_fn_str = unsafe_fn_str + add_section_title(f"### Unsafe Function - {wrap_in_inline_block(item['name'])}", declaration + description)
            context_str = add_section_title("## Reference Information", merge_content(context_str, unsafe_fn_str))

        # delete `unsafe` token in signature
        if add_hints == "all":
            fn = self.code_with_hints
        elif add_hints == "type":
            fn = self.code_with_type
        elif add_hints == "param":
            fn = self.code_with_param
        else:
            # add_hints == "none"
            fn = self.raw_code
        fn_blocks = fn.split("{")
        fn_blocks[0] = fn_blocks[0].replace(" unsafe ", " ")
        fn_body = "{".join(fn_blocks)

        fn_body = prettify_code_indent(fn_body)
        fn_body = delete_debug_assert(fn_body)
        fn_body = add_global_unsafe_block(fn_body)

        fn_body = delete_comment_around_unsafe(fn_body)
        fn_body = self.fn_attributes + "\n" + fn_body
        fn_body = add_prefix_to("function body", wrap_in_code_block(fn_body, "rust"))
        if context_of_reference:
            description = get_description_from_doc(comment_resolve(self.fn_comment))
            description = add_prefix_to("function description", wrap_in_text(description))
        else:
            description = ""
        fn_content = add_section_title("## Function To Be Checked", description + fn_body)

        return merge_content(context_str, fn_content)


def delete_comment_around_unsafe(fn_definition: str) -> str:
    """
    delete comment in fn body, especially the comment before and inside unsafe block.
    return the fn definition with comments before and inside unsafe block deleted;
    The comments away from unsafe blocks are kept.
    """
    # get line number of comment
    comment_lines = set()
    # get node - unsafe block with tree sitter
    root = get_parser().parse(fn_definition.encode("utf-8")).root_node
    unsafe_block_nodes = unsafe_block_query.captures(root)
    for node, _ in unsafe_block_nodes:
        # find minimal parent node of unsafe block
        parent_node = node
        while (
            parent_node.parent is not None
            and parent_node.parent.type != "block"
        ):
            parent_node = parent_node.parent

        # get prev comment line number:
        cur_node = parent_node.prev_sibling
        while cur_node is not None:
            if (
                cur_node.type == "attribute_item"
                or cur_node.type == "inner_attribute_item"
                or cur_node.start_point[0] == node.start_point[0]
            ):
                cur_node = cur_node.prev_sibling
            elif cur_node.type == "line_comment":
                comment_lines.add(cur_node.start_point[0])
                cur_node = cur_node.prev_sibling
            else:
                break

        # get inner unsafe block comment line number:
        inner_line_comments = line_comment_query.captures(parent_node)
        for comment, _ in inner_line_comments:
            comment_lines.add(comment.start_point[0])

    lines = fn_definition.split("\n")
    # support inline comment
    fn_without_comment = ""
    for idx, line in enumerate(lines):
        if idx in comment_lines:
            sign_idx = line.find("//")
            valid_code = line[:sign_idx].rstrip()
            if len(valid_code) > 0:
                fn_without_comment += valid_code + "\n"
        else:
            fn_without_comment += line + "\n"
    return fn_without_comment


def prettify_code_indent(fn_definition: str) -> str:
    # prettify the code indent
    # should call this method before add global unsafe block
    fn_definition = fn_definition.replace("\t", " " * 4)
    fn_lines = fn_definition.split("\n")
    processed_lines = list()
    line_of_brace = -1
    for idx, line in enumerate(fn_lines):
        if "{" in line:
            line_of_brace = idx
            break
    if line_of_brace + 1 >= len(fn_lines):
        return fn_definition
    first_line = fn_lines[line_of_brace + 1]
    # count the number of spaces before the first line
    prefix_space = 0
    for c in first_line:
        if c == " ":
            prefix_space += 1
        else:
            break
    if prefix_space == 4:
        return fn_definition
    else:
        prefix_space -= 4
        processed_lines.append(fn_lines[0])
        for line in fn_lines[1:]:
            processed_lines.append(line[prefix_space:])
    return "\n".join(processed_lines)


def add_global_unsafe_block(fn_definition: str) -> str:
    """
    Add global unsafe block for unsafe function without unsafe block.
    """
    # if the function already has inner unsafe block, return directly
    if "unsafe {" in fn_definition or "unsafe{" in fn_definition:
        return fn_definition
    fn_lines = fn_definition.split("\n")
    # find the first line of function body
    line_of_brace = -1
    for idx, line in enumerate(fn_lines):
        if "{" in line:
            line_of_brace = idx
            break
    if line_of_brace + 1 >= len(fn_lines):
        return fn_definition
    processed_lines = fn_lines[:line_of_brace + 1].copy()
    processed_lines.append("    unsafe {")
    for line in fn_lines[line_of_brace + 1: -1]:
        processed_lines.append("    " + line)
    processed_lines.append("    }")
    processed_lines.append("}")
    return "\n".join(processed_lines)


def delete_debug_assert(fn_definition: str) -> str:
    """
    delete debug_assert! in fn body
    use tree_sitter to handle the case that debug_assert cross multiple lines
    """
    root = get_parser().parse(fn_definition.encode("utf-8")).root_node
    query = get_query("(macro_invocation) @macro")
    macro_nodes = query.captures(root)
    fn_lines = fn_definition.split("\n")
    skip_lines = set()
    if len(macro_nodes) == 0:
        return fn_definition
    for node, _ in macro_nodes:
        identifier = node.child_by_field_name("macro")
        if "debug_assert" in identifier.text.decode():
            for i in range(node.start_point[0], node.end_point[0] + 1):
                skip_lines.add(i)
    return "\n".join([line for idx, line in enumerate(fn_lines) if idx not in skip_lines])



def wrap_in_code_block(content: str, language: str = "rust") -> str:
    """
    wrap content in code block
    """
    if len(content) == 0:
        return ""
    content = content.strip()
    if content.find("\n") == -1:
        return wrap_in_inline_block(content)
    else:
        return f"```{language}\n{content}\n```\n"


def wrap_in_text(content: str) -> str:
    if len(content) == 0:
        return ""
    content = content.strip()
    if content.find("\n") == -1:
        return f'"{content}"\n'
    else:
        return f'"""\n{content}\n"""\n'


def wrap_as_bold_header(content: str) -> str:
    if len(content) == 0:
        return ""
    return f"**{content.strip()}**\n"


def wrap_in_inline_block(content: str) -> str:
    if len(content) == 0:
        return ""
    return f"`{content}`"


def content_inline(content: str):
    if len(content) == 0:
        return ""
    content = content.replace("\n", " ")
    content = " ".join(content.split())
    return content


def add_section_title(title, content: str):
    if len(content) == 0:
        return ""
    content = content.strip()
    return f"{title}:\n{content}\n"


def add_prefix_to(prefix, content: str):
    if len(content) == 0:
        return ""
    content = content.strip()
    if content.find("\n") == -1:
        return f"**{prefix}**: {content}\n"
    else:
        return f"**{prefix}**:\n{content}\n"


def merge_content(content: str, new_content: str, split_line=True):
    if len(content) == 0:
        return new_content
    if len(new_content) == 0:
        return content
    if split_line:
        return content + "\n" + new_content
    else:
        return content + new_content


def comment_resolve(comment: str) -> str:
    comment_lines = comment.split("\n")
    result = "\n".join([line.lstrip("/! \t") for line in comment_lines])
    return result


def get_description_from_doc(md_doc: str) -> str:
    doc = parse_md_doc(md_doc)
    if len(doc) > 0 and doc[0]["type"] == "description":
        description = doc[0]["content"]
    else:
        description = ""
    return description
