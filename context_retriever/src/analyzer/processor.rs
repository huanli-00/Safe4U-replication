use crate::{__private::*, analyzer::workspace::Workspace};
use anyhow::{Error, Result};
use ide::{Documentation, InlayHint, NavigationTarget};
use ide_db::SymbolKind;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};
use tree_sitter::{Node, Parser, Query, QueryCursor};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NavInfo {
    pub src_mod: String,
    pub name: String,
    pub item_type: String,
    // token offset in the fn
    pub offset_in_fn: usize,
    pub declaration: String,
    pub documentation: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FnInfo {
    pub sample_label: String,
    pub repo_name: String,
    pub relative_file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub name: String,
    pub raw_code: String,
    // code with all hints
    pub code_with_hints: String,
    // code with type hint
    pub code_with_type: String,
    // code with parameter hint
    pub code_with_param: String,
    pub params: Vec<String>,
    pub return_type: String,
    pub comment: String,
    pub attributes: String,
    pub detail_info: Vec<NavInfo>,
    pub self_info: NavInfo,
    // pub unsafe_blocks: Vec<(usize, usize)>,
}

#[derive(Debug, Default)]
pub struct ProcessContext {
    pub ws_path: String,
    pub abs_file: String,
    pub fn_info: FnInfo,
    pub cached_tokens: HashSet<String>,
}

impl ProcessContext {
    fn new(ws_path: &Path, repo_name: &str) -> Self {
        let fn_info = FnInfo {
            repo_name: repo_name.into(),
            ..FnInfo::default()
        };
        Self {
            ws_path: ws_path.as_os_str().to_str().unwrap().to_string(),
            fn_info,
            ..Self::default()
        }
    }
}

// processor that load workspace and provide query functions
pub struct Processor {
    workspace: Workspace,
    context: ProcessContext,
}

impl Processor {
    pub fn new(repo_root_path: &Path, repo_name: &str) -> Result<Self> {
        let repo_path = repo_root_path.join(repo_name);
        let workspace = Workspace::new(&repo_path)?;
        Ok(Self {
            workspace,
            context: ProcessContext::new(&repo_path, repo_name),
        })
    }

    pub fn processor_for_crate(
        repo_root_path: &Path,
        repo_name: &str,
        relative_file: &str,
    ) -> Result<(Self, String)> {
        let repo_path = repo_root_path.join(repo_name);
        let mut crate_path = repo_path.to_path_buf();
        let dirs: Vec<&str> = relative_file.split("/").collect();
        for (idx, dir) in dirs[..dirs.len() - 1].iter().enumerate() {
            crate_path = crate_path.join(dir);
            let workspace = Workspace::new(crate_path.as_path());
            if let Ok(workspace) = workspace {
                return Ok((
                    Self {
                        workspace,
                        context: ProcessContext::new(
                            crate_path.as_path(),
                            &format!("{}/{}", repo_name, dirs[..=idx].join("/")),
                        ),
                    },
                    dirs[idx + 1..].join("/"),
                ));
            }
        }
        Err(Error::msg("No workspace found"))
    }

    /// extend target function, including get hover infor and add inlay hints;
    ///
    /// path: relative path to workspace root
    ///
    /// line: line number
    pub fn process_fn_in_line(&mut self, relative_file: &str, line: usize) -> Result<FnInfo> {
        self.context.fn_info.relative_file = relative_file.into();
        self.context.abs_file = format!("{}/{}", self.context.ws_path, relative_file);
        let code = self.get_file_code(&self.context.abs_file);
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_rust::language()).unwrap();
        let tree = parser.parse(&code, None).unwrap();
        let fn_node = Self::get_target_item_in_line(tree.root_node(), line);
        match fn_node {
            None => {
                log::error!("No function found in line {line}");
                Err(Error::msg("No function found in line"))
            }
            Some(fn_node) => {
                self.context.fn_info.start_byte = fn_node.start_byte();
                self.context.fn_info.end_byte = fn_node.end_byte();
                self.context.fn_info.raw_code =
                    code[fn_node.start_byte()..fn_node.end_byte()].into();
                self.context.fn_info.start_line = fn_node.start_position().row;
                self.context.fn_info.end_line = fn_node.end_position().row;
                self.process_fn_node(fn_node)
            }
        }
    }

    /// extend target function, including get hover infor and add inlay hints
    ///
    /// path: relative path to workspace root
    ///
    /// range: (start, end) byte range
    pub fn process_fn_in_range(
        &mut self,
        relative_file: &str,
        range: (usize, usize),
    ) -> Result<FnInfo> {
        self.context.fn_info.relative_file = relative_file.into();
        self.context.fn_info.start_byte = range.0;
        self.context.fn_info.end_byte = range.1;
        self.context.abs_file = format!("{}/{}", self.context.ws_path, relative_file);
        let file_path = Path::new(&self.context.abs_file);
        let file_code = self.workspace.file_text(file_path).unwrap();
        let fn_code = &file_code[range.0..range.1];

        let mut parser = Parser::new();
        parser.set_language(tree_sitter_rust::language()).unwrap();
        let tree = parser.parse(fn_code, None).unwrap();
        let fn_node = tree.root_node().child(0).unwrap();
        self.context.fn_info.raw_code = fn_code.into();
        self.context.fn_info.start_line = match range.0 {
            0 => 0,
            _ => file_code[..range.0].lines().count() - 1,
        };
        self.context.fn_info.end_line =
            self.context.fn_info.start_line + fn_node.end_position().row;
        self.process_fn_node(fn_node)
    }

    pub fn process_fn_node(&mut self, node: Node) -> Result<FnInfo> {
        // get fn name, parameters and return type
        let node_offset = node.start_byte();
        let fn_code = &self.context.fn_info.raw_code;
        let fn_name = node.child_by_field_name("name").unwrap();

        // get description for the function
        let (comment, attributes) = get_prev_comment_and_attributes(
            Some(node),
            &self.get_file_code(&self.context.abs_file),
            false,
        );

        self.context.fn_info.comment = comment;
        self.context.fn_info.attributes = attributes;
        self.context.fn_info.name =
            fn_code[fn_name.start_byte() - node_offset..fn_name.end_byte() - node_offset].into();
        self.context.fn_info.sample_label = format!(
            "{}/{}/{}/{}",
            self.context.fn_info.repo_name,
            self.context.fn_info.relative_file,
            self.context.fn_info.name,
            self.context.fn_info.start_line
        );
        let parameter_pattern = "(parameter pattern: (_) @param) (self_parameter (self) @param)";
        let parameter_query = Query::new(tree_sitter_rust::language(), parameter_pattern).unwrap();
        let mut query_cursor = QueryCursor::new();
        let param_node = match node.child_by_field_name("parameters") {
            None => {
                return Err(Error::msg("No parameters found in function"));
            }
            Some(node) => node,
        };
        let captured_params =
            query_cursor.captures(&parameter_query, param_node, fn_code.as_bytes());

        for (t, _) in captured_params.into_iter() {
            for capture in t.captures {
                let offset_in_fn = capture.node.start_byte() - node_offset;
                let end_offset = capture.node.end_byte() - node_offset;
                let token = &fn_code[offset_in_fn..end_offset];
                self.context.fn_info.params.push(token.into());
            }
        }
        let return_type_node = node.child_by_field_name("return_type");
        if let Some(return_type) = return_type_node {
            let offset_in_fn = return_type.start_byte() - node_offset;
            let end_offset = return_type.end_byte() - node_offset;
            self.context.fn_info.return_type = fn_code[offset_in_fn..end_offset].into();
        }

        self.process_target_node(node);
        // add inlay hints to the code
        self.add_inlay_hints();

        // clear cache after processing and return fn info
        self.context.cached_tokens.clear();
        let repo_name = self.context.fn_info.repo_name.clone();
        Ok(std::mem::replace(
            &mut self.context.fn_info,
            FnInfo {
                repo_name,
                ..FnInfo::default()
            },
        ))
    }

    fn add_inlay_hints(&mut self) {
        self.context.fn_info.code_with_hints = self.context.fn_info.raw_code.clone();
        self.context.fn_info.code_with_param = self.context.fn_info.raw_code.clone();
        self.context.fn_info.code_with_type = self.context.fn_info.raw_code.clone();
        let file_path = Path::new(&self.context.abs_file);
        let fn_offset = self.context.fn_info.start_byte;
        let mut hints = match self.workspace.get_inlay_hints(
            file_path,
            &(
                self.context.fn_info.start_byte,
                self.context.fn_info.end_byte,
            ),
        ) {
            None => return,
            Some(hints) => hints,
        };
        let hint_offset = |hint: &InlayHint| match hint.position {
            ide::InlayHintPosition::Before => usize::from(hint.range.start()),
            ide::InlayHintPosition::After => usize::from(hint.range.end()),
        };

        // add hints to raw code from tail to head
        // sort hints by offset
        hints.sort_by(|a, b| (hint_offset(b)).cmp(&hint_offset(a)));
        for hint in hints {
            let mut start_offset: usize = hint_offset(&hint);
            start_offset -= fn_offset;
            match hint.kind {
                ide::InlayKind::Type => {
                    // add hint token to the code
                    let pad = hint.label.to_string();
                    self.context
                        .fn_info
                        .code_with_hints
                        .insert_str(start_offset, pad.as_str());
                    self.context
                        .fn_info
                        .code_with_type
                        .insert_str(start_offset, pad.as_str());
                }
                ide::InlayKind::Parameter => {
                    // for parameter hint, append a space to the hint
                    let pad = hint.label.to_string() + " ";
                    self.context
                        .fn_info
                        .code_with_hints
                        .insert_str(start_offset, pad.as_str());
                    self.context
                        .fn_info
                        .code_with_param
                        .insert_str(start_offset, pad.as_str());
                }
                _ => return,
            }
        }
    }

    /// get definition and cache it
    /// if the documentation is empty, try to retrieve the comment above the definition(begin with '//' instead of '///')
    fn cache_definition(&mut self, offset: usize) {
        let file_path = Path::new(&self.context.abs_file);
        let file_offset = self.context.fn_info.start_byte + offset;
        let nav_targets = self.workspace.get_definition(file_path, file_offset);
        match nav_targets {
            None => {
                log::error!("No definition found in offset {offset}");
            }
            Some(nav_targets) => {
                for mut nav in nav_targets {
                    let mut nav_info = self.get_navinfo(&mut nav);
                    if self.context.cached_tokens.contains(&nav_info.name)
                        || skip_common_token(&nav_info.name)
                    {
                        continue;
                    }
                    nav_info.offset_in_fn = offset;
                    self.context.cached_tokens.insert(nav_info.name.clone());
                    self.context.fn_info.detail_info.push(nav_info);
                }
            }
        }
    }

    /// get definition and cache it
    fn cache_type_definition(&mut self, offset: usize) {
        let file_path = Path::new(&self.context.abs_file);
        let file_offset = self.context.fn_info.start_byte + offset;
        let nav_targets = self.workspace.get_type_definition(file_path, file_offset);
        match nav_targets {
            None => {
                log::error!("No type definition found in offset {offset}");
            }
            Some(nav_targets) => {
                for mut nav in nav_targets {
                    let mut nav_info = self.get_navinfo(&mut nav);
                    if self.context.cached_tokens.contains(&nav_info.name) {
                        continue;
                    }
                    nav_info.offset_in_fn = offset;
                    self.context.cached_tokens.insert(nav_info.name.clone());
                    self.context.fn_info.detail_info.push(nav_info);
                }
            }
        }
    }

    fn get_doc_from_comment(&self, nav: &NavigationTarget) -> String {
        let file_path = &self.workspace.file_path(nav.file_id);
        let target_range = match nav.focus_range {
            None => nav.focus_or_full_range(),
            Some(range) => range,
        };
        if file_path.eq(&self.context.abs_file)
            && usize::from(target_range.start()) >= self.context.fn_info.start_byte
            && usize::from(target_range.end()) <= self.context.fn_info.end_byte
        {
            return "".to_string();
        }
        let file_code = self.get_file_code(&file_path);
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_rust::language()).unwrap();
        let tree = parser.parse(&file_code, None).unwrap();
        let node = tree.root_node();
        // get line number with byte offset
        let offset: usize = usize::from(target_range.start());
        let line = match offset {
            0 => 0,
            _ => file_code[..offset].lines().count() - 1,
        };
        let node = Self::get_target_item_in_line(node, line);
        let comment = get_prev_comment_and_attributes(node, &file_code, true).0;
        // remove annotation
        let comment_line = comment.split("\n").collect::<Vec<&str>>();
        let mut doc = "".to_string();
        for line in comment_line {
            doc += line.trim_start_matches("//").trim_start();
        }
        doc
    }

    // get file code by path
    fn get_file_code(&self, file_path: &str) -> String {
        self.workspace
            .file_text(Path::new(file_path))
            .unwrap_or("".to_string())
    }

    /// tranverse all target AST node with tree-sitter
    fn process_target_node(&mut self, node: Node) {
        // fill in fn info by find child with field name
        let node_offset = node.start_byte();
        let fn_code = self.context.fn_info.raw_code.clone();
        let fn_name = node.child_by_field_name("name").unwrap();
        self.context.fn_info.name =
            fn_code[fn_name.start_byte() - node_offset..fn_name.end_byte() - node_offset].into();

        // [ref](https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries)
        // [grammar](https://github.com/tree-sitter/tree-sitter-rust/blob/master/grammar.js)
        // [playground](https://tree-sitter.github.io/tree-sitter/playground)
        let target_pattern = r#"
            (self) @self
            (call_expression function: [
                (identifier) @function_call
                (scoped_identifier name: (_) @function_call)
                (field_expression field: (_) @function_call)
            ])
            (type_identifier) @type
            (struct_pattern type: (_) @type)
            (tuple_struct_pattern type: (_) @type)
            (macro_invocation (identifier) @macro)
            "#;
        let target_query = Query::new(tree_sitter_rust::language(), target_pattern).unwrap();
        let mut query_cursor = QueryCursor::new();
        let captured_targets = query_cursor.captures(&target_query, node, fn_code.as_bytes());
        for (t, _) in captured_targets.into_iter() {
            for capture in t.captures {
                let offset_in_fn = capture.node.start_byte() - node_offset;
                let end_offset = capture.node.end_byte() - node_offset;
                let token = &fn_code[offset_in_fn..end_offset];
                if skip_common_token(token) {
                    continue;
                }
                log::debug!("get detail info of token: {token}");
                // skip if the token is already cached
                if self.context.cached_tokens.contains(token) {
                    continue;
                }
                let info_num = self.context.fn_info.detail_info.len();
                if token == "self" || token == "Self" {
                    self.context.cached_tokens.insert("Self".into());
                    self.context.cached_tokens.insert("self".into());
                    self.cache_type_definition(offset_in_fn);
                    // did not find new detail info
                    if self.context.fn_info.detail_info.len() == info_num {
                        continue;
                    }
                    self.context.fn_info.self_info =
                        self.context.fn_info.detail_info.pop().unwrap();
                } else {
                    self.cache_definition(offset_in_fn);
                    if self.context.fn_info.detail_info.len() == info_num {
                        continue;
                    }
                }
            }
        }
    }

    /// get fn item start at `line` with tree-sitter
    fn get_target_item_in_line<'a>(root: Node<'a>, line: usize) -> Option<Node<'a>> {
        let target_item_kinds = [
            "function_item",
            "function_signature_item",
            "struct_item",
            "enum_item",
            "type_item",
            "macro_definition",
            "union_item",
        ];
        let mut node = Some(root);
        while let Some(n) = node {
            if n.end_position().row < line {
                node = n.next_sibling();
            } else if n.start_position().row > line {
                return None;
            } else if target_item_kinds.contains(&n.kind()) {
                return Some(n);
            } else {
                node = n.child(0);
            }
        }
        None
    }

    fn get_navinfo(&self, nav: &mut ide::NavigationTarget) -> NavInfo {
        let doc = nav
            .docs
            .clone()
            .map(Documentation::into)
            .unwrap_or(self.get_doc_from_comment(nav));
        let item_type = map_symbol_kind(nav.kind.unwrap());

        let declaration = match item_type.as_str() {
            "struct" | "enum" | "union" | "type_alias" => {
                let file_content =
                    self.get_file_code(self.workspace.file_path(nav.file_id).as_str());
                file_content[nav.focus_or_full_range().start().into()..nav.full_range.end().into()]
                    .to_string()
            }
            _ => nav.description.clone().unwrap_or("".to_string()),
        };
        NavInfo {
            // FIXME: get the correct module name
            // container_name is not full path...
            src_mod: match &nav.container_name {
                Some(name) => name.clone().into(),
                _ => "".to_string(),
            },
            name: nav.name.to_string(),
            item_type,
            // nav.description is actually the declaration
            declaration,
            documentation: doc,
            ..NavInfo::default()
        }
    }
}

fn get_prev_comment_and_attributes(
    mut node: Option<Node>,
    file_code: &str,
    remove_annotation: bool,
) -> (String, String) {
    let mut comment = "".to_string();
    let mut attributes = "".to_string();
    node = match node {
        None => return (comment, attributes),
        Some(node) => node.prev_sibling(),
    };
    while let Some(line_node) = node {
        if line_node.kind() == "line_comment" {
            // the end_byte should be included otherwise the ending '\n' will be missing
            let mut line_content = &file_code[line_node.start_byte()..=line_node.end_byte()];
            if remove_annotation {
                while line_content.starts_with("/") || line_content.starts_with(" ") {
                    line_content = &line_content[1..];
                }
                if line_content.len() == 0 {
                    line_content = "\n".into();
                }
            } else {
                line_content = line_content.trim_start();
            }
            // add new comment line to the beggining of the comment
            comment = line_content.to_string() + &comment;
        } else if ["attribute_item", "inner_attribute_item", "macro_invocation"]
            .contains(&line_node.kind())
        {
            let line_content = &file_code[line_node.start_byte()..=line_node.end_byte()];
            attributes = line_content.to_string() + &attributes;
        } else {
            break;
        }
        node = line_node.prev_sibling();
    }
    (comment, attributes)
}

fn skip_common_token(s: &str) -> bool {
    let common_calls = [
        // common struct or enum
        "CStr",
        "String",
        "OsStr",
        "OsString",
        "Vec",
        "vec",
        "HashMap",
        "HashSet",
        "Option",
        "Result",
        "Error",
        "Box",
        "Rc",
        "RefCell",
        "Mutex",
        "Arc",
        "Range",
        "Cell",
        "Err",
        "Some",
        "Ok",
        // common safe functions
        "ok",
        "and_then",
        "ok_or",
        "ptr",
        "pointer",
        "as_ptr",
        "as_mut_ptr",
        "len",
        "from",
        "drop",
        "lock",
        "borrow",
        "new",
        "into",
        "unwrap",
        "as_ref",
        "clone",
        "to_string",
        "to_owned",
        "append",
        "push_str",
        "insert",
        "remove",
        "push",
        "pop",
        "is_empty",
        "is_none",
        "is_some",
        "is_err",
        "enumerate",
        "default",
        "iter",
        "iter_mut",
        "collect",
        "into_iter",
        "map",
        "filter",
        "unwrap_or",
        "unwrap_or_else",
        "unwrap_or_default",
        "as_bytes",
        "as_str",
        "as_bytes_mut",
        "as_mut",
        "borrow_mut",
        "set",
        "get",
        "unlock",
        "null",
        "is_null",
        // primitive types
        "c_void",
        "c_int",
        "c_char",
        "char",
        "str",
        "slice",
        "u8",
        "u16",
        "u32",
        "u64",
        "i8",
        "i16",
        "i32",
        "i64",
        "f32",
        "f64",
        "usize",
        "isize",
        // common traint
        "Debug",
        "Display",
        "Clone",
        "PartialEq",
        "PartialOrd",
        "Eq",
        "Ord",
        "Hash",
        "Default",
        "Copy",
        "Send",
        "Sync",
        "Deref",
        "DerefMut",
        "From",
        "FromStr",
        "Into",
        "TryInto",
        "TryFrom",
        "AsRef",
        "AsMut",
        "Iterator",
        "IntoIterator",
        "ToString",
        "Read",
        "Write",
        // common macros
        "println",
        "print",
        "eprintln",
        "format",
        "format_args",
        "dbg",
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
        "concat",
        "concat_idents",
        "eprint",
        "eprintln",
        "write",
        "writeln",
        "panic",
        "unreachable",
        "unimplemented",
        "todo",
        "cfg",
        "cfg_attr",
        "cfg_if",
        "feature",
        "vec",
        "sql",
        "log",
        "env",
        "env_var",
    ];
    common_calls.contains(&s)
}

fn map_symbol_kind(kind: SymbolKind) -> String {
    match kind {
        SymbolKind::Attribute => "attribute",
        SymbolKind::BuiltinAttr => "builtin_attr",
        SymbolKind::Const => "const",
        SymbolKind::ConstParam => "const_param",
        SymbolKind::Derive => "derive",
        SymbolKind::DeriveHelper => "derive_helper",
        SymbolKind::Enum => "enum",
        SymbolKind::Field => "field",
        SymbolKind::Function => "function",
        SymbolKind::Impl => "impl",
        SymbolKind::Label => "label",
        SymbolKind::LifetimeParam => "lifetime_param",
        SymbolKind::Local => "local",
        SymbolKind::Macro => "macro",
        SymbolKind::Module => "module",
        SymbolKind::SelfParam => "self_param",
        SymbolKind::SelfType => "self_type",
        SymbolKind::Static => "static",
        SymbolKind::Struct => "struct",
        SymbolKind::ToolModule => "tool_module",
        SymbolKind::Trait => "trait",
        SymbolKind::TraitAlias => "trait_alias",
        SymbolKind::TypeAlias => "type_alias",
        SymbolKind::TypeParam => "type_param",
        SymbolKind::Union => "union",
        SymbolKind::ValueParam => "value_param",
        SymbolKind::Variant => "variant",
        SymbolKind::Method => "method",
        SymbolKind::ProcMacro => "proc_macro",
    }
    .to_string()
}

#[test]
fn test_skip_token() {
    assert!(skip_common_token("Arc"));
    assert!(skip_common_token("Vec"));
    assert!(skip_common_token("print"));
    assert!(skip_common_token("println"));
}
