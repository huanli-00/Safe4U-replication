# Rust Function Extender

Accept a location of one function in a workspace. Extend the function with following information:

- Inlay Hints: inlay hints inside the function, mainly includes type of variables and parameter names of function call.
- Hover Information: hover information of used types (Struct, Enum) and functions called.

## Pipeline

1. Init a extender with workspace path
2. Input a json file that contains a list of target functions. Each function is localized by relative_file, start_byte and end_byte.
3. Output extended function, including function body with inlay hints added, hover information of Types and Callee functions.

## Implementation Detail

1. The processor init workspace with `project_path`.
2. Parse the function with `syn` crate, visit(tranverse) the AST with customed impl of trait `Visit` and then get hover info for target items.
3. Add inlay hints to function token.