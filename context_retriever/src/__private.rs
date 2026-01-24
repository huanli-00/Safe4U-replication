//! Wrapper around the `ra_ap_*` library
//!
//! Modules are re-exported here for convenience

pub(crate) use ra_ap_ide as ide;
pub(crate) use ra_ap_project_model as project_model;
pub(crate) use ra_ap_vfs as vfs;
pub(crate) use ra_ap_load_cargo as load_cargo;
pub(crate) use ra_ap_paths as paths;
pub(crate) use ra_ap_hir as hir;
pub(crate) use ra_ap_ide_db as ide_db;
// pub(crate) use ra_ap_text_edit as text_edit;