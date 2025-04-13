use std::path::Path;

use anyhow::Result;
use hir::ClosureStyle;
use ide::{AnalysisHost, FilePosition, InlayHint, InlayHintsConfig, NavigationTarget};
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice};
use paths::{AbsPathBuf, Utf8PathBuf};
use project_model::{
    CargoConfig, CargoFeatures, RustLibSource,
};
use vfs::{FileId, Vfs, VfsPath};

use crate::__private::*;

pub struct Workspace {
    pub host: AnalysisHost,
    pub vfs: Vfs,
}

type Navigations = Vec<NavigationTarget>;

impl Workspace {
    pub fn new(path: &Path) -> Result<Self> {
        let cargo_config = default_cargo_config();
        let load_config = default_load_cargo_config();
        // default progress function, do nothing
        let progress = |_: String| {};
        let (db, vfs, _proc_macro_client) =
            load_cargo::load_workspace_at(path, &cargo_config, &load_config, &progress)?;
        Ok(Self {
            host: AnalysisHost::with_database(db),
            vfs,
        })
    }

    pub fn file_id(&self, file_path: &Path) -> Option<FileId> {
        let path = Utf8PathBuf::from_path_buf(file_path.canonicalize().unwrap()).unwrap();
        self.vfs.file_id(&VfsPath::from(AbsPathBuf::assert(path)))
    }

    pub fn file_path(&self, file_id: FileId) -> String {
        self.vfs.file_path(file_id).to_string()
    }

    pub fn file_text(&self, file_path: &Path) -> Option<String> {
        self.file_id(file_path)
            .map(|id| self.host.analysis().file_text(id).unwrap().to_string())
    }

    // get all type definitions for symbol at offset in file
    pub fn get_type_definition(&self, file_path: &Path, offset: usize) -> Option<Navigations> {
        let info = self.file_id(file_path).and_then(|id| {
            self.host
                .analysis()
                .goto_type_definition(FilePosition {
                    file_id: id,
                    offset: offset.try_into().unwrap(),
                })
                .unwrap()
        });
        info.map(|i| i.info)
    }

    // get all type definitions for symbol at offset in file
    pub fn get_definition(&self, file_path: &Path, offset: usize) -> Option<Navigations> {
        let info = self.file_id(file_path).and_then(|id| {
            self.host
                .analysis()
                .goto_definition(FilePosition {
                    file_id: id,
                    offset: offset.try_into().unwrap(),
                })
                .unwrap()
        });
        info.map(|i| i.info)
    }

    // get all inlay hints in range
    pub fn get_inlay_hints(
        &self,
        file_path: &Path,
        range: &(usize, usize),
    ) -> Option<Vec<InlayHint>> {
        self.file_id(file_path).map(|id| {
            self.host
                .analysis()
                .inlay_hints(
                    &default_inlay_hints_config(),
                    id,
                    Some(ide::TextRange::new(
                        (range.0 as u32).into(),
                        (range.1 as u32).into(),
                    )),
                )
                .unwrap()
        })
    }
}

fn default_load_cargo_config() -> LoadCargoConfig {
    LoadCargoConfig {
        // FIXME: !Must enable this to run build scripts(build.rs)!!!
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: false,
    }
}

fn default_cargo_config() -> CargoConfig {
    let features = CargoFeatures::All;
    let sysroot = Some(RustLibSource::Discover);
    CargoConfig {
        features,
        sysroot,
        ..Default::default()
    }
}

fn default_inlay_hints_config() -> InlayHintsConfig {
    // default inlay hints config: if I don't know what it means, I won't config to ignore it
    InlayHintsConfig {
        discriminant_hints: ide::DiscriminantHints::Never,
        render_colons: true,   // enable
        type_hints: true,      // enable
        parameter_hints: true, // enable
        chaining_hints: false,
        lifetime_elision_hints: ide::LifetimeElisionHints::Never,
        closure_return_type_hints: ide::ClosureReturnTypeHints::Always,
        closure_capture_hints: true, // enable
        adjustment_hints: ide::AdjustmentHints::Never,
        adjustment_hints_mode: ide::AdjustmentHintsMode::Prefix,
        adjustment_hints_hide_outside_unsafe: false,
        binding_mode_hints: true, // enable
        hide_named_constructor_hints: false,
        hide_closure_initialization_hints: false,
        closure_style: ClosureStyle::ImplFn,
        param_names_for_lifetime_elision_hints: false,
        max_length: None,
        closing_brace_hints_min_lines: None,
        fields_to_resolve: ide::InlayFieldsToResolve {
            //enable
            resolve_text_edits: true,
            resolve_hint_tooltip: true,
            resolve_label_tooltip: true,
            resolve_label_location: true,
            resolve_label_command: true,
        },
        implicit_drop_hints: false,
        range_exclusive_hints: false,
    }
}
