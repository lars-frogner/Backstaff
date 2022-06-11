//! Command line interface.

pub mod build;
pub mod completions;
pub mod interpolation;
pub mod mesh;
pub mod run;
pub mod snapshot;
pub mod utils;

#[cfg(feature = "command-graph")]
pub mod command_graph;

#[cfg(feature = "seeding")]
pub mod seeding;

#[cfg(feature = "tracing")]
pub mod tracing;

#[cfg(feature = "ebeam")]
pub mod ebeam;

#[macro_export]
macro_rules! update_command_graph {
    ($parent_command_name:expr, $child_command_name:expr) => {
        #[cfg(feature = "command-graph")]
        $crate::cli::command_graph::insert_command_graph_edge(
            $parent_command_name,
            $child_command_name,
        );
    };
}
