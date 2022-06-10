//! File input/output.

pub mod snapshot;
pub mod utils;

use indicatif::{ProgressBar, ProgressStyle};

/// Little- or big-endian byte order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Endianness {
    Native,
    Little,
    Big,
}

/// Whether and how to pass non-essential information to user.
#[derive(Clone, Debug)]
pub enum Verbosity {
    Messages,
    Quiet,
    Progress(ProgressStyle),
}

impl Verbosity {
    const N_PROGRESS_UPDATES: u64 = 100;

    /// Whether messages should be printed.
    pub fn print_messages(&self) -> bool {
        match *self {
            Verbosity::Messages => true,
            Verbosity::Quiet => false,
            Verbosity::Progress(_) => true,
        }
    }

    /// Returns a progress bar that is only visible if this is
    /// the `Progress` variant.
    pub fn create_progress_bar(&self, n_steps: usize) -> ProgressBar {
        let n_steps = n_steps as u64;
        if let Verbosity::Progress(style) = self {
            let progress_bar = ProgressBar::new(n_steps).with_style(style.clone());
            progress_bar.set_draw_delta(n_steps / Self::N_PROGRESS_UPDATES);
            progress_bar
        } else {
            ProgressBar::hidden()
        }
    }
}

/// How to handle existing files.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverwriteMode {
    Ask,
    Always,
    Never,
}
