//! File input/output.

pub mod snapshot;
pub mod utils;

use atomic_counter::{AtomicCounter, RelaxedCounter};
use indicatif::{ProgressBar, ProgressStyle};

/// Little- or big-endian byte order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    const N_PROGRESS_UPDATES: usize = 100;

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
    pub fn create_progress_bar(&self, n_steps: usize) -> ParallelProgressBar {
        if let Verbosity::Progress(style) = self {
            ParallelProgressBar::new(n_steps, Self::N_PROGRESS_UPDATES).with_style(style.clone())
        } else {
            ParallelProgressBar::hidden()
        }
    }
}

/// Progress bar wrapper that uses atomic counters to limit
/// the number of times the lock on the progress bar is taken.
pub struct ParallelProgressBar {
    progress_bar: Option<ProgressBar>,
    visual_tick_counter: RelaxedCounter,
    tick_counter: RelaxedCounter,
    ticks_per_visual_tick: usize,
}

impl ParallelProgressBar {
    pub fn new(max_tick_count: usize, max_visual_tick_count: usize) -> Self {
        let progress_bar = Some(ProgressBar::new(max_tick_count as u64));
        let visual_tick_counter = RelaxedCounter::new(0);
        let tick_counter = RelaxedCounter::new(0);
        let ticks_per_visual_tick = max_tick_count / max_visual_tick_count;
        Self {
            progress_bar,
            visual_tick_counter,
            tick_counter,
            ticks_per_visual_tick,
        }
    }

    pub fn hidden() -> Self {
        let visual_tick_counter = RelaxedCounter::new(0);
        let tick_counter = RelaxedCounter::new(0);
        let ticks_per_visual_tick = 0;
        Self {
            progress_bar: None,
            visual_tick_counter,
            tick_counter,
            ticks_per_visual_tick,
        }
    }

    pub fn with_style(mut self, style: ProgressStyle) -> Self {
        if let Some(progress_bar) = self.progress_bar {
            self.progress_bar = Some(progress_bar.with_style(style));
        }
        self
    }

    pub fn inc(&self) {
        if let Some(ref progress_bar) = self.progress_bar {
            let tick_count = self.tick_counter.inc();
            let visual_tick_count = self.visual_tick_counter.get();
            if tick_count == visual_tick_count * self.ticks_per_visual_tick {
                self.visual_tick_counter.inc();
                progress_bar.inc(self.ticks_per_visual_tick as u64);
            }
        }
    }
}

/// How to handle existing files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverwriteMode {
    Ask,
    Always,
    Never,
}
