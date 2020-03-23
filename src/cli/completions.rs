//! Command line interface for generating a command line completion script.

use super::build;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::io;

/// Creates a subcommand for printing derivable quantities.
pub fn create_completions_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("completions")
        .about("Generate tab-completion script for your shell")
        .setting(AppSettings::Hidden)
        .arg(
            Arg::with_name("shell")
                .value_name("SHELL")
                .required(true)
                .possible_values(&["bash", "zsh", "fish"])
                .help("The shell to generate the script for"),
        )
        .help_message("Print help information")
        .after_help(
            r#"DISCUSSION
    [Adapted from rustup-completions]
    Enable tab completion for Bash, Zsh or Fish
    The script is output on `stdout`, allowing one to re-direct the
    output to the file of their choosing. Where you place the file
    will depend on which shell, and which operating system you are
    using. Your particular configuration may also determine where
    these scripts need to be placed.

    Here are some common set ups for the three supported shells under
    Unix and similar operating systems (such as GNU/Linux).

    BASH:

    Completion files are commonly stored in `/etc/bash_completion.d/` for
    system-wide commands, but can be stored in
    `~/.local/share/bash-completion/completions` for user-specific commands.
    Run the command:

        $ mkdir -p ~/.local/share/bash-completion/completions
        $ cargo run --all-features -- completions bash >> ~/.local/share/bash-completion/completions/backstaff

    This installs the completion script. You may have to log out and
    log back in to your shell session for the changes to take affect.

    BASH (macOS/Homebrew):

    Homebrew stores bash completion files within the Homebrew directory.
    With the `bash-completion` brew formula installed, run the command:

        $ mkdir -p $(brew --prefix)/etc/bash_completion.d
        $ cargo run --all-features -- completions bash > $(brew --prefix)/etc/bash_completion.d/backstaff.bash-completion

    ZSH:

    ZSH completions are commonly stored in any directory listed in
    your `$fpath` variable. To use these completions, you must either
    add the generated script to one of those directories, or add your
    own to this list.

    Adding a custom directory is often the safest bet if you are
    unsure of which directory to use. First create the directory; for
    this example we'll create a hidden directory inside our `$HOME`
    directory:

        $ mkdir ~/.zfunc

    Then add the following lines to your `.zshrc` just before
    `compinit`:

        fpath+=~/.zfunc

    Now you can install the completions script using the following
    command:

        $ cargo run --all-features -- completions zsh > ~/.zfunc/_backstaff

    You must then either log out and log back in, or simply run

        $ exec zsh

    for the new completions to take affect.

    FISH:

    Fish completion files are commonly stored in
    `$HOME/.config/fish/completions`. Run the command:

        $ mkdir -p ~/.config/fish/completions
        $ cargo run --all-features -- completions fish > ~/.config/fish/completions/backstaff.fish

    This installs the completion script. You may have to log out and
    log back in to your shell session for the changes to take affect."#,
        )
}

pub fn run_completions_subcommand<'a, 'b>(arguments: &ArgMatches) {
    let shell = arguments
        .value_of("shell")
        .expect("No value for required argument.")
        .parse()
        .unwrap();
    build::build().gen_completions_to(clap::crate_name!(), shell, &mut io::stdout());
}
