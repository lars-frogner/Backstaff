# bifrost-rust

A flexible tookit for working with Bifrost simulations, written in [Rust](https://www.rust-lang.org/).

## Purpose

The purpose of this project is to provide a fast, reliable and flexible framework for computations on Bifrost simulation data. The original motivation for the project was for me to have a way of experimenting with electron beam simulations. Using the Fortran module integrated in Bifrost would be too cumbersome, and a Python script would be far too slow. As I implemented necessary capabilities like reading and representing snapshots, interpolation and field line tracing, I began to see a lot of uses apart from electron beam simulations, and therefore adopted a modular structure where it would be easy to add new capabilities. Since the design is based on interfaces and generics it is also convenient to manage several different implementations of the same functionality. In the long term, I think such a unified framework would be the ideal place to implement common tasks like snapshot preparation and analysis or experiment with potential Bifrost features.

## Why Rust?

Rust is highly suited for this project, for a number of reasons. It is a low-level systems language with performance on par with C++. It has a strong focus on memory safety, with a unique ownership system that can guarantee the absence of undefined behaviour (i.e. no segfaults). This also makes it easy to parallelize in a reliable manner, as issues like data races can be detected at compile time. Despite the focus on performance it is easy to write modular and elegant code thanks to the presence of zero-cost abstractions and elements from functional programming. The included `cargo` package manager makes it strightforward to download dependencies, compile and run the code and generate documentation. These advantages, helped by the excellent free introductory book [The Rust Programming Language](https://doc.rust-lang.org/book/), mean that the language rapidly is gaining popularity.

## Prerequesites

You need to have the Rust toolchain installed in order to build the binaries. Installation instructions can be found at https://www.rust-lang.org/tools/install.

## Installation

Simply clone this repository.

## Compilation

You can compile the code using the `cargo build` command. Make sure to add the `--release` flag so that optimizations are turned on. The code consists of a core API as well as a set of optional features, some of which are included by default. You can specify additional features to include with the `--features` flag. The `--no-default-features` flag can be used to disable the default features.

## Features

Currently the available features are:
* `cli`: A module exposing a command line interface (CLI) for applying the various tools in the library. This feature is included by default.
* `tracing`: A module for tracing field lines. Including it will add the `snapshot-trace` subcommand to the CLI.
* `ebeam`: A module for simulating electron beams. Including it will add the `snapshot-ebeam` subcommand to the CLI. Requires `tracing`.

## Using the command line program

The simplest way to run the command line executable is with `cargo run` command. This will also perform any necessary compilation prior to running the program. All arguments following a double dash (`--`) will be passed along to the command line program.

Actions are specified and configured through a hierachy of subcommands, which can be inspected by looking at their help texts. For example, the help text for the `snapshot` subcommand can be viewed as follows:
```console
$ cargo run --release -- snapshot help
bifrost-snapshot
Specify input snapshot to perform further actions on

USAGE:
    bifrost snapshot [FLAGS] [OPTIONS] <PARAM_PATH> <SUBCOMMAND>

FLAGS:
    -v, --verbose    Print status messages while reading fields
    -h, --help       Prints help information

OPTIONS:
    -g, --grid-type=<TYPE>
            Type of grid to assume for the snapshot
             [default: horizontally-regular]  [possible values: horizontally-regular, regular]
    -e, --endianness=<ENDIANNESS>
            Endianness to assume for the snapshot
             [default: little]  [possible values: little, big]

ARGS:
    <PARAM_PATH>    Path to the parameter (.idl) file for the snapshot

SUBCOMMANDS:
    inspect     Inspect properties of the snapshot
    slice       Extract a 2D slice of a quantity field in the snapshot
    resample    Creates a resampled version of the snapshot
    help        Prints this message or the help of the given subcommand(s)
```

Printing some statistics for density and temperature in a snapshot could look like this:
```console
$ cargo run --release -- snapshot en024031_emer3.0str_ebeam_351.idl inspect statistics r tg
*************** Statistics for r ***************
Number of values: 452984832
Number of NaNs:   0
Minimum value:    0.00000001773288 at [316, 580, 92] = (9.875, 18.125, -8.746757)
Maximum value:    308.31104 at [707, 345, 767] = (22.09375, 10.78125, 2.513908)
Average value:    13.070756
*************** Statistics for tg ***************
Number of values: 452984832
Number of NaNs:   0
Minimum value:    1998.9036 at [674, 375, 432] = (21.0625, 11.71875, -2.244387)
Maximum value:    4063136.3 at [114, 375, 395] = (3.5625, 11.71875, -2.701525)
Average value:    30039.16
```

Tracing a set of 1000 field lines from random locations in the photosphere could look like this:
```console
$ cargo run --release --features tracing -- snapshot en024031_emer3.0str_ebeam_351.idl trace -v field_lines.pickle slice_seeder x 0.0 random 1000
Found 1000 start positions
Successfully traced 1000 field lines
Saving field lines in field_lines.pickle
```

## Documentation

The API documentation can be generated and viewed in your browser by running `cargo doc --open` in the project repository. If using non-default features you need to specify them with a `--features` flag in order for them to be included in the documentation.
