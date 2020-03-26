# Backstaff

A flexible tookit for working with Bifrost simulations, written in [Rust](https://www.rust-lang.org/).

## Purpose

The purpose of this project is to provide a fast, reliable and flexible framework for computations on Bifrost simulation data. The original motivation for the project was for me to have a way of experimenting with electron beam simulations. Using the Fortran module integrated in Bifrost would be too cumbersome, and a Python script would be far too slow. As I implemented necessary capabilities like reading and representing snapshots, interpolation and field line tracing, I began to see a lot of uses apart from electron beam simulations, and therefore adopted a modular structure where it would be easy to add new capabilities. Since the design is based on interfaces and generics it is also convenient to manage several different implementations of the same functionality. In the long term, I think such a unified framework would be the ideal place to implement common tasks like snapshot preparation and analysis or experiment with potential Bifrost features.

## Why Rust?

Rust is highly suited for this project, for a number of reasons. It is a low-level systems language with performance on par with C++. It has a strong focus on memory safety, with a unique ownership system that can guarantee the absence of undefined behaviour (i.e. no segfaults). This also makes it easy to parallelize in a reliable manner, as issues like data races can be detected at compile time. Despite the focus on performance it is easy to write modular and elegant code thanks to the presence of zero-cost abstractions and elements from functional programming. The included `cargo` package manager makes it strightforward to download dependencies, compile and run the code and generate documentation. These advantages, helped by the excellent free introductory book [The Rust Programming Language](https://doc.rust-lang.org/book/), mean that the language rapidly is gaining popularity.

## Prerequesites

You need to have the Rust toolchain installed in order to build the binaries. Installation instructions can be found at https://www.rust-lang.org/tools/install.

If you want to work with [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) files, the `netCDF-C` library must be available to link with. Installation instructions can be found at https://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html.

## Installation

### Using `cargo install`

The Rust package manager `cargo` can be used to download, build and install the `backstaff` binary:
```
$ cargo install --git=https://github.com/lars-frogner/Backstaff.git
```
By default the binary will be placed in `$HOME/.cargo/bin`. A different directory can be specified with the option `--root=<DIR>`.

If installing with the `netcdf` feature (see the [Features](#features) section), you may have to inform the linker about the path to the external `netcdf` library. This is easily done through the `RUSTFLAGS` environment variable:
```
$ RUSTFLAGS='-L /path/to/library/directory' cargo install ...
```

**_NOTE:_** Compilation can be quite slow because some of the code relies heavily on macros, which are time consuming to compile. If this is an issue, compilation can be sped up by adding the `--branch=const-generics-interp`, which installs a branch using the experimental [Const generics](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md) Rust functionality to avoid macros. However, this requires that you activate the nightly Rust compiler by running `rustup default nightly` prior to `cargo install`. Revert to the stable compiler afterwards by running `rustup default stable`.

### Compiling from source

You can compile the code in this repository using the `cargo build` command. Make sure to add the `--release` flag so that optimizations are turned on.

## Features

The code consists of a core API as well as a set of optional features, some of which are included by default. You can specify additional features by adding the `--features` flag to `cargo install` or `cargo build`, e.g. `cargo build --features=tracing,netcdf`. The `--no-default-features` flag can be used to disable the default features.

Currently the available features are:
* `cli`: A module exposing a command line interface (CLI) for applying the various tools in the library. This feature is included by default, but can be disabled if you only want to use the API.
* `tracing`: A module for tracing field lines. Including it will add the `snapshot-trace` subcommand to the CLI.
* `ebeam`: A module for simulating electron beams. Including it will add the `snapshot-ebeam` subcommand to the CLI. Requires `tracing`.
* `netcdf`: Support for reading and writing simulation data in the [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) format (using the [CF conventions](http://cfconventions.org/)).

## API documentation

The API documentation can be generated and viewed in your browser by running `cargo doc --open` in the project repository. If using non-default features you need to specify them with a `--features` flag in order for them to be included in the documentation.

## Using the command line program

If you have installed the binary, simply run the `backstaff` command. If you instead are running directly from the repository, the simplest way to run the command line executable is with `cargo run` command. This will also perform any necessary compilation prior to running the program. All arguments following a double dash (`--`) will then be passed along to the `backstaff` program.

Actions are specified and configured through a hierachy of subcommands, which can be inspected by looking at their help texts. For example, the help text for the `snapshot` subcommand can be viewed as follows:
```
$ backstaff snapshot -h
```
```
backstaff-snapshot
Specify input snapshot to perform further actions on

USAGE:
    backstaff snapshot [FLAGS] [OPTIONS] <INPUT_FILE> <SUBCOMMAND>

FLAGS:
    -v, --verbose    Print status messages related to reading
    -h, --help       Print help information

OPTIONS:
    -r, --snap-range=<FIRST>,<LAST>    Inclusive range of snapshot numbers associated with the input snapshot to
                                       process [default: only process INPUT_FILE]
    -e, --endianness=<ENDIANNESS>      Endianness to assume for snapshots in native binary format
                                        [default: little]  [possible values: little, big, native]

ARGS:
    <INPUT_FILE>    Path to the file representing the snapshot.
                    Assumes the following format based on the file extension:
                        *.idl: Parameter file with associated .snap [and .aux] file
                        *.nc: NetCDF file using the CF convention (requires the netcdf feature)

SUBCOMMANDS:
    inspect     Inspect properties of the snapshot
    slice       Extract a 2D slice of a quantity field in the snapshot
    resample    Creates a resampled version of the snapshot
    write       Write snapshot data to file
```

Here is a graph of the command hierarchy available when all features are enabled.
![command_graph](figures/command_graph.png "Command graph")

This graph was created with the hidden `backstaff-command_graph` command, which outputs the command hierarchy graph in DOT format for rendering with [Graphviz](https://www.graphviz.org/).

## Examples

### Printing snapshot statistics

Printing some statistics for density and temperature in a snapshot could look like this:
```
$ backstaff snapshot photo_tr_001.idl inspect --included-quantities=r,tg statistics --slice-depths=-1
```
```
================================================================================
                  Statistics for r from snapshot 1 of photo_tr
--------------------------------------------------------------------------------
                         For all r, all x, all y, all z
================================================================================
Number of values:    452984832
Minimum value:       1.8624148e-8 at (  8.969,   6.625,  -5.802) [287, 212, 188]
Maximum value:       323.40679932 at ( 11.688,   4.500,   2.514) [374, 144, 767]
Average value:       13.853664398
5th percentile:      8.8137845e-8
30th percentile:     1.6243816e-7
50th percentile:     3.2699592e-7
70th percentile:     0.0031316155
95th percentile:     105.14699554
--------------------------------------------------------------------------------
                               In slice at z = -1
--------------------------------------------------------------------------------
Number of values:    589824
Minimum value:       3.3142376e-7 at ( 18.969,  14.500,  -1.000) [607, 464, 532]
Maximum value:       0.0221519507 at ( 17.719,  16.156,  -1.000) [567, 517, 532]
Average value:       0.0026214391
5th percentile:      3.2273060e-4
30th percentile:     0.0013502331
50th percentile:     0.0020803101
70th percentile:     0.0030976003
95th percentile:     0.0066802846
================================================================================
                 Statistics for tg from snapshot 1 of photo_tr
--------------------------------------------------------------------------------
                        For all tg, all x, all y, all z
================================================================================
Number of values:    452984832
Minimum value:       1998.7125244 at (  8.469,  19.344,  -1.240) [271, 619, 513]
Maximum value:       4.55296300e6 at ( 15.344,  14.969, -14.297) [491, 479,   0]
Average value:       9.67478562e5
5th percentile:      2988.0380859
30th percentile:     12443.400391
50th percentile:     45016.046875
70th percentile:     2.00756875e6
95th percentile:     2.36561025e6
--------------------------------------------------------------------------------
                               In slice at z = -1
--------------------------------------------------------------------------------
Number of values:    589824
Minimum value:       2003.6245117 at ( 14.156,   7.844,  -1.000) [453, 251, 532]
Maximum value:       1.52222038e6 at ( 18.938,  14.906,  -1.000) [606, 477, 532]
Average value:       5080.4008789
5th percentile:      2319.4228516
30th percentile:     2574.3144531
50th percentile:     2911.8168945
70th percentile:     3771.8098145
95th percentile:     5850.3222656
--------------------------------------------------------------------------------
```

### Tracing magnetic field lines

Here is a more complicated example where we trace a set of magnetic field lines from 100x100 regularly spaced locations in the upper chromosphere:
```
$ backstaff --timing \
    snapshot photo_tr_001.idl \
    trace --verbose field_lines.fl \
        basic_tracer --max-length=100.0 \
        slice_seeder --axis=z --coord=-2.0 \
            regular --shape=100,100
```
```
Found 10000 start positions
Successfully traced 10000 field lines
Saving field lines in field_lines.fl
Elapsed time: 12.731385333 s
```
**_NOTE:_** The program has to be built with the `--features=tracing` option in order for the `trace` command to become available.

Using the `backstaff` Python package included in this repository, we can easily read and visualize the field line data:
```python
import backstaff.field_lines as field_lines
field_line_set = field_lines.FieldLineSet3.from_file('field_lines.fl')
field_lines.plot_field_lines(field_line_set, alpha=0.01, output_path='field_lines.png')
```

This is the resulting figure:
![field_lines](figures/field_lines.png "Magnetic field lines")

### Creating NetCDF files for visualization

By enabling the `netcdf` feature, it is easy to convert snapshot data into the NetCDF format, which is supported by various visualization tools like [ParaView](https://www.paraview.org/) and [VAPOR](https://www.vapor.ucar.edu/). In the following example, the temperature and mass density fields in a set of Bifrost snapshots are resampled to a regular 512<sup>3</sup> grid and written to a set of NetCDF files.
```
$ backstaff \
    snapshot -v photo_tr_001.idl --snap-range=1,3 \
    resample -v --sample-location=center regular_grid --shape=512,512,512 \
    write -v --strip --included-quantities=r,tg photo_tr.nc
```
```
Writing grid to photo_tr_001.nc
Reading r from photo_tr_001.snap
Resampling r
Writing r to photo_tr_001.nc
Reading tg from photo_tr_001.aux
Resampling tg
Writing tg to photo_tr_001.nc
...
Writing grid to photo_tr_003.nc
Reading r from photo_tr_003.snap
Resampling r
Writing r to photo_tr_003.nc
Reading tg from photo_tr_003.aux
Resampling tg
Writing tg to photo_tr_003.nc
```

Here is a ParaView volume rendering of the resulting temperature field from one of the `.nc` files:
![volume_rendering](figures/volume_rendering.png "Volume rendering")

## Enabling tab-completion for `backstaff` arguments

The [clap](https://clap.rs/) argument parser powering the `backstaff` CLI can generate tab-completion files compatible with various shells. This can be done for `backstaff` using the hidden `backstaff-completions` command. To generate files for your shell, start by running
```
$ backstaff completions -h
```
and follow the instructions.

**_NOTE:_**  Unfortunately, automatic completion of file paths is currently not supported by the generated completion file, meaning that paths will not be completed inside the `backstaff` command if argument completion is enabled.
