# Preparing Bifrost data for visualization using Backstaff

## Installing the `backstaff` tool

### Making sure NetCDF is available

In this tutorial we will create [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) files for visualization, and for that we need to have the NetCDF library available.

To check if is available on your system, run
```
nc-config --prefix
```
This will print the path to the directory where the library is located if it exists. If the command fails, install the library using (on MacOS)

```
brew install netcdf
```
or (on Linux)
```
sudo apt install netcdf
```

### Running the `setup_backstaff` installation script

The Backstaff repository contains an interactive shell script called `setup_backstaff` that lets you select the desired features, verifies any dependencies and builds and installs the `backstaff` binary.

We can download and run it in one go using the following command:
```
bash <(curl -s https://raw.githubusercontent.com/lars-frogner/Backstaff/main/setup_backstaff)
```

If you do not have the Rust toolchain installed on your system, you will be prompted to install it. Say yes and follow the instructions.

We can then move on to selecting features.

The feature system allows us to include only the functionality we need when building the binary. By not including unneeded features we can avoid unneccesary dependencies and reduce the compilation time and executable binary size. You can find descriptions of the available features at https://github.com/lars-frogner/Backstaff/blob/main/README.md#features.

For this tutorial we will need the `statistics`, `derivation`, `tracing` and `netcdf` features, so say yes to include each of these when prompted. If you think you may need additional features in the future you can include those now as well. You can always re-run the installation script at a later point to change the feature set.

When prompted about link time optimization you can say no, and you should say no to using a different version than the default. Finally, you may specify the path where the binary should be installed, or just accept the default path.

When the installation has finished you will be asked if you want to setup tab-completion for the `backstaff` command. Say yes and follow the instructions if you want that. But be aware that this may affect the responsiveness of constructing commands if you are installing with a lot of features.

If you installed `backstaff` to a directory in your `$PATH`, you should now be able to execute the binary:
```
backstaff -h
```
You should see a help text like this:
```
backstaff 0.3.0
Lars Frogner <lars.frogner@astro.uio.no>
A flexible tookit for working with Bifrost simulations

For documentation, see https://github.com/lars-frogner/Backstaff

USAGE:
    backstaff [OPTIONS] <SUBCOMMAND>

OPTIONS:
    -t, --timing
            Display elapsed time when done

    -P, --protected-file-types=<EXTENSIONS>...
            List of extensions for file types that never should be overwritten automatically
            (comma-separated) [default: idl,snap,aux]

    -N, --no-protected-files
            Allow any file type to be overwritten automatically

    -h, --help
            Print help information

    -V, --version
            Print version information

SUBCOMMANDS:
    snapshot       Specify input snapshot to perform further actions on
    create_mesh    Create a Bifrost mesh file
```

## Fetching demo snapshots

For this tutorial we will be working with a set of small 3D snapshots from a Bifrost simulation. Use the following commands to download the snapshots from the GitHub repository and extract them:
```
curl -LO https://github.com/lars-frogner/Backstaff/releases/latest/download/demo_snapshots.zip
unzip demo_snapshots.zip
```

In the extracted `demo_snapshots` folder you will find three sets of files representing a snapshot of the simulation at three points in time. Each snapshot has a `*.idl` parameter file containing all metadata for the snapshot in plaintext, as well as a binary `*.snap` and `*.aux` file containing the actual values for respectively the primary and auxiliary quantities. There is also a shared `*.mesh` text file that defines the simulation grid.

## Inspecting a snapshot

The first thing we will do is to find some statistics for the density and temperature fields in a snapshot:
```
backstaff snapshot demo_snapshots/en2431em_413.idl -v inspect -I=r,tg statistics
```
A `backstaff` command consists of a chain of subcommands, each with a set of arguments and/or options specified directly after the name of the subcommand. In the above command, the first subcommand is `snapshot`, which we have given a (required) argument specifying the path to the parameter file of the snapshot to read, followed by the option `-v` (verbose) to turn on status messages related to reading snapshots.

There are a number of different subcommands we could have specified after the `snapshot` subcommand depending on what we wanted to do with the snapshot. In this case we used the `inspect` subcommand with the option `-I=r,tg`, which says to include only the quantities `r` (density) and `tg` (temperature) in the inspection. We could also have used the long form `--included-quantities=r,tg`.

We finished off with the `statistics` subcommand without any options to let us inspect the default statistics for the quantities.

Here is the output:
```
Reading parameters from en2431em_413.idl
Reading grid from en2431em.mesh
Detected horizontally regular grid
Reading r from en2431em_413.snap
================================================================================
                 Statistics for r from snapshot 413 of en2431em                 
--------------------------------------------------------------------------------
                         For all r, all x, all y, all z                         
================================================================================
Number of values:    1000000
Minimum value:       1.8765084e-8 at ( 17.864,   9.944,  -9.409) [ 74,  41,   9]
Maximum value:       263.97241211 at (  2.744,  17.144,   2.445) [ 11,  71,  99]
Average value:       12.518044472
Reading tg from en2431em_413.aux
================================================================================
                Statistics for tg from snapshot 413 of en2431em                 
--------------------------------------------------------------------------------
                        For all tg, all x, all y, all z                         
================================================================================
Number of values:    1000000
Minimum value:       1998.7069092 at (  7.304,  10.184,  -1.540) [ 30,  42,  62]
Maximum value:       7.12432150e6 at ( 13.304,   3.224, -14.057) [ 55,  13,   0]
Average value:       7.52753062e5
--------------------------------------------------------------------------------
```

## Converting a snapshot to NetCDF

Since the native format used for Bifrost snapshots is not standard, 3D visualization tools like [ParaView](https://www.paraview.org/) and [VAPOR](https://www.vapor.ucar.edu/) don't know how to interpret the simulation data directly. The solution is to convert the data into a format that they do understand, and one such format is NetCDF (`*.nc`). Doing this with Backstaff (installed with the `netcdf` feature) is simple:
```
cd demo_snapshots
backstaff snapshot en2431em_413.idl write en2431em_413.nc
```
Here we made use of another one of `snapshot`'s subcommands, `write`. It needs an argument specifying the path of the output file that should be written. It looks at the file extension of the output file to determine which format to write in, which in this case is NetCDF due to the `.nc` extension.

Both `snapshot` and `write` can handle snapshot data in both NetCDF and native (`*.idl`) format, so we can read in and write out any combination of the two formats.

## Simple visualization with ParaView

### Loading the data

Let us load `en2431em_413.nc` into ParaView to have a look at it. In ParaView, click File -> Open, find and double-click on `en2431em_413.nc`. When asked which reader to use, double-click on the one labeled "NetCDF Reader".

Before the data is actually loaded there are a couple of tweaks we should make in the Properties panel (usually to the left of the render view). First, uncheck the Spherical Coordinates checkbox, as our data grid uses Cartesian coordinates.

Second, have a look at the Dimensions dropdown list just above the checkbox. Each list element represents a grid in the NetCDF file. The reason why the file contains multiple grids is that Bifrost uses a staggered grid, meaning that some quantities are defined in the centers of the grid cells while others are defined on the faces. ParaView assumes that all quantities loaded with a grid are defined at the same location, so the file needs a separate grid for each place in a grid cell quantities can be defined. For example, temperature (like all other auxiliary quantities) is a cell-centered quantity, so in order to get access to it we must load the data with the `(zm, ym, xm)` grid (`zm` is the array of cell-centered z-coordinates, etc.). If we hit Apply after selecting this grid, all the cell-centered quantities in the file are loaded. If we instead selected `(zmdn, ym, xm)`, all quantities defined in the middle of the bottom face of the grid cells, like `pz` and `bz`, would be loaded instead. But let us stick with the cell-centered quantities for now.

In the render view you should see the outline of the simulation box. We can add axes by checking the Axes Grid checkbox a bit down in the Properties panel. To have a look at the actual data, we can apply a Filter, which is ParaView's mechanism for transforming the data into various forms that can be visualised.

### 2D slice

We start by rendering a 2D slice through the simulation box. Click Filters -> Common -> Slice to create a Slice filter. The position and orientation of the slice plane can be adjusted either by pulling on the plane or its normal vector in the render view, or by using the relevant widgets in the Properties panel. Once happy, hit Apply and the slice will be rendered. Dissapointingly, it is just plain white. We can fix that by going to the Coloring section of the Properties panel and changing from Solid Color to `tg` in the dropdown list, which will color the slice according to the local temperature. You can change or adjust the color map in the Color Map Editor (make sure the Slice filter is selected in the Pipeline Browser). You may end up with something that looks like this:
![slice](/Backstaff/figures/tutorial_paraview_slice.jpg "2D slice in ParaView")

### Isosurfaces

Let us try something a bit more exciting and render some isosurfaces. Hide our slice plane by clicking on the eye next to Slice1 in the Pipeline Browser. Then, select `en2431em_413.nc`, as we will apply the new filter to the original data source, not the data outputted from the Slice filter. Click Filters -> Common -> Contour. In the properties of the newly created Contour filter, select `tg` in the Contour By list, and then add `1e4` and `2e6` in the Isosurfaces box below. Hit Apply and behold the isosurfaces for 10 000 K and 2 MK. By default, the coloring of the surfaces is based on the quantity we create the contours for, but that can easily be changed by selecting another quantity under Coloring in the Properties panel. Here is what you might get if you color the contours by resistive heating (`qjoule`):
![contours](/Backstaff/figures/tutorial_paraview_contours.jpg "Isosurfaces in ParaView")
