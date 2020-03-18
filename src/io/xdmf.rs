//! Representation of data in XDMF format.

use crate::geometry::{
    Dim3::{X, Y, Z},
    In3D, Point3, Vec3,
};
use std::borrow::Cow;
use std::fmt::{self, Display, Formatter};
use std::io::Write;
use xml::{
    writer::{events::XmlEvent, EventWriter},
    EmitterConfig,
};

#[derive(Clone, Copy, Debug)]
pub enum Version {
    V20,
    V22,
    V30,
}

impl Display for Version {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let string = match self {
            Self::V20 => "2.0",
            Self::V22 => "2.2",
            Self::V30 => "3.0",
        };
        write!(f, "{}", string)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    XML,
    HDF,
    Binary,
}

impl Display for Format {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let string = match self {
            Self::XML => "XML",
            Self::HDF => "HDF",
            Self::Binary => "Binary",
        };
        write!(f, "{}", string)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NumberType {
    Float,
}

impl Display for NumberType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let string = match self {
            Self::Float => "Float",
        };
        write!(f, "{}", string)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Precision {
    Single,
    Double,
}

impl Display for Precision {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let string = match self {
            Self::Single => "4",
            Self::Double => "8",
        };
        write!(f, "{}", string)
    }
}

#[derive(Clone, Debug)]
pub struct Dimensions {
    rank: usize,
    dimensions: Vec<usize>,
}

impl Dimensions {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let rank = dimensions.len();
        assert!(rank > 0);
        Self { rank, dimensions }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl Display for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let string = self
            .dimensions
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        write!(f, "{}", string)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RectMeshType {
    Regular,
    General,
}

#[derive(Clone, Debug)]
pub struct Xdmf {
    version: Version,
    domains: Vec<Domain>,
}

impl Xdmf {
    pub fn new(domain: Domain) -> Self {
        Self {
            version: Version::V30,
            domains: vec![domain],
        }
    }

    pub fn add_domain(&mut self, domain: Domain) -> &mut Self {
        self.domains.push(domain);
        self
    }

    pub fn create_domain(&mut self, grid: Grid) -> &mut Domain {
        self.domains.push(Domain::new(grid));
        self.domains.last_mut().unwrap()
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), xml::writer::Error> {
        let mut xml_writer = EmitterConfig::new()
            .perform_indent(true)
            .create_writer(writer);
        xml_writer.write(
            XmlEvent::start_element("Xdmf")
                .ns("xi", "http://www.w3.org/2001/XInclude")
                .attr("VERSION", &self.version.to_string()),
        )?;
        for domain in &self.domains {
            domain.write(&mut xml_writer)?;
        }
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Domain {
    name: Option<Cow<'static, str>>,
    grids: Vec<Grid>,
}

impl Domain {
    pub fn new(grid: Grid) -> Self {
        Self {
            name: None,
            grids: vec![grid],
        }
    }

    pub fn with_name<S: Into<Cow<'static, str>>>(&mut self, name: S) -> &mut Self {
        self.name = Some(name.into());
        self
    }

    pub fn add_grid(&mut self, grid: Grid) -> &mut Self {
        assert!(self.grids.first().unwrap().has_name());
        assert!(grid.has_name());
        self.grids.push(grid);
        self
    }

    pub fn create_uniform_grid<S: Into<Cow<'static, str>>>(
        &mut self,
        name: S,
        topology: Topology,
        geometry: Geometry,
    ) -> &mut UniformGrid {
        self.add_grid(Grid::Uniform(
            UniformGrid::new(topology, geometry).with_name(name),
        ));
        if let Grid::Uniform(ref mut grid) = self.grids.last_mut().unwrap() {
            grid
        } else {
            unreachable!()
        }
    }

    pub fn create_grid_collection<S: Into<Cow<'static, str>>>(
        &mut self,
        name: S,
        grids: Vec<UniformGrid>,
    ) -> &mut GridCollection {
        self.add_grid(Grid::Collection(GridCollection::new(grids).with_name(name)));
        if let Grid::Collection(ref mut collection) = self.grids.last_mut().unwrap() {
            collection
        } else {
            unreachable!()
        }
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        xml_writer.write(XmlEvent::start_element("Domain"))?;
        for grid in &self.grids {
            grid.write(xml_writer)?
        }
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Grid {
    Uniform(UniformGrid),
    Collection(GridCollection),
}

impl Grid {
    fn has_name(&self) -> bool {
        match self {
            Self::Uniform(grid) => grid.has_name(),
            Self::Collection(grid) => grid.has_name(),
        }
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        match self {
            Self::Uniform(grid) => grid.write(xml_writer)?,
            Self::Collection(grid) => grid.write(xml_writer)?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct UniformGrid {
    name: Option<Cow<'static, str>>,
    times: Vec<Time>,
    topology: Topology,
    geometry: Geometry,
    attributes: Vec<Attribute>,
    attribute_names: Vec<String>,
}

impl UniformGrid {
    pub fn new(topology: Topology, geometry: Geometry) -> Self {
        Self {
            name: None,
            times: Vec::new(),
            topology,
            geometry,
            attributes: Vec::new(),
            attribute_names: Vec::new(),
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.map(|n| n.as_ref())
    }

    pub fn with_name<S: Into<Cow<'static, str>>>(self, name: S) -> Self {
        self.set_name(name);
        self
    }

    pub fn set_name<S: Into<Cow<'static, str>>>(&mut self, name: S) -> &mut Self {
        self.name = Some(name.into());
        self
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    pub fn add_time(&mut self, time: Time) -> &mut Self {
        self.times.push(time);
        self
    }

    pub fn add_attribute(&mut self, attribute: Attribute) -> &mut Self {
        if let Some(name) = attribute.name() {
            self.attribute_names.push(name.to_string());
        } else {
            assert!(self.attributes.is_empty());
        }
        self.attributes.push(attribute);
        self
    }

    fn has_same_attributes(&self, other: &Self) -> bool {
        self.attribute_names
            .iter()
            .zip(&other.attribute_names)
            .filter(|(mine, others)| mine != others)
            .count()
            == 0
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        let mut event = XmlEvent::start_element("Grid").attr("GridType", "Uniform");
        if let Some(name) = self.name {
            event = event.attr("Name", &name);
        }
        xml_writer.write(event)?;
        self.topology.write(xml_writer)?;
        self.geometry.write(xml_writer)?;
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GridCollection {
    name: Option<Cow<'static, str>>,
    grids: Vec<UniformGrid>,
}

impl GridCollection {
    pub fn new(grids: Vec<UniformGrid>) -> Self {
        assert!(!grids.is_empty());
        let first_grid = grids.first().unwrap();
        if grids.len() > 1 {
            assert!(first_grid.has_name());
        }
        for other_grid in grids.iter().skip(1) {
            assert!(other_grid.has_name());
            assert!(other_grid.has_same_attributes(first_grid));
        }
        Self { name: None, grids }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.map(|n| n.as_ref())
    }

    pub fn with_name<S: Into<Cow<'static, str>>>(self, name: S) -> Self {
        self.set_name(name);
        self
    }

    pub fn set_name<S: Into<Cow<'static, str>>>(&mut self, name: S) -> &mut Self {
        self.name = Some(name.into());
        self
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    pub fn add_grid(&mut self, grid: UniformGrid) -> &mut Self {
        let existing_grid = self.grids.first().unwrap();
        assert!(existing_grid.has_name());
        assert!(grid.has_name());
        assert!(grid.has_same_attributes(existing_grid));
        self.grids.push(grid);
        self
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        let mut event = XmlEvent::start_element("Grid").attr("GridType", "Collection");
        if let Some(name) = self.name {
            event = event.attr("Name", &name);
        }
        xml_writer.write(event)?;
        for grid in &self.grids {
            grid.write(xml_writer)?;
        }
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Time {
    value: Cow<'static, str>,
}

impl Time {
    pub fn new<S: Into<Cow<'static, str>>>(value: S) -> Self {
        Self {
            value: value.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Attribute {
    name: Option<Cow<'static, str>>,
    data_items: Vec<DataItem>,
}

impl Attribute {
    pub fn new() -> Self {
        Self {
            name: None,
            data_items: Vec::new(),
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.map(|n| n.as_ref())
    }

    pub fn with_name<S: Into<Cow<'static, str>>>(self, name: S) -> Self {
        self.set_name(name);
        self
    }

    pub fn set_name<S: Into<Cow<'static, str>>>(&mut self, name: S) -> &mut Self {
        self.name = Some(name.into());
        self
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }
}

#[derive(Clone, Debug)]
pub enum Topology {
    RectMesh3D(RectMesh3DTopology),
}

impl Topology {
    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        match self {
            Self::RectMesh3D(topology) => topology.write(xml_writer)?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RectMesh3DTopology {
    mesh_type: RectMeshType,
    dimensions: Dimensions,
}

impl RectMesh3DTopology {
    pub fn new(mesh_type: RectMeshType, shape: In3D<usize>) -> Self {
        Self {
            mesh_type,
            dimensions: Dimensions::new(vec![shape[X], shape[Y], shape[Z]]),
        }
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        xml_writer.write(
            XmlEvent::start_element("Topology")
                .attr(
                    "TopologyType",
                    match self.mesh_type {
                        RectMeshType::Regular => "3DCoRectMesh",
                        RectMeshType::General => "3DRectMesh",
                    },
                )
                .attr("Dimensions", &self.dimensions.to_string()),
        )?;
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum DataItem {
    Uniform(UniformDataItem),
    Collection(DataItemCollection),
}

impl DataItem {
    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        match self {
            Self::Uniform(data_item) => data_item.write(xml_writer)?,
            Self::Collection(data_item) => data_item.write(xml_writer)?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct UniformDataItem {
    format: Format,
    number_type: NumberType,
    precision: Precision,
    dimensions: Dimensions,
    data: Cow<'static, str>,
}

impl UniformDataItem {
    pub fn new<S: Into<Cow<'static, str>>>(
        format: Format,
        precision: Precision,
        dimensions: Dimensions,
        data: S,
    ) -> Self {
        Self {
            format,
            number_type: NumberType::Float,
            precision,
            dimensions,
            data: data.into(),
        }
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        xml_writer.write(
            XmlEvent::start_element("DataItem")
                .attr("ItemType", "Uniform")
                .attr("Format", &self.format.to_string())
                .attr("NumberType", &self.number_type.to_string())
                .attr("Precision", &self.precision.to_string())
                .attr("Rank", &self.dimensions.rank().to_string())
                .attr("Dimensions", &self.dimensions.to_string()),
        )?;
        xml_writer.write(XmlEvent::characters(&self.data))?;
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DataItemCollection {
    name: Cow<'static, str>,
    items: Vec<DataItem>,
}

impl DataItemCollection {
    pub fn new<S: Into<Cow<'static, str>>>(name: S, items: Vec<DataItem>) -> Self {
        assert!(!items.is_empty());
        Self {
            name: name.into(),
            items,
        }
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        xml_writer.write(XmlEvent::start_element("DataItem").attr("ItemType", "Collection"))?;
        for item in &self.items {
            item.write(xml_writer)?;
        }
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Geometry {
    OriginDxDyDz(OriginDxDyDzGeometry),
}

impl Geometry {
    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        match self {
            Self::OriginDxDyDz(geometry) => geometry.write(xml_writer)?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct OriginDxDyDzGeometry {
    origin: Point3<f32>,
    dxdydz: Vec3<f32>,
}

impl OriginDxDyDzGeometry {
    pub fn new(origin: Point3<f32>, dxdydz: Vec3<f32>) -> Self {
        Self { origin, dxdydz }
    }

    fn write<W: Write>(&self, xml_writer: &mut EventWriter<W>) -> Result<(), xml::writer::Error> {
        xml_writer
            .write(XmlEvent::start_element("Geometry").attr("GeometryType", "ORIGIN_DXDYDZ"))?;
        xml_writer.write(XmlEvent::comment("Origin"))?;
        UniformDataItem::new(
            Format::XML,
            Precision::Single,
            Dimensions::new(vec![3]),
            format!("{} {} {}", self.origin[X], self.origin[Y], self.origin[Z]),
        )
        .write(xml_writer)?;
        xml_writer.write(XmlEvent::comment("DxDyDz"))?;
        UniformDataItem::new(
            Format::XML,
            Precision::Single,
            Dimensions::new(vec![3]),
            format!("{} {} {}", self.dxdydz[X], self.dxdydz[Y], self.dxdydz[Z]),
        )
        .write(xml_writer)?;
        xml_writer.write(XmlEvent::end_element())?;
        Ok(())
    }
}
