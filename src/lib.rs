use numpy::ToPyArray;
use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObject;
use std::ops::Range;
use tdms_rs::api::reader::TdmsChannel;
use tdms_rs::{DataType, PropertyValue, TdmsError, TdmsFile};

fn tdms_err(e: TdmsError) -> PyErr {
    PyValueError::new_err(format!("TDMS error: {e}"))
}

fn channel_by_name<'a>(
    file: &'a TdmsFile,
    group: &str,
    channel: &str,
) -> PyResult<TdmsChannel<'a>> {
    let g = file
        .group(group)
        .ok_or_else(|| PyKeyError::new_err(format!("group {group:?} not found in file")))?;
    g.channel(channel)
        .ok_or_else(|| PyKeyError::new_err(format!("channel {channel:?} not found in group {group:?}")))
}

fn dtype_name(dt: &DataType) -> &'static str {
    match dt {
        DataType::I8 => "I8",
        DataType::I16 => "I16",
        DataType::I32 => "I32",
        DataType::I64 => "I64",
        DataType::U8 => "U8",
        DataType::U16 => "U16",
        DataType::U32 => "U32",
        DataType::U64 => "U64",
        DataType::Float => "Float",
        DataType::Double => "Double",
        DataType::String => "String",
        DataType::Boolean => "Boolean",
        DataType::TimeStamp => "TimeStamp",
    }
}

/// Serialize a TDMS property value into a Python value.
///
/// TimeStamp properties are returned as a `(seconds, fraction)` tuple where
/// `fraction` counts 2^64-ths of a second since the TDMS/LabVIEW epoch
/// (1904-01-01). The high-level Python API converts these to `datetime`.
fn property_to_py<'py>(
    py: Python<'py>,
    v: &PropertyValue,
) -> PyResult<Py<PyAny>> {
    Ok(match v {
        PropertyValue::I8(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::I16(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::I32(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::I64(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::U8(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::U16(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::U32(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::U64(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::Float(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::Double(x) => x.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::Boolean(b) => {
            let obj = (*b).into_pyobject(py)?;
            obj.to_owned().into_any().unbind()
        }
        PropertyValue::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        PropertyValue::TimeStamp((secs, frac)) => {
            (*secs, *frac as i64).into_pyobject(py)?.into_any().unbind()
        }
    })
}

fn properties_to_dict<'a>(
    py: Python<'_>,
    iter: impl Iterator<Item = (&'a str, &'a PropertyValue)>,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in iter {
        d.set_item(k, property_to_py(py, v)?)?;
    }
    Ok(d.unbind())
}

/// Read a range of POD (numeric / boolean) samples from a channel into a NumPy array.
fn read_channel_into_numpy(
    py: Python<'_>,
    channel: &TdmsChannel<'_>,
    range: Range<usize>,
) -> PyResult<Py<PyAny>> {
    macro_rules! read_typed {
        ($ty:ty) => {{
            let len = range.end - range.start;
            let mut buf: Vec<$ty> = vec![<$ty>::default(); len];
            channel.read(range, &mut buf).map_err(tdms_err)?;
            buf.to_pyarray(py).into_any().unbind()
        }};
    }
    Ok(match channel.dtype() {
        DataType::I8 => read_typed!(i8),
        DataType::I16 => read_typed!(i16),
        DataType::I32 => read_typed!(i32),
        DataType::I64 => read_typed!(i64),
        DataType::U8 => read_typed!(u8),
        DataType::U16 => read_typed!(u16),
        DataType::U32 => read_typed!(u32),
        DataType::U64 => read_typed!(u64),
        DataType::Float => read_typed!(f32),
        DataType::Double => read_typed!(f64),
        DataType::Boolean => read_typed!(bool),
        DataType::String | DataType::TimeStamp => {
            return Err(PyNotImplementedError::new_err(format!(
                "channel data type {dt:?} cannot be read as numeric samples (metadata only)",
                dt = channel.dtype()
            )))
        }
    })
}

#[pyclass(module = "polars_tdms._core")]
struct TdmsHandle {
    file: TdmsFile,
}

#[pymethods]
impl TdmsHandle {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = TdmsFile::open(path).map_err(tdms_err)?;
        Ok(Self { file })
    }

    /// Names of all groups in the file.
    fn groups(&self) -> Vec<String> {
        self.file.groups().map(|g| g.name().to_string()).collect()
    }

    /// File-level properties as a dict.
    fn file_properties(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        properties_to_dict(py, self.file.properties())
    }

    /// Group-level properties as a dict.
    fn group_properties(&self, py: Python<'_>, group: &str) -> PyResult<Py<PyDict>> {
        let g = self
            .file
            .group(group)
            .ok_or_else(|| PyKeyError::new_err(format!("group {group:?} not found in file")))?;
        properties_to_dict(py, g.properties())
    }

    /// Info for a single channel: {"name", "dtype", "len", "properties"}.
    fn channel_info(&self, py: Python<'_>, group: &str, channel: &str) -> PyResult<Py<PyDict>> {
        let c = channel_by_name(&self.file, group, channel)?;
        let info = PyDict::new(py);
        info.set_item("name", c.name())?;
        info.set_item("dtype", dtype_name(&c.dtype()))?;
        info.set_item("len", c.len())?;
        info.set_item("properties", properties_to_dict(py, c.properties())?)?;
        Ok(info.unbind())
    }

    /// Full metadata for one group:
    /// {"name", "properties", "channels": [channel_info, ...]}.
    fn group_metadata(&self, py: Python<'_>, group: &str) -> PyResult<Py<PyDict>> {
        let g = self
            .file
            .group(group)
            .ok_or_else(|| PyKeyError::new_err(format!("group {group:?} not found in file")))?;
        let meta = PyDict::new(py);
        meta.set_item("name", g.name())?;
        meta.set_item("properties", properties_to_dict(py, g.properties())?)?;
        let channels = PyList::empty(py);
        for c in g.channels() {
            let info = PyDict::new(py);
            info.set_item("name", c.name())?;
            info.set_item("dtype", dtype_name(&c.dtype()))?;
            info.set_item("len", c.len())?;
            info.set_item("properties", properties_to_dict(py, c.properties())?)?;
            channels.append(info.unbind())?;
        }
        meta.set_item("channels", channels.unbind())?;
        Ok(meta.unbind())
    }

    /// Read a slice of channels samples. Returns a NumPy array with the
    /// channel's native dtype. Bounds are `start..end` (end exclusive).
    fn read_channel_range(
        &self,
        py: Python<'_>,
        group: &str,
        channel: &str,
        start: usize,
        end: usize,
    ) -> PyResult<Py<PyAny>> {
        if end < start {
            return Err(PyValueError::new_err(
                "read_channel_range: end must be >= start",
            ));
        }
        let c = channel_by_name(&self.file, group, channel)?;
        read_channel_into_numpy(py, &c, start..end)
    }

    /// Read the full channel. Returns a NumPy array.
    fn read_channel(
        &self,
        py: Python<'_>,
        group: &str,
        channel: &str,
    ) -> PyResult<Py<PyAny>> {
        let c = channel_by_name(&self.file, group, channel)?;
        read_channel_into_numpy(py, &c, 0..c.len())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TdmsHandle>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}