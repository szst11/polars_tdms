use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::File;
use std::io::{self, IoSliceMut, Read, Seek, SeekFrom};

const MAX_SPAN: u64 = 8 * 1024 * 1024;
const MAX_GAP: u64 = 64 * 1024;
const MAX_IOVECS: usize = 64;

struct Fragment {
    offset: u64,
    target: usize,
    len: usize,
}

pub(crate) struct ReadBuffers {
    pub data: Vec<u8>,
    pub bases: Vec<usize>,
    pub seeks: usize,
    pub reads: usize,
    pub bytes_read: u64,
}

fn read_vectors<R: Read>(
    reader: &mut R,
    buffers: &mut [IoSliceMut<'_>],
    calls: &mut usize,
) -> io::Result<()> {
    let mut remaining = buffers;
    while !remaining.is_empty() {
        *calls += 1;
        match reader.read_vectored(remaining) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => IoSliceMut::advance_slices(&mut remaining, n),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Benchmark-only raw reader. The caller supplies an independently parsed,
/// little-endian contiguous-channel layout; this is not a TDMS parser.
#[pyfunction]
pub fn _read_fragment_buffers(
    py: Python<'_>,
    path: &str,
    buffer_sizes: Vec<usize>,
    fragments: Vec<(u64, usize, usize, usize)>,
    strategy: &str,
) -> PyResult<(Vec<Py<PyBytes>>, usize, usize, u64)> {
    let result = read_buffers(path, &buffer_sizes, fragments, strategy)?;
    let buffers = result
        .bases
        .windows(2)
        .map(|range| PyBytes::new(py, &result.data[range[0]..range[1]]).unbind())
        .collect();
    Ok((buffers, result.seeks, result.reads, result.bytes_read))
}

pub(crate) fn read_buffers(
    path: &str,
    buffer_sizes: &[usize],
    fragments: Vec<(u64, usize, usize, usize)>,
    strategy: &str,
) -> PyResult<ReadBuffers> {
    if !matches!(strategy, "scalar" | "vectored" | "coalesced") {
        return Err(PyValueError::new_err("unknown fragment read strategy"));
    }
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut bases = vec![0usize];
    for size in buffer_sizes {
        let next = bases
            .last()
            .unwrap()
            .checked_add(*size)
            .ok_or_else(|| PyValueError::new_err("output buffer sizes overflow"))?;
        bases.push(next);
    }
    let total = *bases.last().unwrap();
    if total as u64 > file_len {
        return Err(PyValueError::new_err("output buffers exceed file size"));
    }
    let mut plan = Vec::with_capacity(fragments.len());
    for (offset, channel, target, len) in fragments {
        let size = buffer_sizes
            .get(channel)
            .ok_or_else(|| PyValueError::new_err("fragment channel index out of bounds"))?;
        if len == 0
            || target.checked_add(len).map_or(true, |end| end > *size)
            || offset
                .checked_add(len as u64)
                .map_or(true, |end| end > file_len)
        {
            return Err(PyValueError::new_err(
                "fragment range out of bounds or empty",
            ));
        }
        plan.push(Fragment {
            offset,
            target: bases[channel] + target,
            len,
        });
    }
    plan.sort_unstable_by_key(|f| f.target);
    let mut end = 0;
    for f in &plan {
        if f.target != end {
            return Err(PyValueError::new_err(
                "fragments must cover output buffers exactly once",
            ));
        }
        end += f.len;
    }
    if end != total {
        return Err(PyValueError::new_err(
            "fragments do not fill output buffers",
        ));
    }
    plan.sort_unstable_by_key(|f| f.offset);
    let mut disk_end = 0;
    for f in &plan {
        if f.offset < disk_end {
            return Err(PyValueError::new_err("overlapping file fragments"));
        }
        disk_end = f.offset + f.len as u64;
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(total)
        .map_err(|e| PyMemoryError::new_err(e.to_string()))?;
    output.resize(total, 0u8);
    let mut reads = 0;
    let mut seeks = 0;
    let mut bytes_read = 0;
    let mut scratch = Vec::new();
    let mut first = 0;
    while first < plan.len() {
        let start = plan[first].offset;
        let mut stop = first + 1;
        let mut span_end = start + plan[first].len as u64;
        let mut payload = plan[first].len as u64;
        if strategy != "scalar" {
            while stop < plan.len() {
                let f = &plan[stop];
                let next_end = f.offset + f.len as u64;
                let next_payload = payload + f.len as u64;
                let fits = if strategy == "vectored" {
                    f.offset == span_end && stop - first < MAX_IOVECS
                } else {
                    f.offset - span_end <= MAX_GAP
                        && next_end - start <= MAX_SPAN
                        && next_end - start - next_payload <= next_payload
                };
                if !fits {
                    break;
                }
                span_end = next_end;
                payload = next_payload;
                stop += 1;
            }
        }
        file.seek(SeekFrom::Start(start))?;
        seeks += 1;
        let run = &plan[first..stop];
        if strategy == "scalar" || (strategy == "coalesced" && run.len() == 1) {
            let f = &run[0];
            reads += 1;
            file.read_exact(&mut output[f.target..f.target + f.len])?;
        } else if strategy == "vectored" {
            // Split in destination order to borrow disjoint output slices safely,
            // then reorder the slices into their contiguous on-disk order.
            let mut destinations: Vec<_> = run.iter().collect();
            destinations.sort_unstable_by_key(|f| f.target);
            let mut tail = output.as_mut_slice();
            let mut cursor = 0;
            let mut slices = Vec::with_capacity(run.len());
            for f in destinations {
                let (_, rest) = tail.split_at_mut(f.target - cursor);
                let (data, rest) = rest.split_at_mut(f.len);
                slices.push((f.offset, IoSliceMut::new(data)));
                tail = rest;
                cursor = f.target + f.len;
            }
            slices.sort_unstable_by_key(|(offset, _)| *offset);
            let mut vectors: Vec<_> = slices.into_iter().map(|(_, slice)| slice).collect();
            read_vectors(&mut file, &mut vectors, &mut reads)?;
        } else {
            let len = usize::try_from(span_end - start)
                .map_err(|_| PyValueError::new_err("coalesced span exceeds address space"))?;
            scratch.resize(len, 0);
            reads += 1;
            file.read_exact(&mut scratch)?;
            for f in run {
                let source = (f.offset - start) as usize;
                output[f.target..f.target + f.len]
                    .copy_from_slice(&scratch[source..source + f.len]);
            }
        }
        bytes_read += span_end - start;
        first = stop;
    }
    Ok(ReadBuffers {
        data: output,
        bases,
        seeks,
        reads,
        bytes_read,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ShortReader {
        data: io::Cursor<Vec<u8>>,
        interrupt: bool,
    }

    impl Read for ShortReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.interrupt {
                self.interrupt = false;
                return Err(io::ErrorKind::Interrupted.into());
            }
            let len = buffer.len().min(1);
            self.data.read(&mut buffer[..len])
        }
    }

    #[test]
    fn vectored_short_reads_and_interruption() {
        let mut reader = ShortReader {
            data: io::Cursor::new(b"abcdef".to_vec()),
            interrupt: true,
        };
        let mut first = [0; 2];
        let mut second = [0; 4];
        let mut buffers = [IoSliceMut::new(&mut first), IoSliceMut::new(&mut second)];
        let mut calls = 0;
        read_vectors(&mut reader, &mut buffers, &mut calls).unwrap();
        assert_eq!(&first, b"ab");
        assert_eq!(&second, b"cdef");
        assert_eq!(calls, 7);
    }

    #[test]
    fn vectored_unexpected_eof() {
        let mut reader = io::Cursor::new(b"ab");
        let mut output = [0; 3];
        let mut buffers = [IoSliceMut::new(&mut output)];
        let mut calls = 0;
        let error = read_vectors(&mut reader, &mut buffers, &mut calls).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn vectored_single_read_across_buffers() {
        let mut reader = io::Cursor::new(b"abcde");
        let mut first = [0; 2];
        let mut second = [0; 3];
        let mut buffers = [IoSliceMut::new(&mut first), IoSliceMut::new(&mut second)];
        let mut calls = 0;
        read_vectors(&mut reader, &mut buffers, &mut calls).unwrap();
        assert_eq!(&first, b"ab");
        assert_eq!(&second, b"cde");
        assert_eq!(calls, 1);
    }
}
