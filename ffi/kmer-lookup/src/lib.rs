pub mod db_format;

use std::{
    convert::TryFrom,
    ffi::CString,
    fs::File,
    io::{self, BufReader},
    iter::Sum,
    ops::Not,
    path::PathBuf,
};

use fftw::{
    array::AlignedVec,
    plan::{C2CPlan, C2CPlan64},
    types::{Flag, Sign},
};
use fnv::FnvHashMap;
use itertools::Itertools;
use num_complex::Complex;
use num_traits::{float::FloatCore, Float, NumAssignRef, NumCast, NumRef, RefNum};
use smallvec::SmallVec;

#[derive(Debug)]
pub struct KmerLookup {
    kmer_len: u16,
    kmer_step: usize,
    max_sequence_distance: Option<Distance>,
    max_gc_diff: Option<Option<f64>>,
    min_kmers: usize,
    max_kmer_merge_distance: Option<usize>,
    threads: Option<u16>,
    max_reactivity: f64,
    min_complexity: f64,
    max_matches_every_nt: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Distance {
    Integral(usize),
    Fractional(f64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KmerLookupResult {
    pub db_id: CString,
    pub db: Vec<KmerLookupDbResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KmerLookupDbResult {
    pub db: [usize; 2],
    pub query: [usize; 2],
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("FFTW error: {0}")]
    Fftw(#[from] fftw::error::Error),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("DB reader error: {0}")]
    Reader(#[from] db_format::ReaderError),

    #[error("DB reader entry error: {0}")]
    ReaderEntry(#[from] db_format::EntryError),
}

impl From<db_format::Error> for Error {
    fn from(err: db_format::Error) -> Self {
        use db_format::Error as DbError;

        match err {
            DbError::Io(err) => Self::Io(err),
            DbError::Reader(err) => Self::Reader(err),
        }
    }
}

impl From<db_format::EntryIoError> for Error {
    fn from(err: db_format::EntryIoError) -> Self {
        use db_format::EntryIoError as DbError;
        match err {
            DbError::Entry(err) => Self::ReaderEntry(err),
            DbError::Io(err) => Self::Io(err),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KmerLookupBuilder<KmerLen> {
    kmer_len: KmerLen,
    kmer_step: usize,
    max_sequence_distance: Option<Distance>,
    max_gc_diff: Option<Option<f64>>,
    min_kmers: usize,
    max_kmer_merge_distance: Option<usize>,
    threads: Option<u16>,
    max_reactivity: f64,
    min_complexity: f64,
    max_matches_every_nt: usize,
}

impl Default for KmerLookupBuilder<()> {
    fn default() -> Self {
        Self {
            kmer_len: (),
            kmer_step: 1,
            max_sequence_distance: None,
            max_gc_diff: None,
            min_kmers: 2,
            max_kmer_merge_distance: Some(10),
            threads: None,
            max_reactivity: f64::INFINITY,
            min_complexity: 0.,
            max_matches_every_nt: usize::MAX,
        }
    }
}

impl<KmerLen> KmerLookupBuilder<KmerLen> {
    pub fn kmer_len(self, len: u16) -> KmerLookupBuilder<u16> {
        let Self {
            kmer_len: _,
            kmer_step,
            max_sequence_distance,
            max_gc_diff,
            min_kmers,
            max_kmer_merge_distance,
            threads,
            max_reactivity,
            min_complexity,
            max_matches_every_nt,
        } = self;

        KmerLookupBuilder {
            kmer_len: len,
            kmer_step,
            max_sequence_distance,
            max_gc_diff,
            min_kmers,
            max_kmer_merge_distance,
            threads,
            max_reactivity,
            min_complexity,
            max_matches_every_nt,
        }
    }

    pub fn max_sequence_distance(&mut self, distance: Option<Distance>) -> &mut Self {
        self.max_sequence_distance = distance;
        self
    }

    pub fn max_gc_diff(&mut self, diff: Option<Option<f64>>) -> &mut Self {
        self.max_gc_diff = diff;
        self
    }

    pub fn max_kmer_merge_distance(&mut self, distance: Option<usize>) -> &mut Self {
        self.max_kmer_merge_distance = distance;
        self
    }

    pub fn max_matches_every_nt(&mut self, value: usize) -> &mut Self {
        self.max_matches_every_nt = value;
        self
    }

    pub fn threads(&mut self, num_threads: Option<u16>) -> &mut Self {
        self.threads = num_threads;
        self
    }

    pub fn max_reactivity(&mut self, value: f64) -> &mut Self {
        self.max_reactivity = value;
        self
    }

    pub fn min_complexity(&mut self, value: f64) -> &mut Self {
        self.min_complexity = value;
        self
    }

    pub fn min_kmers(&mut self, value: usize) -> &mut Self {
        self.min_kmers = value;
        self
    }

    pub fn kmer_step(&mut self, step: usize) -> Result<&mut Self, KmerLookupBuilderError> {
        if step == 0 {
            return Err(KmerLookupBuilderError::InvalidKmerStep);
        }

        self.kmer_step = step;
        Ok(self)
    }
}

impl KmerLookupBuilder<u16> {
    pub fn build(self) -> KmerLookup {
        KmerLookup::from(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KmerLookupBuilderError {
    #[error("reactivity data and sequence have different lenghts.")]
    InconsistentLength,

    #[error("invalid kmer step, it must be greater than 0.")]
    InvalidKmerStep,
}

pub type ReactivityAndSequence = (Vec<f64>, Vec<u8>);

impl From<KmerLookupBuilder<u16>> for KmerLookup {
    fn from(builder: KmerLookupBuilder<u16>) -> Self {
        let KmerLookupBuilder {
            kmer_len,
            kmer_step,
            max_sequence_distance,
            max_gc_diff,
            min_kmers,
            max_kmer_merge_distance,
            threads,
            max_reactivity,
            min_complexity,
            max_matches_every_nt,
        } = builder;

        Self {
            kmer_len,
            kmer_step,
            max_sequence_distance,
            max_gc_diff,
            min_kmers,
            max_kmer_merge_distance,
            threads,
            max_reactivity,
            min_complexity,
            max_matches_every_nt,
        }
    }
}

#[derive(Debug)]
struct DbData {
    sequence: Vec<u8>,
    reactivity: Vec<f64>,
    transformed_reactivity: AlignedVec<Complex<f64>>,
}

impl DbData {
    fn new(
        sequence: Vec<u8>,
        mut reactivity: Vec<f64>,
        kmer_lookup: &KmerLookup,
    ) -> Result<Self, fftw::error::Error> {
        reactivity.iter_mut().for_each(|x| {
            *x = if x.is_nan() {
                -999.
            } else {
                x.min(kmer_lookup.max_reactivity)
            }
        });
        let transformed_reactivity = transform_db(&reactivity)?;

        Ok(Self {
            sequence,
            reactivity,
            transformed_reactivity,
        })
    }
}

#[derive(Debug)]
enum WorkMessage {
    Entry(db_format::Entry),
    Stop,
}

impl WorkMessage {
    fn into_entry(self) -> Option<db_format::Entry> {
        match self {
            Self::Entry(entry) => Some(entry),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct KmerLookupRunData {
    db_path: PathBuf,
    query: Vec<f64>,
    query_sequence: Vec<u8>,
}

impl KmerLookupRunData {
    pub fn new<Q, QS>(db_path: impl Into<PathBuf>, query: Q, query_sequence: QS) -> Option<Self>
    where
        Q: AsRef<[f64]> + Into<Vec<f64>>,
        QS: AsRef<[u8]> + Into<Vec<u8>>,
    {
        if query.as_ref().len() != query_sequence.as_ref().len() {
            return None;
        }

        let db_path = db_path.into();
        let query = query.into();
        let query_sequence = query_sequence.into();

        Some(Self {
            db_path,
            query,
            query_sequence,
        })
    }

    fn saturate_query(&mut self, max_reactivity: f64) {
        self.query
            .iter_mut()
            .filter(|x| x.is_nan().not())
            .for_each(|x| *x = x.min(max_reactivity));
    }
}

impl KmerLookup {
    pub fn builder() -> KmerLookupBuilder<()> {
        Default::default()
    }

    pub fn run(&self, mut data: KmerLookupRunData) -> Result<Vec<KmerLookupResult>, Error> {
        data.saturate_query(self.max_reactivity);

        let threads = self
            .threads
            .map(|threads| threads.into())
            .unwrap_or_else(num_cpus::get);
        if threads <= 1 {
            self.run_single_thread(&data)
        } else {
            self.run_multi_thread(&data, threads)
        }
    }

    fn run_single_thread(&self, data: &KmerLookupRunData) -> Result<Vec<KmerLookupResult>, Error> {
        let db_reader = BufReader::new(File::open(&data.db_path)?);
        let mut db_reader = db_format::Reader::new(db_reader)?;

        let mut results = Vec::new();
        for entry in db_reader.entries() {
            self.handle_entry(data, entry?, &mut results)?;
        }

        Ok(results)
    }

    fn handle_entry(
        &self,
        data: &KmerLookupRunData,
        entry: db_format::Entry,
        results: &mut Vec<KmerLookupResult>,
    ) -> Result<(), Error> {
        let db_format::Entry {
            id,
            sequence,
            reactivity,
        } = entry;

        let db_data = DbData::new(sequence, reactivity, self)?;
        let matching_kmers = self.get_matching_kmers(data, &db_data)?;
        let grouped = self.group_matching_kmers(&matching_kmers);

        if grouped.is_empty().not() {
            results.push(KmerLookupResult {
                db_id: id,
                db: grouped,
            });
        }
        Ok(())
    }

    fn run_multi_thread(
        &self,
        data: &KmerLookupRunData,
        threads: usize,
    ) -> Result<Vec<KmerLookupResult>, Error> {
        use crossbeam::{
            channel::{self, TrySendError},
            scope,
        };

        debug_assert!(threads > 1);
        let workers = threads - 1;
        scope(|scope| {
            let (sender, receiver) = channel::bounded(workers * 3);

            let join_handles: Vec<_> = (0..workers)
                .map(|_| {
                    let receiver = receiver.clone();
                    scope.spawn(|_| self.run_worker(data, receiver))
                })
                .collect();

            let db_reader = BufReader::new(File::open(&data.db_path)?);
            let mut db_reader = db_format::Reader::new(db_reader)?;

            let mut results = Vec::new();
            for entry in db_reader.entries() {
                if let Err(err) = sender.try_send(WorkMessage::Entry(entry?)) {
                    match err {
                        TrySendError::Full(message) => {
                            let entry = message.into_entry().unwrap();
                            self.handle_entry(data, entry, &mut results)?;
                        }
                        TrySendError::Disconnected(_) => {
                            panic!("all queue receiver are disconnected unexpectedly")
                        }
                    }
                }
            }

            for _ in 0..workers {
                sender
                    .send(WorkMessage::Stop)
                    .expect("all queue receiver are disconnected unexpectedly");
            }
            for join_handle in join_handles {
                let result = join_handle.join().unwrap();
                results.extend(result?);
            }

            Ok(results)
        })
        .unwrap()
    }

    fn run_worker(
        &self,
        data: &KmerLookupRunData,
        receiver: crossbeam::channel::Receiver<WorkMessage>,
    ) -> Result<Vec<KmerLookupResult>, Error> {
        let mut results = Vec::new();
        for message in receiver {
            match message {
                WorkMessage::Entry(entry) => self.handle_entry(data, entry, &mut results)?,
                WorkMessage::Stop => break,
            }
        }

        Ok(results)
    }

    fn group_matching_kmers(&self, matching_kmers: &[[usize; 2]]) -> Vec<KmerLookupDbResult> {
        let mut diagonals = FnvHashMap::<_, SmallVec<[_; 8]>>::default();
        for pair in matching_kmers {
            let &[i, j] = pair;
            let diagonal = isize::try_from(i).unwrap() - isize::try_from(j).unwrap();
            diagonals.entry(diagonal).or_default().push(pair);
        }

        let kmer_len: usize = self.kmer_len.into();
        let max_distance = self.max_kmer_merge_distance.unwrap_or(0) + kmer_len;
        let mut groups = Vec::new();
        let mut add_group = |first: &[usize; 2], last: &[usize; 2]| {
            let group = KmerLookupDbResult {
                query: [first[0], last[0] + kmer_len - 1],
                db: [first[1], last[1] + kmer_len - 1],
            };
            groups.push(group);
        };
        let min_kmers = self
            .max_kmer_merge_distance
            .is_some()
            .then(|| self.min_kmers)
            .unwrap_or(1);

        for diagonal in diagonals.values_mut() {
            diagonal.sort_unstable();
            let mut iter = diagonal.iter().copied();
            // Each diagonal contains at least one element
            let mut first = iter.next().unwrap();
            let mut last = first;
            let mut group_kmers = 1;
            for pair in iter {
                if pair[0] > last[0] + max_distance {
                    if group_kmers >= min_kmers {
                        add_group(first, last);
                    }
                    group_kmers = 1;
                    first = pair;
                } else {
                    group_kmers += 1;
                }
                last = pair;
            }

            if group_kmers >= self.min_kmers {
                add_group(first, last);
            }
        }

        groups
    }

    fn get_matching_kmers(
        &self,
        data: &KmerLookupRunData,
        db_data: &DbData,
    ) -> Result<Vec<[usize; 2]>, Error> {
        let max_sequence_distance =
            self.max_sequence_distance
                .map(|max_sequence_distance| match max_sequence_distance {
                    Distance::Integral(dist) => dist,
                    Distance::Fractional(frac) => (frac * self.kmer_len as f64) as usize,
                });
        let max_gc_diff = self.max_gc_diff.map(|max_gc_diff| {
            max_gc_diff.unwrap_or_else(|| 0.2676 * (self.kmer_len as f64 * -0.053).exp())
        });

        let mut mass = Mass::new(db_data.reactivity.len())?;

        let matches = data
            .query
            .windows(self.kmer_len.into())
            .zip(data.query_sequence.windows(self.kmer_len.into()))
            .enumerate()
            .step_by(self.kmer_step)
            .filter(|(_, (kmer, _))| kmer.iter().any(|x| x.is_nan()).not())
            .filter(|(_, (kmer, _))| gini_index(kmer) >= self.min_complexity)
            .map(|(kmer_index, (kmer, kmer_sequence))| {
                let mut complex_distances =
                    mass.run(&db_data.reactivity, &db_data.transformed_reactivity, kmer)?;
                for dist in &mut *complex_distances {
                    *dist = Complex::new(dist.norm(), 0.);
                }

                let (mean_dist, stddev_dist) = mean_stddev(&complex_distances, 1);
                let max_distance = mean_dist.re - stddev_dist.re * 3.;

                let kmer_gc_fraction = max_gc_diff.is_some().then(|| {
                    let gc_count = kmer_sequence
                        .iter()
                        .filter(|&&c| c == b'C' || c == b'G')
                        .count();
                    gc_count as f64 / self.kmer_len as f64
                });

                let mut kmer_data = complex_distances
                    .into_iter()
                    .enumerate()
                    .map(|(index, dist)| (index, dist.re))
                    .zip(db_data.sequence.windows(self.kmer_len.into()))
                    .filter(move |&((_, dist), _)| dist <= max_distance)
                    .map(|((index, _), db_sequence)| (index, db_sequence))
                    .collect::<Vec<_>>();

                if kmer_data.is_empty().not()
                    && db_data.reactivity.len() / kmer_data.len() < self.max_matches_every_nt
                {
                    kmer_data.clear();
                }

                Ok::<_, fftw::error::Error>(
                    kmer_data
                        .into_iter()
                        .filter(move |(_, db_sequence)| {
                            max_sequence_distance
                                .map(move |max_sequence_distance| {
                                    {
                                        hamming_distance(kmer_sequence, db_sequence)
                                            <= max_sequence_distance
                                    }
                                })
                                .unwrap_or(true)
                        })
                        .filter(move |(_, db_sequence)| {
                            max_gc_diff
                                .map(move |max_gc_diff| {
                                    let gc_count = db_sequence
                                        .iter()
                                        .filter(|&&c| c == b'C' || c == b'G')
                                        .count();
                                    let gc_fraction = gc_count as f64 / self.kmer_len as f64;
                                    (gc_fraction - kmer_gc_fraction.unwrap()).abs() <= max_gc_diff
                                })
                                .unwrap_or(true)
                        })
                        .map(move |(sequence_index, _)| [kmer_index, sequence_index]),
                )
            })
            .flatten_ok()
            .collect::<Result<Vec<_>, _>>()?;

        Ok(matches)
    }
}

struct Mass {
    fw_plan: C2CPlan64,
    bw_plan: C2CPlan64,
    aligned_query: AlignedVec<Complex<f64>>,
    query_transform: AlignedVec<Complex<f64>>,
    product: AlignedVec<Complex<f64>>,
    product_inverse: AlignedVec<Complex<f64>>,
}

impl Mass {
    fn new(size: usize) -> Result<Self, fftw::error::Error> {
        let fw_plan: C2CPlan64 = C2CPlan::aligned(&[size], Sign::Forward, Flag::ESTIMATE)?;
        let bw_plan: C2CPlan64 = C2CPlan::aligned(&[size], Sign::Backward, Flag::ESTIMATE)?;
        let aligned_query = AlignedVec::new(size);
        let query_transform = aligned_query.clone();
        let product = aligned_query.clone();
        let product_inverse = aligned_query.clone();

        Ok(Self {
            fw_plan,
            bw_plan,
            aligned_query,
            query_transform,
            product,
            product_inverse,
        })
    }

    fn run(
        &mut self,
        db: &[f64],
        db_transform: &AlignedVec<Complex<f64>>,
        query: &[f64],
    ) -> Result<Vec<Complex<f64>>, fftw::error::Error> {
        let ts_len = db_transform.len();
        let query_len = query.len();

        query
            .iter()
            .rev()
            .copied()
            .zip(self.aligned_query.iter_mut())
            .for_each(|(q, y)| y.re = q);
        self.fw_plan
            .c2c(&mut *self.aligned_query, &mut self.query_transform)?;

        self.product
            .iter_mut()
            .zip(&**db_transform)
            .zip(&*self.query_transform)
            .for_each(|((z, x), y)| *z = x * y);

        self.bw_plan
            .c2c(&mut *self.product, &mut *self.product_inverse)?;

        // Normalize results
        let scale_factor = 1. / (ts_len as f64);
        for z in &mut *self.product_inverse {
            *z *= scale_factor;
        }

        let mean_sigma_x = db.windows(query_len).map(|window| mean_stddev(window, 0));
        let (mean_y, sigma_y) = mean_stddev(query, 0);

        let query_len_f64 = query_len as f64;
        Ok(self.product_inverse[query_len - 1..ts_len]
            .iter()
            .zip(mean_sigma_x)
            .map(|(z, (mean_x, sigma_x))| {
                let squared = 2.
                    * (query_len_f64 - (z - query_len_f64 * mean_x * mean_y) / (sigma_x * sigma_y));
                squared.sqrt()
            })
            .collect())
    }
}

fn transform_db(db: &[f64]) -> Result<AlignedVec<Complex<f64>>, fftw::error::Error> {
    let ts_len = db.len();
    let mut aligned_db = AlignedVec::new(ts_len);
    db.iter()
        .copied()
        .zip(aligned_db.iter_mut())
        .for_each(|(t, x)| *x = Complex::new(t, 0.));

    let mut db_transform = AlignedVec::new(ts_len);
    let mut fw_plan: C2CPlan64 = C2CPlan::aligned(&[ts_len], Sign::Forward, Flag::ESTIMATE)?;
    fw_plan.c2c(&mut *aligned_db, &mut db_transform)?;

    Ok(db_transform)
}

fn hamming_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(a, b)| a != b).count()
}

fn gini_index<T>(data: &[T]) -> T
where
    T: NumAssignRef + Float + Sum,
    for<'a> &'a T: RefNum<T>,
{
    let mut iter = data.iter();
    let mut diffs = T::zero();
    let mut sums = T::zero();
    while let Some(x_a) = iter.next() {
        sums += x_a;
        diffs += iter.clone().map(|x_b| (x_a - x_b).abs()).sum::<T>();
    }

    diffs / (T::from(data.len()).unwrap() * sums)
}

trait ComplexExt {
    fn sqrt(&self) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn is_finite(&self) -> bool;
}

impl ComplexExt for f64 {
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    fn powi(&self, n: i32) -> Self {
        f64::powi(*self, n)
    }

    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
}

impl<T> ComplexExt for Complex<T>
where
    T: Float + FloatCore,
{
    fn sqrt(&self) -> Self {
        <Complex<T>>::sqrt(*self)
    }

    fn powi(&self, n: i32) -> Self {
        Complex::powi(self, n)
    }

    fn is_finite(&self) -> bool {
        Complex::is_finite(*self)
    }
}

fn mean_stddev<T>(data: &[T], degree_of_freedom: u8) -> (T, T)
where
    T: NumCast + NumRef + Sum + for<'a> Sum<&'a T> + ComplexExt,
    for<'a> &'a T: RefNum<T>,
{
    let (sample_size, sum) = data
        .iter()
        .filter(|x| x.is_finite())
        .fold((0usize, T::zero()), |(count, sum), x| (count + 1, sum + x));
    let sample_size = T::from(sample_size).unwrap();
    let mean = sum / &sample_size;
    let var_sum: T = data
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (x - &mean).powi(2))
        .sum();
    (
        mean,
        (var_sum / (sample_size - T::from(degree_of_freedom).unwrap())).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use super::*;

    #[test]
    fn test_mass() {
        let ts = [1., 1., 1., 2., 1., 1., 4., 5.];
        let ts_t = transform_db(&ts).unwrap();
        let query = [2., 1., 1., 4.];
        let result = Mass::new(ts.len())
            .unwrap()
            .run(ts.as_ref(), &ts_t, query.as_ref())
            .unwrap();

        const EXPECTED: [Complex<f64>; 5] = [
            Complex::new(0.67640791, -1.37044402e-16),
            Complex::new(3.43092352, 0.00000000e+00),
            Complex::new(3.43092352, 1.02889035e-17),
            Complex::new(0., 0.),
            Complex::new(1.85113597, 1.21452707e-17),
        ];

        assert_abs_diff_eq!(&*result, EXPECTED.as_ref(), epsilon = 1e-7);
    }

    const NAN: f64 = f64::NAN;

    const QUERY_SEQUENCE: &[u8] = b"GATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATCTGATACTGGCAAGCTTGAGTCTCGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACGAAGACTGACGCTCAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGG";
    #[allow(clippy::approx_constant)]
    const QUERY: &[f64] = &[
        0.052, 0.046, 0.108, 0.241, 0.221, 1.224, 0.246, 0.846, 1.505, 0.627, 0.078, 0.002, 0.056,
        0.317, 0.114, 0.157, 0.264, 1.016, 2.925, 2.205, 1.075, 1.210, 0.191, 0.016, 0.045, 0.015,
        0.087, 0.572, 0.052, 0.157, 0.796, 2.724, 0.027, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004,
        0.003, 0.063, 0.144, 0.072, 0.054, 0.096, 0.112, 0.002, 0.000, 0.019, 0.026, 0.021, 1.022,
        2.108, 0.111, 0.000, 0.007, 0.000, 0.002, 0.000, 0.010, 0.037, 0.078, 0.152, 0.355, 1.738,
        0.715, 0.211, 0.179, 0.036, 0.046, 0.159, 0.257, 0.312, 0.931, 0.798, 0.618, 0.935, 0.147,
        0.015, 0.014, 0.031, 0.147, 0.149, 0.577, 1.052, 1.410, 0.487, 0.636, 0.238, 0.286, 0.462,
        1.586, 1.683, 0.597, 1.165, 1.265, 2.094, 0.422, 0.462, 1.900, 4.055, 0.481, 0.511, 0.087,
        1.217, 1.180, 0.094, 0.018, 0.033, 0.081, 0.148, 0.163, 0.160, 1.019, 0.339, 0.507, 1.039,
        0.824, 0.122, 0.420, 0.429, 0.913, 1.383, 0.610, 0.417, 0.825, 0.743, 0.433, 0.401, 0.993,
        0.497, 0.404, 0.407, 0.316, 0.017, 0.005, 0.046, 0.072, 0.037, 0.091, 0.282, 0.203, 0.033,
        0.004, 0.021, 0.262, 0.157, 0.050, 0.019, 0.059, 0.102, 0.083, 0.066, 0.040, 0.075, 0.061,
        0.573, 0.631, 0.427, 0.265, 1.190, 0.066, 0.042, 0.085, 0.424, 0.413, 0.375, 0.447, 0.035,
        0.045, 0.037, 0.242, 0.221, 0.157, 0.170, 0.370, 1.238, 0.743, 0.571, 0.138, 0.837, 0.859,
        0.042, 0.021, 0.080, 0.318, 0.195, 0.792, 1.581, 1.058, 2.004, 1.512, 2.273, 1.256, 0.036,
        0.005, 0.094, 0.091, 0.464, 0.741,
    ];

    const DB_SEQUENCE: &[u8] = b"TTTATCGGAGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCA\
            AGTCGAGCGGACAGATGGGAGCTTGCTCCCTGATGTTAGCGGCGGACGGGTGAGTAACACGTGGGTAACCTGCCTGTAAGACTGG\
            GATAACTCCGGGAAACCGGGGCTAATACCGGATGGTTGTTTGAACCGCATGGTTCAAACATAAAAGGTGGCTTCGGCTACCACTT\
            ACAGATGGACCCGCGGCGCATTAGCTAGTTGGTGAGGTAACGGCTCACCAAGGCAACGATGCGTAGCCGACCTGAGAGGGTGATC\
            GGCCACACTGGGACTGAGACACGGCCCAGACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCGCAATGGACGAAAGTCTGACGGA\
            GCAACGCCGCGTGAGTGATGAAGGTTTTCGGATCGTAAAGCTCTGTTGTTAGGGAAGAACAAGTACCGTTCGAATAGGGCGGTAC\
            CTTGACGGTACCTAACCAGAAAGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTGTCCGGAATT\
            ATTGGGCGTAAAGGGCTCGCAGGCGGTTTCTTAAGTCTGATGTGAAAGCCCCCGGCTCAACCGGGGAGGGTCATTGGAAACTGGG\
            GAACTTGAGTGCAGAAGAGGAGAGTGGAATTCCACGTGTAGCGGTGAAATGCGTAGAGATGTGGAGGAACACCAGTGGCGAAGGC\
            GACTCTCTGGTCTGTAACTGACGCTGAGGAGCGAAAGCGTGGGGAGCGAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAAC\
            GATGAGTGCTAAGTGTTAGGGGGTTTCCGCCCCTTAGTGCTGCAGCTAACGCATTAAGCACTCCGCCTGGGGAGTACGGTCGCAA\
            GACTGAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGAAGCAACGCGAAGAACCTTACCA\
            GGTCTTGACATCCTCTGACAATCCTAGAGATAGGACGTCCCCTTCGGGGGCAGAGTGACAGGTGGTGCATGGTTGTCGTCAGCTC\
            GTGTCGTGAGATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTGATCTTAGTTGCCAGCATTCAGTTGGGCACTCTAAGGTGA\
            CTGCCGGTGACAAACCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGACCTGGGCTACACACGTGCTACAATGG\
            ACAGAACAAAGGGCAGCGAAACCGCGAGGTTAAGCCAATCCCACAAATCTGTTCTCAGTTCGGATCGCAGTCTGCAACTCGACTG\
            CGTGAAGCTGGAATCGCTAGTAATCGCGGATCAGCATGCCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACC\
            ACGAGAGTTTGTAACACCCGAAGTCGGTGAGGTAACCTTTTAGGAGCCAGCCGCCGAAGGTGGGACAGATGATTGGGGTGAAGTC\
            GTAACAAGGTAGCCGTATCGGAAGGTGCGGCTGGATCACCTCCTTTCT";
    const DB: &[f64] = &[
        NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 0.062, 0.341, 1.105,
        0.021, 0.320, 0.000, 0.054, 0.082, 0.289, 0.665, 0.174, 0.242, 1.248, 0.718, 0.035, 0.178,
        0.000, 0.273, 0.343, 0.000, 0.000, 0.175, 0.294, 0.142, 0.322, 0.885, 0.279, 0.058, 0.000,
        0.007, 0.000, 0.121, 0.106, 0.455, 0.137, 0.092, 0.112, 0.068, 0.159, 0.800, 0.593, 0.715,
        0.361, 0.581, 0.216, 0.185, 0.038, 0.593, 0.238, 0.072, 0.177, 0.392, 0.139, 0.357, 0.235,
        0.199, NAN, 0.082, 0.273, 0.316, 0.958, NAN, 1.218, 0.314, 0.041, 0.090, 0.457, 0.592,
        1.882, 2.266, 5.088, 2.483, 1.443, 1.552, 0.000, 0.000, 0.101, 0.311, 0.335, 0.134, 0.718,
        0.180, 0.110, 0.167, 0.291, 0.379, 0.458, 3.607, 1.904, 0.119, 0.304, 0.201, 0.050, 0.309,
        0.003, 0.200, 0.389, 0.590, 0.349, 0.465, 0.184, 0.352, 0.160, 0.351, 0.130, 0.513, 0.593,
        0.064, 0.061, 0.035, 0.000, 0.040, 0.036, 0.077, 0.331, 0.083, 0.119, 0.581, 0.580, 0.011,
        0.000, 0.095, 0.033, 0.126, 0.019, 0.159, 0.421, 0.534, 0.240, 0.202, 0.000, 0.000, 0.061,
        0.064, 0.179, 0.053, 0.071, 0.119, 0.117, 0.114, 0.072, 0.073, 0.029, 0.107, 0.204, 0.279,
        0.513, 0.010, 0.084, 0.071, 0.046, 0.122, 0.049, 0.031, 0.228, 0.138, 0.201, 1.015, 0.777,
        0.067, 0.212, 0.034, 0.125, 0.309, 0.527, 3.717, 6.653, 0.267, 0.149, 0.014, 0.000, 0.000,
        0.000, 0.055, 0.103, 0.000, 0.104, 0.103, 1.352, 2.460, 1.574, 1.616, 0.000, 0.000, 0.060,
        0.038, 0.006, 0.064, 0.000, 0.044, 0.005, 0.000, 0.465, 0.035, 0.636, 0.087, 0.350, 0.054,
        0.018, 0.132, 0.049, 0.073, 0.440, 0.874, 1.338, 4.775, 3.409, 0.184, 0.188, 0.252, 0.007,
        0.000, 0.019, 0.158, 0.000, 0.021, 0.016, NAN, 0.070, 0.062, 0.162, 0.152, 0.203, 0.045,
        0.075, 0.136, 0.023, 0.032, 0.000, 0.062, 0.076, 0.188, 0.183, 0.123, 0.124, 0.328, 3.735,
        0.864, 3.097, 0.264, 0.084, 0.166, 0.797, 7.286, 1.166, 0.124, 0.026, 0.000, 0.011, 0.000,
        0.056, 0.081, 0.009, 0.090, 0.248, 0.382, 0.809, NAN, 1.992, 0.118, 0.000, 0.000, 0.002,
        0.000, 0.000, 0.000, 0.019, 0.302, 0.088, 0.000, 0.058, NAN, 3.595, 1.139, 0.214, 0.010,
        0.055, 0.014, 0.016, 0.075, 0.050, 0.085, 0.112, 0.016, 0.058, 0.231, 0.513, 0.043, 0.151,
        0.577, 0.522, 0.419, 0.243, 1.276, 0.986, 0.234, 0.310, 0.445, 0.410, 0.498, 0.303, 0.306,
        0.004, 0.068, 0.030, 0.034, 0.279, 0.000, 0.190, 0.179, 0.088, 0.154, 0.265, 0.821, 0.242,
        0.232, 0.397, 0.353, 0.124, 0.521, 0.385, 0.339, 0.651, 0.865, 0.103, 0.023, 0.038, 0.011,
        0.000, 0.130, 0.327, 0.177, 0.068, 1.151, 0.052, 0.000, 0.682, 1.551, 7.803, 1.643, 0.125,
        0.132, 0.099, 0.068, 0.056, 0.077, 0.095, 0.145, 0.067, 0.166, 0.355, 1.197, 0.435, 0.202,
        0.160, 0.457, 0.139, 0.498, 0.246, 0.327, 0.444, 0.190, 0.061, 0.000, 0.172, 0.453, 0.286,
        1.260, 1.281, 0.211, 0.043, 0.023, 0.109, 0.662, 0.505, 0.229, 0.442, 0.050, 0.000, 0.091,
        0.137, 0.156, 0.115, 0.239, 0.102, 0.077, 0.173, 0.060, 0.205, 0.154, 0.179, 0.000, 0.072,
        0.006, 0.186, 0.133, 0.370, 0.737, 0.894, 0.328, 0.442, 0.255, 1.351, 1.007, 6.596, 25.671,
        2.392, 1.200, 0.636, 1.735, 2.845, 1.143, 1.666, 2.465, 7.192, 12.230, 6.393, 0.498, 0.155,
        0.205, 0.292, 0.280, 0.465, 0.058, 0.272, 0.420, 0.190, 0.097, 0.000, 0.288, 0.303, 0.851,
        0.855, 0.303, 0.176, 0.087, 0.114, 0.104, 0.168, 0.528, 0.745, 0.571, 0.734, 0.302, 0.252,
        0.330, 0.000, 0.011, 0.012, 0.078, 0.000, NAN, 0.038, 0.092, 0.263, 0.050, 0.108, 0.286,
        0.673, 0.833, 3.207, NAN, 10.814, 0.050, 0.000, 0.000, 0.000, 0.000, 0.019, 0.014, 0.000,
        0.017, 0.019, 0.071, 0.075, 0.135, 0.000, 0.171, 0.039, 0.069, 0.062, 0.064, 0.004, 0.006,
        0.000, 0.012, 0.041, 0.060, 0.017, 0.958, 5.108, 1.616, 0.364, 0.312, 0.077, 0.018, 0.047,
        0.089, 0.141, 0.471, 0.363, 0.399, 0.174, 0.167, 0.315, 0.105, 0.146, 0.089, 0.033, 0.043,
        0.112, 0.812, 0.516, 0.320, 0.238, 0.105, 0.088, 0.249, 0.976, 0.231, 1.131, 0.224, 0.444,
        0.693, 1.946, 6.156, 0.440, 1.818, 1.084, 0.342, 0.007, 0.034, 0.003, 0.026, 0.185, 0.356,
        0.000, 0.000, 0.076, 0.419, 0.349, 0.110, 0.053, 0.043, 0.128, 0.047, 0.169, NAN, 0.090,
        0.049, 0.026, 0.220, 0.649, 1.070, 1.380, 2.489, 1.953, 2.318, 2.673, 4.799, 0.757, 0.147,
        0.049, 0.203, 0.091, 0.290, 0.142, 0.520, 1.388, 0.747, 0.818, 0.290, 0.079, 0.260, 0.343,
        0.636, 0.296, 1.774, 4.121, 32.001, 0.089, 0.147, 0.000, 0.000, 0.040, 0.007, 0.044, 0.077,
        0.146, 0.101, 0.160, 0.105, 0.000, 0.075, 0.122, 0.193, 0.065, 0.090, 0.449, 0.265, 1.419,
        0.316, 0.323, 0.626, 0.264, 0.000, 0.000, 0.018, 0.000, 0.035, 0.048, 0.136, 0.325, 1.744,
        1.853, 0.446, 0.829, 0.157, 0.043, 0.040, 0.040, 0.090, 0.309, 1.043, 0.561, 0.497, 0.752,
        8.159, 0.041, 0.067, 0.055, 0.209, 0.294, 0.093, 0.045, 0.208, 0.240, 0.090, 0.352, 0.461,
        0.049, 0.288, NAN, 0.091, 0.124, 0.104, 0.414, 0.743, 0.115, 0.028, 0.000, 0.044, 0.003,
        0.054, 0.019, 0.089, 0.282, 0.244, 1.372, 1.179, 0.682, 0.333, 0.039, 0.026, 0.134, 0.138,
        0.593, 0.376, 0.971, 0.564, 1.306, 1.717, 0.280, 0.047, 0.047, 0.115, 0.321, 0.800, 1.384,
        2.098, 2.749, 1.163, 1.298, 0.832, 0.661, 1.129, 3.666, 3.748, 1.502, 2.923, 3.276, 4.844,
        1.418, 2.402, 7.579, 16.169, 0.797, 0.972, 0.208, 2.506, 3.204, 2.611, 0.348, 0.490, 0.356,
        0.392, 0.492, 0.342, 1.221, 0.629, 1.097, 0.296, 0.736, 0.490, 0.235, 0.324, 0.565, 2.834,
        2.200, 0.281, 0.814, 0.686, 0.402, 0.215, 0.632, 0.336, 0.266, 0.396, 0.000, 0.009, 0.000,
        0.000, 0.000, 0.022, 0.036, 0.161, 0.261, 0.053, 0.068, 0.036, 0.194, 0.426, 0.578, 6.251,
        0.012, 0.086, 0.020, 0.056, 0.050, 0.000, 0.047, 0.433, 0.703, 0.756, 0.692, 1.834, 0.754,
        0.780, 0.283, 0.688, 0.436, 0.436, 0.391, 0.000, 0.093, 0.048, 0.585, 0.698, 0.396, 0.331,
        1.120, 2.582, 1.226, 0.738, 0.200, 0.601, 0.290, 0.049, 0.117, 0.268, 0.708, 0.497, 2.001,
        3.928, 2.840, 4.899, 3.712, 5.464, 2.886, 0.069, 0.000, 0.262, 0.130, 0.717, 0.970, 0.870,
        0.394, 0.748, 0.536, 0.053, 0.000, 0.064, 0.083, 0.325, 0.032, 0.163, 1.630, 1.937, 0.330,
        2.792, 3.440, 3.613, 2.724, 1.081, 2.358, 0.049, 0.013, 0.019, 0.000, 0.000, 0.035, 0.151,
        0.202, 0.316, 0.062, 0.070, 0.632, 0.406, 0.568, 0.306, 0.493, 0.228, 0.222, 0.364, 2.987,
        0.949, 6.935, 9.264, 2.962, 5.298, 1.860, 0.203, 0.000, 0.000, 0.000, 0.004, 0.120, 0.128,
        0.157, 0.331, 0.184, 0.114, 0.135, 0.062, 0.082, 0.030, 0.048, 0.092, 0.097, 0.057, 0.063,
        0.164, 0.000, 0.024, 0.175, 0.500, 1.068, 0.210, 0.087, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.106, 0.131, 0.564, 0.000, 0.125, 0.928, 0.648, 0.565, 1.062, 0.832, 0.102, 0.235, 0.269,
        0.310, 0.341, 0.000, 0.055, 0.000, 0.044, 0.285, 0.622, 0.026, 0.486, 0.019, 0.000, 0.061,
        0.000, 0.887, 0.888, 0.356, 0.452, 0.109, 0.000, 0.787, 0.127, 0.466, 0.745, 0.849, 0.985,
        0.369, 0.836, 0.997, 1.295, 0.061, 0.000, 0.140, 0.488, 0.286, 0.247, 0.449, 0.601, 0.052,
        0.000, 0.000, 0.129, 0.000, 0.000, 0.071, 0.094, 0.070, 0.000, 0.031, 0.002, 0.030, 0.067,
        0.070, 0.104, 0.000, 0.000, 0.000, 0.000, 0.027, 0.022, 0.038, 0.046, 0.215, 0.064, 0.095,
        0.147, 0.091, 0.121, 0.080, 0.147, 0.722, 2.543, 2.376, 2.516, 5.340, 2.277, 1.973, 1.131,
        2.245, 0.904, 1.524, 1.088, 0.368, 0.666, 0.243, 0.165, 0.196, 0.034, 0.134, 0.403, 0.389,
        1.234, 0.059, 0.000, 0.032, 0.072, 0.353, 0.113, 2.816, 4.629, 1.820, 0.271, 0.110, 0.263,
        0.545, 0.139, 0.003, 0.045, 0.000, 0.028, 0.064, 0.188, 0.158, 0.530, 0.203, 0.416, 0.156,
        0.010, 0.000, 0.046, 0.085, 0.115, 0.089, 0.611, 0.652, 0.143, 0.107, 0.086, 0.164, 0.084,
        0.783, 1.817, 1.600, 0.041, 0.061, 0.058, 0.780, 1.567, 4.661, 8.240, 3.039, 0.083, 0.009,
        0.121, 0.111, 0.016, 0.111, 0.090, 0.108, 0.339, 2.636, 0.279, 0.593, 0.778, 0.684, 0.548,
        2.324, 2.113, 1.190, NAN, 2.556, 1.137, 0.206, 0.034, 0.108, 0.520, 0.783, 0.414, 0.776,
        0.197, 0.162, 0.153, 0.375, 0.547, 0.321, 0.450, 0.028, 0.032, 0.109, 0.091, 0.275, 0.433,
        0.419, 0.302, 0.336, 0.169, 0.438, 0.349, 0.842, 0.165, 0.286, 0.393, 0.389, 0.757, 0.643,
        0.246, 0.470, 0.578, 0.455, 1.020, 1.294, 1.442, 4.392, 0.737, 0.016, 0.060, 0.060, 0.089,
        0.417, 0.429, 0.483, 0.042, 0.041, 0.047, 0.064, 0.914, 0.965, 1.207, 1.162, 2.204, 0.000,
        0.021, 0.139, 0.631, 4.257, 0.748, 0.011, 0.120, 0.001, 0.010, 0.241, 0.049, 0.151, 0.171,
        0.257, 0.092, 0.053, 0.260, 0.491, 1.636, 1.045, 1.647, 1.923, 3.996, 1.572, 1.294, 1.326,
        1.102, 1.135, 0.767, 0.357, 0.265, 0.578, 4.447, 2.240, 0.344, 0.025, 0.074, 0.126, 0.180,
        0.336, 0.274, 0.154, 0.356, 0.131, 0.752, 1.198, 0.332, 0.010, 0.076, 0.181, 0.328, 0.617,
        0.995, 2.113, 7.438, 1.289, 2.026, 0.822, 0.090, 0.062, 0.105, 0.096, 0.020, 0.202, 0.333,
        0.158, 0.170, 0.838, 1.493, 3.024, 0.406, 0.089, 0.020, 0.445, 1.038, 2.202, 0.056, 0.073,
        0.039, 0.092, 0.237, 0.470, 0.247, 1.520, 2.555, 2.694, 1.011, 0.253, 0.409, 0.361, 0.550,
        2.247, 2.069, 0.000, 0.010, 0.217, 0.843, 8.720, 10.132, 1.607, 2.414, 0.002, 0.000, 0.014,
        0.047, 0.060, 0.000, 0.035, 0.088, 0.048, 0.124, 0.038, 11.194, 0.165, 0.000, 1.831, 1.847,
        0.085, 0.053, 0.000, 0.000, 0.000, 0.004, 0.111, 0.101, 0.462, 1.792, 0.026, 0.029, 0.028,
        0.042, 0.070, 0.236, 0.222, 0.662, 0.246, 0.090, 0.158, 0.356, 0.506, 0.202, 0.471, 6.149,
        8.865, 2.984, 2.725, 0.159, 0.176, 0.178, 0.138, 0.029, 0.001, 0.100, 0.346, 0.344, 0.297,
        0.084, 0.104, 0.237, 0.236, 0.147, 0.110, 0.076, 0.303, 1.944, 0.288, 1.140, 0.936, 0.029,
        0.060, 0.111, 0.401, 1.406, 0.079, 0.072, 0.072, 0.000, 0.021, 0.225, 0.880, 0.714, 0.471,
        20.160, 5.704, 2.012, 0.052, 0.047, 0.119, 0.146, 0.070, 0.004, 0.000, 0.065, 0.000, 0.000,
        0.000, 0.000, 0.008, 0.000, 0.000, 0.036, 0.077, 0.082, 0.052, 0.048, 0.065, 0.098, 0.006,
        0.137, 0.017, 0.000, 0.000, 0.032, 0.000, 0.015, 0.000, 0.004, 0.033, 0.000, 0.009, 0.694,
        4.143, 3.463, 0.165, 0.538, 0.897, 0.162, 0.151, 0.027, 0.001, 0.268, 0.335, 0.167, 0.084,
        0.073, 0.022, 0.000, 0.000, 0.014, 0.006, 0.065, 0.013, 0.099, 0.125, 1.148, 1.327, 0.051,
        0.245, 0.132, 0.123, 3.628, 0.076, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.272,
        0.000, 0.000, 0.163, 0.118, 0.966, 0.186, 0.338, 0.920, 0.064, 0.042, 0.007, 0.018, 0.076,
        0.759, 0.047, 0.475, 1.912, 4.365, 4.122, 3.026, 0.722, 1.905, 0.223, 0.371, 2.103, 2.450,
        2.442, 1.295, 0.129, 0.196, 0.103, 0.471, 0.323, 0.536, 0.412, 0.104, 0.607, 0.490, 0.369,
        0.964, 0.558, 0.588, 0.450, 0.064, 1.154, 1.427, 0.101, 0.634, 1.141, 0.567, 0.301, 0.284,
        0.109, 0.063, 0.073, 0.077, 0.296, 0.568, 0.990, 1.168, 0.749, 2.836, 3.365, 0.980, 0.431,
        0.381, 1.493, 0.817, 5.718, 1.144, 0.621, 4.504, 4.093, 5.693, 0.465, 1.289, 3.555, 5.129,
        9.864, 19.367, 2.306, 0.119, 0.158, 0.067, 0.033, 0.061, 0.132, 0.178, 0.166, 0.000, 0.023,
        0.034, 0.012, 0.000, 0.056, 0.048, 0.000, 0.177, 0.000, 0.000, 0.000, 0.000, 0.003, 0.000,
        0.000, 0.052, 0.270, 0.116, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
        NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
        NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
        NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    ];

    const MATCHES: &[[usize; 2]] = &[
        [0, 17],
        [0, 676],
        [0, 1158],
        [0, 1380],
        [0, 1435],
        [1, 18],
        [1, 677],
        [1, 1159],
        [1, 1381],
        [2, 19],
        [2, 32],
        [2, 678],
        [2, 1160],
        [2, 1382],
        [3, 20],
        [3, 33],
        [3, 890],
        [4, 21],
        [4, 34],
        [4, 115],
        [4, 891],
        [5, 1001],
        [5, 1426],
        [6, 614],
        [6, 658],
        [6, 670],
        [6, 754],
        [6, 877],
        [6, 1002],
        [6, 1427],
        [7, 755],
        [7, 1236],
        [7, 1428],
        [8, 581],
        [8, 1429],
        [9, 1255],
        [10, 6],
        [10, 963],
        [10, 1314],
        [12, 1168],
        [13, 1111],
        [13, 1169],
        [14, 1033],
        [14, 1170],
        [14, 1214],
        [15, 1034],
        [15, 1171],
        [15, 1215],
        [16, 1035],
        [16, 1216],
        [17, 425],
        [17, 849],
        [19, 427],
        [19, 825],
        [19, 851],
        [19, 1038],
        [19, 1406],
        [20, 335],
        [20, 1039],
        [20, 1236],
        [21, 326],
        [21, 336],
        [21, 810],
        [21, 879],
        [21, 981],
        [21, 1040],
        [21, 1237],
        [32, 355],
        [33, 954],
        [33, 1042],
        [35, 630],
        [35, 1317],
        [36, 77],
        [36, 631],
        [36, 802],
        [37, 78],
        [37, 632],
        [37, 803],
        [38, 79],
        [38, 534],
        [38, 633],
        [38, 804],
        [38, 930],
        [38, 1230],
        [54, 160],
        [54, 239],
        [54, 354],
        [54, 603],
        [54, 661],
        [54, 1129],
        [63, 129],
        [63, 443],
        [63, 580],
        [63, 659],
        [63, 755],
        [63, 1003],
        [63, 1428],
        [64, 572],
        [64, 756],
        [65, 1166],
        [66, 184],
        [66, 1167],
        [66, 1256],
        [67, 185],
        [67, 620],
        [67, 1110],
        [67, 1168],
        [67, 1198],
        [67, 1257],
        [68, 186],
        [68, 621],
        [68, 1111],
        [68, 1169],
        [68, 1258],
        [69, 187],
        [69, 622],
        [69, 1033],
        [69, 1095],
        [69, 1170],
        [69, 1214],
        [69, 1259],
        [70, 188],
        [70, 383],
        [70, 623],
        [70, 1034],
        [70, 1171],
        [70, 1215],
        [70, 1260],
        [71, 189],
        [71, 1035],
        [71, 1172],
        [71, 1216],
        [71, 1261],
        [71, 1403],
        [72, 385],
        [74, 1013],
        [75, 683],
        [75, 810],
        [75, 1014],
        [75, 1165],
        [76, 131],
        [76, 1015],
        [77, 132],
        [77, 403],
        [77, 496],
        [77, 619],
        [77, 741],
        [78, 133],
        [78, 497],
        [78, 620],
        [78, 1359],
        [79, 134],
        [79, 498],
        [79, 621],
        [79, 1018],
        [80, 41],
        [80, 520],
        [80, 688],
        [80, 1019],
        [80, 1077],
        [80, 1104],
        [80, 1185],
        [81, 42],
        [81, 1020],
        [81, 1105],
        [82, 690],
        [83, 236],
        [83, 771],
        [84, 692],
        [86, 1295],
        [86, 1317],
        [87, 77],
        [87, 920],
        [87, 1296],
        [87, 1435],
        [88, 816],
        [88, 1394],
        [88, 1436],
        [89, 1395],
        [89, 1437],
        [90, 1416],
        [90, 1448],
        [93, 19],
        [93, 922],
        [93, 1230],
        [93, 1273],
        [94, 20],
        [94, 702],
        [94, 923],
        [95, 21],
        [95, 924],
        [96, 22],
        [96, 636],
        [96, 892],
        [96, 1233],
        [97, 23],
        [97, 304],
        [97, 545],
        [97, 1272],
        [97, 1363],
        [98, 894],
        [98, 1286],
        [98, 1364],
        [99, 1287],
        [100, 114],
        [100, 548],
        [104, 118],
        [104, 130],
        [104, 158],
        [104, 444],
        [104, 503],
        [104, 572],
        [104, 1265],
        [105, 119],
        [105, 287],
        [106, 292],
        [106, 882],
        [106, 955],
        [106, 1226],
        [107, 16],
        [107, 121],
        [107, 630],
        [107, 886],
        [107, 1227],
        [108, 17],
        [108, 122],
        [108, 887],
        [108, 1228],
        [109, 18],
        [109, 123],
        [109, 888],
        [109, 1229],
        [110, 19],
        [111, 1415],
        [112, 543],
        [112, 1114],
        [112, 1208],
        [112, 1416],
        [113, 544],
        [113, 870],
        [113, 1115],
        [113, 1417],
        [114, 545],
        [114, 871],
        [114, 1158],
        [114, 1363],
        [114, 1418],
        [115, 546],
        [115, 724],
        [115, 1159],
        [115, 1268],
        [115, 1286],
        [115, 1364],
        [115, 1419],
        [116, 141],
        [116, 547],
        [116, 801],
        [116, 1118],
        [116, 1190],
        [116, 1287],
        [116, 1420],
        [117, 608],
        [117, 874],
        [117, 1288],
        [118, 441],
        [118, 1289],
        [119, 323],
        [119, 657],
        [119, 876],
        [119, 1234],
        [119, 1290],
        [119, 1349],
        [120, 13],
        [120, 334],
        [121, 85],
        [121, 388],
        [121, 928],
        [121, 936],
        [122, 86],
        [122, 389],
        [122, 929],
        [122, 937],
        [122, 1000],
        [123, 1423],
        [124, 732],
        [125, 733],
        [126, 734],
        [127, 611],
        [128, 612],
        [128, 736],
        [129, 613],
        [129, 737],
        [131, 84],
        [131, 1176],
        [132, 38],
        [132, 85],
        [132, 928],
        [134, 404],
        [134, 742],
        [134, 1025],
        [134, 1249],
        [134, 1436],
        [135, 405],
        [135, 1026],
        [135, 1184],
        [135, 1250],
        [135, 1437],
        [136, 21],
        [136, 543],
        [137, 22],
        [137, 303],
        [137, 544],
        [137, 1271],
        [138, 23],
        [138, 304],
        [138, 545],
        [138, 1272],
        [139, 24],
        [139, 250],
        [139, 546],
        [139, 1147],
        [139, 1286],
        [140, 25],
        [140, 547],
        [140, 1148],
        [140, 1190],
        [141, 26],
        [141, 56],
        [141, 302],
        [141, 548],
        [141, 941],
        [141, 1149],
        [147, 4],
        [147, 182],
        [147, 493],
        [147, 662],
        [147, 962],
        [148, 494],
        [148, 618],
        [148, 663],
        [148, 1314],
        [149, 237],
        [149, 295],
        [149, 757],
        [149, 828],
        [150, 480],
        [150, 630],
        [150, 758],
        [150, 1109],
        [151, 337],
        [151, 481],
        [151, 631],
        [151, 1110],
        [151, 1224],
        [151, 1379],
        [152, 482],
        [152, 632],
        [152, 1111],
        [152, 1239],
        [152, 1380],
        [152, 1470],
        [163, 190],
        [163, 385],
        [163, 1036],
        [163, 1064],
        [163, 1262],
        [163, 1404],
        [166, 239],
        [166, 354],
        [166, 616],
        [166, 661],
        [166, 952],
        [166, 1129],
        [166, 1477],
        [167, 355],
        [167, 617],
        [167, 662],
        [167, 953],
        [168, 241],
        [168, 356],
        [168, 663],
        [168, 776],
        [168, 1181],
        [169, 357],
        [169, 1248],
        [170, 1249],
        [170, 1316],
        [171, 507],
        [171, 716],
        [171, 1250],
        [171, 1317],
        [172, 48],
        [172, 77],
        [172, 508],
        [172, 717],
        [172, 802],
        [173, 78],
        [173, 509],
        [173, 803],
        [174, 510],
        [177, 27],
        [177, 167],
        [177, 1020],
        [178, 28],
        [178, 168],
        [178, 681],
        [178, 1021],
        [179, 682],
        [182, 965],
        [182, 1133],
        [183, 76],
        [183, 562],
        [183, 791],
        [183, 1168],
        [184, 77],
        [184, 563],
        [184, 792],
        [184, 844],
        [184, 1169],
        [185, 78],
        [185, 564],
        [185, 793],
        [185, 845],
        [185, 1170],
        [186, 79],
        [186, 565],
        [186, 794],
        [186, 1171],
        [187, 80],
        [187, 795],
        [187, 1455],
        [187, 1482],
        [188, 81],
        [188, 796],
    ];

    const GROUPS: &[KmerLookupDbResult] = &[
        KmerLookupDbResult {
            db: [676, 689],
            query: [0, 13],
        },
        KmerLookupDbResult {
            db: [1158, 1171],
            query: [0, 13],
        },
        KmerLookupDbResult {
            db: [1380, 1393],
            query: [0, 13],
        },
        KmerLookupDbResult {
            db: [17, 32],
            query: [0, 15],
        },
        KmerLookupDbResult {
            db: [32, 45],
            query: [2, 15],
        },
        KmerLookupDbResult {
            db: [890, 902],
            query: [3, 15],
        },
        KmerLookupDbResult {
            db: [1001, 1013],
            query: [5, 17],
        },
        KmerLookupDbResult {
            db: [1426, 1440],
            query: [5, 19],
        },
        KmerLookupDbResult {
            db: [754, 766],
            query: [6, 18],
        },
        KmerLookupDbResult {
            db: [1168, 1182],
            query: [12, 26],
        },
        KmerLookupDbResult {
            db: [1214, 1227],
            query: [14, 27],
        },
        KmerLookupDbResult {
            db: [1033, 1051],
            query: [14, 32],
        },
        KmerLookupDbResult {
            db: [425, 438],
            query: [17, 30],
        },
        KmerLookupDbResult {
            db: [849, 862],
            query: [17, 30],
        },
        KmerLookupDbResult {
            db: [335, 347],
            query: [20, 32],
        },
        KmerLookupDbResult {
            db: [1236, 1248],
            query: [20, 32],
        },
        KmerLookupDbResult {
            db: [630, 644],
            query: [35, 49],
        },
        KmerLookupDbResult {
            db: [77, 90],
            query: [36, 49],
        },
        KmerLookupDbResult {
            db: [802, 815],
            query: [36, 49],
        },
        KmerLookupDbResult {
            db: [755, 767],
            query: [63, 75],
        },
        KmerLookupDbResult {
            db: [1166, 1183],
            query: [65, 82],
        },
        KmerLookupDbResult {
            db: [184, 200],
            query: [66, 82],
        },
        KmerLookupDbResult {
            db: [1256, 1272],
            query: [66, 82],
        },
        KmerLookupDbResult {
            db: [1110, 1122],
            query: [67, 79],
        },
        KmerLookupDbResult {
            db: [620, 634],
            query: [67, 81],
        },
        KmerLookupDbResult {
            db: [1033, 1046],
            query: [69, 82],
        },
        KmerLookupDbResult {
            db: [1214, 1227],
            query: [69, 82],
        },
        KmerLookupDbResult {
            db: [383, 396],
            query: [70, 83],
        },
        KmerLookupDbResult {
            db: [1013, 1031],
            query: [74, 92],
        },
        KmerLookupDbResult {
            db: [683, 713],
            query: [75, 105],
        },
        KmerLookupDbResult {
            db: [131, 145],
            query: [76, 90],
        },
        KmerLookupDbResult {
            db: [496, 509],
            query: [77, 90],
        },
        KmerLookupDbResult {
            db: [619, 632],
            query: [77, 90],
        },
        KmerLookupDbResult {
            db: [41, 53],
            query: [80, 92],
        },
        KmerLookupDbResult {
            db: [1104, 1116],
            query: [80, 92],
        },
        KmerLookupDbResult {
            db: [1295, 1307],
            query: [86, 98],
        },
        KmerLookupDbResult {
            db: [1435, 1448],
            query: [87, 100],
        },
        KmerLookupDbResult {
            db: [1394, 1406],
            query: [88, 100],
        },
        KmerLookupDbResult {
            db: [922, 935],
            query: [93, 106],
        },
        KmerLookupDbResult {
            db: [1230, 1244],
            query: [93, 107],
        },
        KmerLookupDbResult {
            db: [19, 34],
            query: [93, 108],
        },
        KmerLookupDbResult {
            db: [892, 905],
            query: [96, 109],
        },
        KmerLookupDbResult {
            db: [1363, 1375],
            query: [97, 109],
        },
        KmerLookupDbResult {
            db: [545, 559],
            query: [97, 111],
        },
        KmerLookupDbResult {
            db: [1286, 1298],
            query: [98, 110],
        },
        KmerLookupDbResult {
            db: [114, 134],
            query: [100, 120],
        },
        KmerLookupDbResult {
            db: [1226, 1240],
            query: [106, 120],
        },
        KmerLookupDbResult {
            db: [886, 899],
            query: [107, 120],
        },
        KmerLookupDbResult {
            db: [16, 30],
            query: [107, 121],
        },
        KmerLookupDbResult {
            db: [1415, 1431],
            query: [111, 127],
        },
        KmerLookupDbResult {
            db: [543, 558],
            query: [112, 127],
        },
        KmerLookupDbResult {
            db: [1114, 1129],
            query: [112, 127],
        },
        KmerLookupDbResult {
            db: [870, 887],
            query: [113, 130],
        },
        KmerLookupDbResult {
            db: [1158, 1170],
            query: [114, 126],
        },
        KmerLookupDbResult {
            db: [1363, 1375],
            query: [114, 126],
        },
        KmerLookupDbResult {
            db: [1286, 1301],
            query: [115, 130],
        },
        KmerLookupDbResult {
            db: [1234, 1261],
            query: [119, 146],
        },
        KmerLookupDbResult {
            db: [85, 97],
            query: [121, 133],
        },
        KmerLookupDbResult {
            db: [388, 400],
            query: [121, 133],
        },
        KmerLookupDbResult {
            db: [928, 940],
            query: [121, 133],
        },
        KmerLookupDbResult {
            db: [936, 948],
            query: [121, 133],
        },
        KmerLookupDbResult {
            db: [732, 807],
            query: [124, 199],
        },
        KmerLookupDbResult {
            db: [611, 624],
            query: [127, 140],
        },
        KmerLookupDbResult {
            db: [84, 96],
            query: [131, 143],
        },
        KmerLookupDbResult {
            db: [404, 416],
            query: [134, 146],
        },
        KmerLookupDbResult {
            db: [1025, 1037],
            query: [134, 146],
        },
        KmerLookupDbResult {
            db: [1436, 1448],
            query: [134, 146],
        },
        KmerLookupDbResult {
            db: [21, 37],
            query: [136, 152],
        },
        KmerLookupDbResult {
            db: [543, 559],
            query: [136, 152],
        },
        KmerLookupDbResult {
            db: [303, 315],
            query: [137, 149],
        },
        KmerLookupDbResult {
            db: [1271, 1283],
            query: [137, 149],
        },
        KmerLookupDbResult {
            db: [1147, 1160],
            query: [139, 152],
        },
        KmerLookupDbResult {
            db: [493, 505],
            query: [147, 159],
        },
        KmerLookupDbResult {
            db: [662, 674],
            query: [147, 159],
        },
        KmerLookupDbResult {
            db: [480, 493],
            query: [150, 163],
        },
        KmerLookupDbResult {
            db: [630, 643],
            query: [150, 163],
        },
        KmerLookupDbResult {
            db: [1109, 1122],
            query: [150, 163],
        },
        KmerLookupDbResult {
            db: [1379, 1391],
            query: [151, 163],
        },
        KmerLookupDbResult {
            db: [616, 628],
            query: [166, 178],
        },
        KmerLookupDbResult {
            db: [952, 964],
            query: [166, 178],
        },
        KmerLookupDbResult {
            db: [239, 252],
            query: [166, 179],
        },
        KmerLookupDbResult {
            db: [661, 674],
            query: [166, 179],
        },
        KmerLookupDbResult {
            db: [354, 368],
            query: [166, 180],
        },
        KmerLookupDbResult {
            db: [1248, 1261],
            query: [169, 182],
        },
        KmerLookupDbResult {
            db: [1316, 1328],
            query: [170, 182],
        },
        KmerLookupDbResult {
            db: [716, 728],
            query: [171, 183],
        },
        KmerLookupDbResult {
            db: [507, 521],
            query: [171, 185],
        },
        KmerLookupDbResult {
            db: [77, 89],
            query: [172, 184],
        },
        KmerLookupDbResult {
            db: [802, 814],
            query: [172, 184],
        },
        KmerLookupDbResult {
            db: [27, 39],
            query: [177, 189],
        },
        KmerLookupDbResult {
            db: [167, 179],
            query: [177, 189],
        },
        KmerLookupDbResult {
            db: [1020, 1032],
            query: [177, 189],
        },
        KmerLookupDbResult {
            db: [681, 693],
            query: [178, 190],
        },
        KmerLookupDbResult {
            db: [562, 576],
            query: [183, 197],
        },
        KmerLookupDbResult {
            db: [1168, 1182],
            query: [183, 197],
        },
        KmerLookupDbResult {
            db: [76, 92],
            query: [183, 199],
        },
        KmerLookupDbResult {
            db: [844, 856],
            query: [184, 196],
        },
    ];

    fn create_kmer_lookup() -> KmerLookup {
        let mut builder = KmerLookup::builder().kmer_len(12);

        builder
            .max_kmer_merge_distance(Some(10))
            .max_matches_every_nt(200)
            .max_reactivity(1.5)
            .min_complexity(0.2)
            .min_kmers(2);

        builder.build()
    }

    fn create_kmer_lookup_data(kmer_lookup: &KmerLookup) -> KmerLookupRunData {
        let mut data = KmerLookupRunData::new(Path::new("."), QUERY, QUERY_SEQUENCE).unwrap();
        data.saturate_query(kmer_lookup.max_reactivity);
        data
    }

    #[test]
    fn matching_kmers() {
        let kmer_lookup = create_kmer_lookup();
        let data = create_kmer_lookup_data(&kmer_lookup);
        let db_data = DbData::new(DB_SEQUENCE.to_vec(), DB.to_vec(), &kmer_lookup).unwrap();
        let mut matches = kmer_lookup.get_matching_kmers(&data, &db_data).unwrap();
        matches.sort_unstable();
        assert_eq!(matches, MATCHES);
    }

    #[test]
    fn group_matching_kmers() {
        let mut groups = create_kmer_lookup().group_matching_kmers(MATCHES);
        groups.sort_unstable_by(|a, b| a.query.cmp(&b.query).then_with(|| a.db.cmp(&b.db)));
        assert_eq!(groups, GROUPS);
    }

    #[test]
    fn gini_index() {
        const DATA: &[f64] = &[
            9., 28., 43., 6., 21., 42., 4., 1., 50., 15., 32., 18., 37., 26., 7., 10., 24., 23.,
            40., 17., 35., 41., 2., 38., 36., 49., 3., 39., 45., 31., 46., 29., 22., 34., 11., 44.,
            13., 27., 5., 30., 25., 12., 33., 19., 47., 8., 48., 14., 20., 16.,
        ];

        assert_relative_eq!(super::gini_index(DATA), 0.327, epsilon = 0.0005);
    }

    #[test]
    fn saturate_query() {
        let mut data = KmerLookupRunData::new(Path::new("."), QUERY, QUERY_SEQUENCE).unwrap();
        data.query[0] = f64::NAN;
        data.query[1] = f64::INFINITY;
        data.query[2] = -f64::INFINITY;
        data.saturate_query(0.);
        let query = &mut data.query;

        assert!(query[3..].iter().all(|&n| n == 0.));
        assert!(query[0].is_nan());
        assert!(query[1] < f64::EPSILON);
        assert!(query[2] < 0. && query[2].is_infinite());
    }
}
