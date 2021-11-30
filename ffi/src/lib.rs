#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    ffi::{CStr, OsStr},
    os::{raw::c_char, unix::ffi::OsStrExt},
    path::Path,
    ptr,
};

use cstr::cstr;

use kmer_lookup::{Distance, KmerLookupRunData};
pub use kmer_lookup::{KmerLookup, KmerLookupResult};

pub type KmerLookupBuilder = kmer_lookup::KmerLookupBuilder<u16>;
pub type KmerLookupResults = Vec<KmerLookupResult>;
pub type KmerLookupOkErr = Result<KmerLookupResults, Error>;

#[derive(Debug)]
pub enum Error {
    KmerLookup(kmer_lookup::Error),
    Query,
}

impl From<kmer_lookup::Error> for Error {
    fn from(error: kmer_lookup::Error) -> Self {
        Error::KmerLookup(error)
    }
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_new(_: c_char, kmer_len: u16) -> Box<KmerLookupBuilder> {
    Box::new(kmer_lookup::KmerLookupBuilder::default().kmer_len(kmer_len))
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_kmer_step(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    kmer_step: usize,
) -> i8 {
    kmer_lookup_builder
        .kmer_step(kmer_step)
        .ok()
        .map(|_| 0)
        .unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_max_sequence_distance(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    distance: f64,
) -> i8 {
    let distance = if (0.0..=1.0).contains(&distance) {
        Distance::Fractional(distance)
    } else if distance > 1. {
        Distance::Integral(distance as usize)
    } else {
        return -1;
    };
    kmer_lookup_builder.max_sequence_distance(Some(distance));
    0
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_max_gc_diff(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    max_gc_diff: f64,
) {
    kmer_lookup_builder.max_gc_diff(Some(Some(max_gc_diff).filter(|&diff| diff > 0.)));
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_min_kmers(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    min_kmers: usize,
) {
    kmer_lookup_builder.min_kmers(min_kmers);
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_max_kmer_merge_distance(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    distance: usize,
) {
    kmer_lookup_builder.max_kmer_merge_distance(Some(distance).filter(|&distance| distance > 0));
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_threads(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    threads: u16,
) {
    kmer_lookup_builder.threads(Some(threads).filter(|&threads| threads > 0));
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_max_reactivity(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    max_reactivity: f64,
) {
    kmer_lookup_builder.max_reactivity(max_reactivity);
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_min_complexity(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    min_complexity: f64,
) {
    kmer_lookup_builder.min_complexity(min_complexity);
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_max_matches_every_nt(
    kmer_lookup_builder: &mut KmerLookupBuilder,
    max_matches_every_nt: usize,
) {
    kmer_lookup_builder.max_matches_every_nt(max_matches_every_nt);
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_build(
    kmer_lookup_builder: &KmerLookupBuilder,
) -> Box<KmerLookup> {
    Box::new(kmer_lookup_builder.clone().build())
}

#[no_mangle]
pub extern "C" fn kmer_lookup_builder_DESTROY(kmer_lookup_builder: Box<KmerLookupBuilder>) {
    drop(kmer_lookup_builder)
}

/// # Safety
/// - `sequence` and `query_sequence` must point to C strings;
/// - `query_len` must describe the len of the slice of `f64` `query`.
#[no_mangle]
pub unsafe extern "C" fn kmer_lookup_run(
    kmer_lookup: &kmer_lookup::KmerLookup,
    db_path: *const c_char,
    query: *const f64,
    query_len: usize,
    query_sequence: *const c_char,
) -> Box<KmerLookupOkErr> {
    let db_path = unsafe { Path::new(OsStr::from_bytes(CStr::from_ptr(db_path).to_bytes())) };
    let query = unsafe { std::slice::from_raw_parts(query, query_len) };
    let query_sequence = unsafe { CStr::from_ptr(query_sequence).to_bytes() };

    let data = match KmerLookupRunData::new(db_path, query, query_sequence) {
        Some(data) => data,
        None => return Box::new(Err(Error::Query)),
    };

    Box::new(kmer_lookup.run(data).map_err(Error::from))
}

#[no_mangle]
pub extern "C" fn kmer_lookup_DESTROY(kmer_lookup: Box<KmerLookup>) {
    drop(kmer_lookup)
}

#[no_mangle]
pub extern "C" fn kmer_lookup_ok_err_is_ok(kmer_lookup_ok_err: &KmerLookupOkErr) -> u8 {
    kmer_lookup_ok_err.is_ok().into()
}

#[no_mangle]
pub extern "C" fn kmer_lookup_ok_err_get_ok(
    kmer_lookup_ok_err: &KmerLookupOkErr,
) -> Option<Box<KmerLookupResults>> {
    kmer_lookup_ok_err.as_ref().ok().cloned().map(Box::new)
}

#[no_mangle]
pub extern "C" fn kmer_lookup_ok_err_get_err(
    kmer_lookup_ok_err: &KmerLookupOkErr,
) -> *const c_char {
    match kmer_lookup_ok_err {
        Ok(_) => cstr!("").as_ptr(),
        Err(Error::Query) => cstr!("query data is incoherent").as_ptr(),
        Err(Error::KmerLookup(kmer_lookup_err)) => {
            use kmer_lookup::Error::*;
            match kmer_lookup_err {
                Fftw(_) => cstr!("FFTW error").as_ptr(),
                Io(_) => cstr!("I/O error").as_ptr(),
                Reader(_) => cstr!("reader error").as_ptr(),
                ReaderEntry(_) => cstr!("reader entry error").as_ptr(),
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn kmer_lookup_ok_err_DESTROY(kmer_lookup_result: Box<KmerLookupOkErr>) {
    drop(kmer_lookup_result);
}

#[no_mangle]
#[allow(clippy::ptr_arg)]
pub extern "C" fn kmer_lookup_results_len(kmer_lookup_results: &KmerLookupResults) -> usize {
    kmer_lookup_results.len()
}

#[no_mangle]
#[allow(clippy::ptr_arg)]
pub extern "C" fn kmer_lookup_results_get(
    kmer_lookup_results: &KmerLookupResults,
    index: usize,
) -> Option<&KmerLookupResult> {
    kmer_lookup_results.get(index)
}

#[no_mangle]
#[allow(clippy::ptr_arg)]
pub extern "C" fn kmer_lookup_results_result_id(
    kmer_lookup_results: &KmerLookupResults,
    index: usize,
) -> *const c_char {
    kmer_lookup_results
        .get(index)
        .map(|result| result.db_id.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
#[allow(clippy::ptr_arg)]
pub extern "C" fn kmer_lookup_results_result_len(
    kmer_lookup_results: &KmerLookupResults,
    index: usize,
) -> usize {
    kmer_lookup_results
        .get(index)
        .map(|result| result.db.len())
        .unwrap_or(0)
}

macro_rules! create_results_result_get {
    ($name:ident, $field:ident, $index:expr) => {
        #[no_mangle]
        #[allow(clippy::ptr_arg)]
        pub extern "C" fn $name(
            kmer_lookup_results: &KmerLookupResults,
            result_index: usize,
            db_index: usize,
        ) -> usize {
            kmer_lookup_results
                .get(result_index)
                .and_then(|result| result.db.get(db_index))
                .map(|db_result| db_result.$field[$index])
                .unwrap_or_default()
        }
    };
}

create_results_result_get!(kmer_lookup_results_result_get_db_start, db, 0);
create_results_result_get!(kmer_lookup_results_result_get_db_end, db, 1);
create_results_result_get!(kmer_lookup_results_result_get_query_start, query, 0);
create_results_result_get!(kmer_lookup_results_result_get_query_end, query, 1);

#[no_mangle]
pub extern "C" fn kmer_lookup_results_DESTROY(kmer_lookup_results: Box<KmerLookupResults>) {
    drop(kmer_lookup_results);
}
