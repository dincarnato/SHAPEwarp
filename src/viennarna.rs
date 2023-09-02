#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::pedantic)]

use core::slice;
use std::{
    ffi::{c_char, c_int, c_short, c_uchar, c_uint, c_ulonglong, c_void, CStr, CString},
    fmt, iter,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Deref, Not},
    ptr::{self, NonNull},
};

use bitflags::bitflags;
use viennarna_mfe_sys::{
    hc_basepair, hc_nuc, vrna_callback_free_auxdata, vrna_callback_recursion_status,
    vrna_exp_param_t, vrna_fc_s, vrna_fc_s__bindgen_ty_1, vrna_fc_s__bindgen_ty_1__bindgen_ty_1,
    vrna_fc_s__bindgen_ty_1__bindgen_ty_2, vrna_fc_type_e, vrna_fc_type_e_VRNA_FC_TYPE_COMPARATIVE,
    vrna_fc_type_e_VRNA_FC_TYPE_SINGLE, vrna_fold_compound_comparative, vrna_fold_compound_free,
    vrna_fold_compound_t, vrna_gr_aux_t, vrna_hc_depot_t, vrna_hc_t,
    vrna_hc_type_e_VRNA_HC_DEFAULT, vrna_hc_type_e_VRNA_HC_WINDOW, vrna_md_set_default, vrna_md_t,
    vrna_mfe, vrna_msa_t, vrna_mx_mfe_t, vrna_mx_pf_t, vrna_param_t, vrna_sc_bp_storage_t,
    vrna_sc_init, vrna_sc_s, vrna_sc_set_stack_comparative, vrna_sc_type_e_VRNA_SC_DEFAULT,
    vrna_sc_type_e_VRNA_SC_WINDOW, vrna_sd_t, vrna_seq_t, vrna_seq_type_e_VRNA_SEQ_DNA,
    vrna_seq_type_e_VRNA_SEQ_RNA, vrna_seq_type_e_VRNA_SEQ_UNKNOWN, vrna_ud_t, FLT_OR_DBL,
    VRNA_OPTION_DEFAULT, VRNA_OPTION_MFE, VRNA_OPTION_PF, VRNA_OPTION_WINDOW,
};

use crate::{
    db_file::ReactivityLike, gapped_reactivity::GappedReactivityLike,
    gapped_sequence::GappedSequenceLike, Molecule, Reactivity,
};

pub use self::inner::*;

// Placeholder for core::ffi::c_size_t
#[allow(non_camel_case_types)]
type c_size_t = usize;

pub struct FoldCompound {
    pub(crate) inner: ptr::NonNull<vrna_fold_compound_t>,
    pub(crate) init_options: FoldCompoundOptions,
}

#[derive(Debug)]
#[repr(transparent)]
pub struct ModelDetails(vrna_md_t);

impl ModelDetails {
    pub fn from_ref(model_details_raw: &vrna_md_t) -> &Self {
        // Safety
        // - `ModelDetails` is transparent over `vrna_md_t`.
        unsafe { &*(model_details_raw as *const vrna_md_t).cast() }
    }

    pub fn max_bp_span_mut(&mut self) -> &mut c_int {
        &mut self.0.max_bp_span
    }
}

impl Default for ModelDetails {
    fn default() -> Self {
        let mut model = MaybeUninit::uninit();

        // Safety: vrna_md_set_default is expected to correctly initialize the structure
        unsafe {
            vrna_md_set_default(model.as_mut_ptr());
            Self(model.assume_init())
        }
    }
}

impl PartialEq for ModelDetails {
    fn eq(&self, other: &Self) -> bool {
        (self.0.temperature == other.0.temperature)
            & (self.0.betaScale == other.0.betaScale)
            & (self.0.pf_smooth == other.0.pf_smooth)
            & (self.0.dangles == other.0.dangles)
            & (self.0.special_hp == other.0.special_hp)
            & (self.0.noLP == other.0.noLP)
            & (self.0.noGU == other.0.noGU)
            & (self.0.noGUclosure == other.0.noGUclosure)
            & (self.0.logML == other.0.logML)
            & (self.0.circ == other.0.circ)
            & (self.0.gquad == other.0.gquad)
            & (self.0.uniq_ML == other.0.uniq_ML)
            & (self.0.energy_set == other.0.energy_set)
            & (self.0.backtrack == other.0.backtrack)
            & (self.0.backtrack_type == other.0.backtrack_type)
            & (self.0.compute_bpp == other.0.compute_bpp)
            & (self.0.nonstandards == other.0.nonstandards)
            & (self.0.max_bp_span == other.0.max_bp_span)
            & (self.0.min_loop_size == other.0.min_loop_size)
            & (self.0.window_size == other.0.window_size)
            & (self.0.oldAliEn == other.0.oldAliEn)
            & (self.0.ribo == other.0.ribo)
            & (self.0.cv_fact == other.0.cv_fact)
            & (self.0.nc_fact == other.0.nc_fact)
            & (self.0.sfact == other.0.sfact)
            & (self.0.rtype == other.0.rtype)
            & (self.0.alias == other.0.alias)
            & (self.0.pair == other.0.pair)
            & (self.0.pair_dist == other.0.pair_dist)
    }
}

impl FoldCompound {
    pub(crate) fn new_comparative<S>(
        sequences: &[S],
        model_details: Option<&ModelDetails>,
        options: FoldCompoundOptions,
    ) -> Result<Self, FoldCompoundError>
    where
        S: GappedSequenceLike,
    {
        let mut iter = sequences
            .iter()
            .map(|sequence| sequence.to_cstring(Some(Molecule::Rna)));
        let first_sequence = iter.next().ok_or(FoldCompoundError::NoSequences)?;

        let sequence_len = first_sequence.to_bytes().len();
        let sequences = iter::once(Ok(first_sequence))
            .chain(iter.map(|sequence| {
                (sequence.to_bytes().len() == sequence_len)
                    .then_some(sequence)
                    .ok_or(FoldCompoundError::UnequalSequences)
            }))
            .collect::<Result<Vec<CString>, FoldCompoundError>>()?;

        let sequences: Vec<_> = sequences
            .iter()
            .map(|x| x.as_ptr())
            .chain(iter::once(ptr::null()))
            .collect();

        // Safety:
        // - sequences must be a non-NULL NULL-terminated array of non NULL NULL-terminated
        //   C-strings, which is being built just beforehand.
        // - md_p must be a nullable pointer to a vrna_md_t (or vrna_md_s) structure; using an
        //   `Option<&ModelDetail>` is fine because `ModelDetail` is transparent.
        // - md_p is never changed by the implementation, even if the pointer is requested to be
        //   mutable.
        // - options must be a set of OR-ed VRNA_OPTIONs (only a subset of them)
        // - both sequences and model details are copied by the function (the strings are copied,
        //   not only the pointers), therefore it is not a problem to drop the original values at
        //   the end of the scope.
        let pointer = unsafe {
            vrna_fold_compound_comparative(
                sequences.as_slice().as_ptr().cast_mut(),
                mem::transmute(model_details),
                options.bits(),
            )
        };

        let inner = NonNull::new(pointer).ok_or(FoldCompoundError::Vienna)?;
        Ok(Self {
            inner,
            init_options: options,
        })
    }

    /// Adds SHAPE reactivity in order to run consensus structure prediction (Deigan et al.
    /// method).
    ///
    /// # Panics
    /// Panics if `options` does not contain the [`FoldCompoundOptions::DEFAULT`] flag.
    pub(crate) fn add_shape_reactivity_for_deigan_consensus_structure_prediction<R, RR>(
        &mut self,
        reactivities: &[R],
        slope: f64,
        intercept: f64,
        options: FoldCompoundOptions,
    ) -> Result<(), AddShapeReactivityError>
    where
        R: GappedReactivityLike<RR>,
        RR: ReactivityLike,
    {
        assert!(
            options.contains(FoldCompoundOptions::DEFAULT),
            "add_shape_reactivity_for_deigan_consensus_structure_prediction only supports the \
            options argument with the `DEFAULT` flag on.",
        );

        let (_, comparative) = self
            .split_mut()
            .into_comparative()
            .ok_or(AddShapeReactivityError::InvalidFoldCompoundType)?;

        if Ok(reactivities.len()) != usize::try_from(comparative.0.n_seq) {
            return Err(AddShapeReactivityError::InvalidContraintsNumber);
        }

        // Underyling behavior:
        // - vrna_sc_init calls vrna_sc_remove because the pointer is not null. This function, when
        //   the type is "comparative", if scs is not null iterates over it using n_seq as length
        //   and calls vrna_sc_free on each element. This:
        //   - runs free_sc_up
        //   - runs free_sc_bp
        //   - frees energy_stack
        //   - frees exp_energy_stack
        //   - if free_data is not null, calls this function passing "data" field to it
        //   - frees it
        // - when the type is "comparative", it allocs scs with length n_seq and iterates over the
        //   allocated memory and calls init_sc_default on each element. This function:
        //   - zero-allocates the size of a vrna_sc_t
        //   - copy into the new memory an initialized version of the struct with the parameter
        //     type equal to VRNA_SC_DEFAULT
        //   - calls nullify on the allocated struct, which
        //     - sets the state to STATE_CLEAN
        //     - sets a bunch of values to NULL
        //     - sets the n field to the n argument, which is length in fold compound
        //     - returns the allocated struct
        //
        // Safety:
        // We **do not** manually change the internal data of the fold compound in any way, and
        // this function does not have any specific requirement. From the behavior described in the
        // part of the comment above (checked on 2022-11-12), the call should be safe since the
        // correct behavior is granted by the correct implementation of the underlying library.
        unsafe { vrna_sc_init(self.as_mut()) };

        let (_, comparative) = self.split_mut().into_comparative().unwrap();

        #[allow(clippy::cast_precision_loss)]
        let weight = if reactivities.is_empty() {
            0.
        } else {
            // This is necessary because the float type for ViennaRNA can be changed through
            // compilation parameters.
            #[allow(clippy::cast_lossless)]
            let n_seq = comparative.0.n_seq as FLT_OR_DBL;

            reactivities.len() as FLT_OR_DBL / n_seq
        };

        // outer len = n_seq
        // inner_len = vc->length + 1
        let contributions: Box<[_]> = reactivities
            .iter()
            .map(|reactivities| {
                let contributions = reactivities
                    .alignment()
                    .filter(|base_or_gap| base_or_gap.is_base())
                    .zip(
                        reactivities
                            .reactivity()
                            .map(|reactivity| {
                                deigan_conversion(reactivity.value(), slope, intercept) * weight
                            })
                            .chain(iter::repeat(0.)),
                    )
                    .map(|(_, energy)| energy);

                iter::once(0.).chain(contributions).collect::<Box<_>>()
            })
            .collect();
        // This is needed because Box<[FLT_OR_DBL]> is a fat pointer, we need something compatible
        // with a slice of "slim" pointers.
        let contributions: Box<[_]> = contributions.iter().map(|inner| inner.as_ptr()).collect();

        // Safety:
        // - no problem with the fold compound, aliasing rules granted by `as_mut`
        // - contributions is never modified inside the function (which has a wrong constness for
        //   this parameter, as always)
        // - options should be just a folding compound options (probably...?)
        // - the fallibility does not leave any dirty data around
        let ret = unsafe {
            vrna_sc_set_stack_comparative(
                self.as_mut(),
                contributions.as_ptr().cast_mut(),
                options.bits(),
            )
        };

        if ret == 1 {
            Ok(())
        } else {
            Err(AddShapeReactivityError::Vienna)
        }
    }

    #[cfg(test)]
    #[inline]
    pub fn is_comparative(&self) -> bool {
        self.as_ref().type_ == vrna_fc_type_e_VRNA_FC_TYPE_COMPARATIVE
    }

    fn split(&self) -> FoldCompoundSplit {
        let inner = self.as_ref();
        let vrna_fc_s {
            type_,
            length,
            cutpoint,
            strand_number,
            strand_order,
            strand_order_uniq,
            strand_start,
            strand_end,
            strands,
            nucleotides,
            alignment,
            hc,
            matrices,
            exp_matrices,
            params,
            exp_params,
            iindx,
            jindx,
            stat_cb,
            auxdata,
            free_auxdata,
            domains_struc,
            domains_up,
            aux_grammar,
            __bindgen_anon_1,
            maxD1,
            maxD2,
            reference_pt1,
            reference_pt2,
            referenceBPs1,
            referenceBPs2,
            bpdist,
            mm1,
            mm2,
            window_size,
            ptype_local,
        } = inner;

        let common = FoldCompoundCommon {
            type_,
            length,
            cutpoint,
            strand_number,
            strand_order,
            strand_order_uniq,
            strand_start,
            strand_end,
            strands,
            nucleotides,
            alignment,
            hc,
            matrices,
            exp_matrices,
            params,
            exp_params,
            iindx,
            jindx,
            stat_cb,
            auxdata,
            free_auxdata,
            domains_struc,
            domains_up,
            aux_grammar,
            max_d1: maxD1,
            max_d2: maxD2,
            reference_pt1,
            reference_pt2,
            reference_bps1: referenceBPs1,
            reference_bps2: referenceBPs2,
            bpdist,
            mm1,
            mm2,
            window_size,
            ptype_local,
        };
        let anon = FoldCompoundAnonInner(__bindgen_anon_1);
        FoldCompoundSplit(common, anon)
    }

    fn split_mut(&mut self) -> FoldCompoundSplitMut {
        let inner = self.as_mut();
        let vrna_fc_s {
            type_,
            length,
            cutpoint,
            strand_number,
            strand_order,
            strand_order_uniq,
            strand_start,
            strand_end,
            strands,
            nucleotides,
            alignment,
            hc,
            matrices,
            exp_matrices,
            params,
            exp_params,
            iindx,
            jindx,
            stat_cb,
            auxdata,
            free_auxdata,
            domains_struc,
            domains_up,
            aux_grammar,
            __bindgen_anon_1,
            maxD1,
            maxD2,
            reference_pt1,
            reference_pt2,
            referenceBPs1,
            referenceBPs2,
            bpdist,
            mm1,
            mm2,
            window_size,
            ptype_local,
        } = inner;

        let common = FoldCompoundCommonMut {
            type_,
            length,
            cutpoint,
            strand_number,
            strand_order,
            strand_order_uniq,
            strand_start,
            strand_end,
            strands,
            nucleotides,
            alignment,
            hc,
            matrices,
            exp_matrices,
            params,
            exp_params,
            iindx,
            jindx,
            stat_cb,
            auxdata,
            free_auxdata,
            domains_struc,
            domains_up,
            aux_grammar,
            max_d1: maxD1,
            max_d2: maxD2,
            reference_pt1,
            reference_pt2,
            reference_bps1: referenceBPs1,
            reference_bps2: referenceBPs2,
            bpdist,
            mm1,
            mm2,
            window_size,
            ptype_local,
        };
        let anon = FoldCompoundAnonInnerMut(__bindgen_anon_1);
        FoldCompoundSplitMut(common, anon)
    }

    #[inline]
    pub fn as_ref(&self) -> &vrna_fold_compound_t {
        // Safety:
        // - The pointer is assumed to be aligned, dereferenceable and contains a valid instance
        //   because of the return from `vrna_fold_compound_comparative`.
        // - Aliasing rules are respected by the reference type of `as_ref`/`as_mut`.
        unsafe { self.inner.as_ref() }
    }

    #[inline]
    pub fn as_mut(&mut self) -> &mut vrna_fold_compound_t {
        // Safety:
        // - The pointer is assumed to be aligned, dereferenceable and contains a valid instance
        //   because of the return from `vrna_fold_compound_comparative`.
        // - Aliasing rules are respected by the reference type of `as_ref`/`as_mut`.
        unsafe { self.inner.as_mut() }
    }

    /// Calculate the minimum free energy structure.
    ///
    /// # Panics
    /// This will panic if the fold compound has not been initialized with the DEFAULT options.
    pub fn minimum_free_energy(&mut self) -> (Structure, f32) {
        assert_eq!(
            self.init_options,
            FoldCompoundOptions::DEFAULT,
            "fold compound minimum_free_energy has been called but the structure has not been \
             instantiated with DEFAULT options"
        );

        let n = usize::try_from(self.as_ref().length).unwrap();
        let mut raw_structure = vec![0u8; n + 1].into_boxed_slice();

        // Safety;
        // - The inner fold compound is alive.
        // - The raw structure is correctly allocated to be (vc->length + 1) chars (as in
        //   RNAalifold.c)
        // - raw_structure is not reallocated by vrna_mfe (otherwise we could not have used the
        //   Rust allocator, which can differ from the one from ViennaRNA).
        let minimum_energy =
            unsafe { vrna_mfe(self.inner.as_ptr(), raw_structure.as_mut_ptr().cast()) };

        (Structure(raw_structure), minimum_energy)
    }
}

impl PartialEq for FoldCompound {
    fn eq(&self, other: &Self) -> bool {
        self.split().into_checked() == other.split().into_checked()
    }
}

#[derive(Debug)]
pub struct Structure(Box<[u8]>);

impl Structure {
    pub fn usable(&self) -> &[u8] {
        self.0.splitn(2, |&c| c == b'\0').next().unwrap_or(&*self.0)
    }
}

impl fmt::Display for Structure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The structure must only contain dot-bracket characters, which are always valid UTF-8.
        f.write_str(std::str::from_utf8(self.usable()).unwrap())
    }
}

#[inline]
fn energy_bp_len(length: c_uint) -> c_uint {
    assert!(length < c_uint::MAX - 2);
    (length + 1).checked_mul(length + 2).unwrap() / 2
}

#[inline]
fn energy_bp_local_len(length: c_uint) -> c_uint {
    length.checked_add(2).unwrap()
}

#[inline]
fn deigan_conversion(reactivity: Reactivity, slope: f64, intercept: f64) -> FLT_OR_DBL {
    if reactivity.is_nan() || reactivity < 0. {
        0.
    } else {
        let converted = (slope * (f64::from(reactivity) + 1.).ln() + intercept) as FLT_OR_DBL;
        if converted.is_nan() {
            0.
        } else {
            converted
        }
    }
}

struct FoldCompoundSplit<'a>(FoldCompoundCommon<'a>, FoldCompoundAnonInner<'a>);

#[derive(Debug, PartialEq)]
struct FoldCompoundSplitChecked<'a> {
    common: FoldCompoundCommon<'a>,
    inner: FoldCompoundInner<'a>,
}

impl<'a> FoldCompoundSplit<'a> {
    pub fn into_checked(self) -> FoldCompoundSplitChecked<'a> {
        let inner = if *self.0.type_ == vrna_fc_type_e_VRNA_FC_TYPE_SINGLE {
            FoldCompoundInner::Single(FoldCompoundInnerSingle(unsafe {
                &self.1 .0.__bindgen_anon_1
            }))
        } else if *self.0.type_ == vrna_fc_type_e_VRNA_FC_TYPE_COMPARATIVE {
            FoldCompoundInner::Comparative(FoldCompoundInnerComparative {
                common: self.0.clone(),
                inner: unsafe { &self.1 .0.__bindgen_anon_2 },
            })
        } else {
            panic!("Invalid fold compound type");
        };

        FoldCompoundSplitChecked {
            common: self.0,
            inner,
        }
    }
}

struct FoldCompoundSplitMut<'a>(FoldCompoundCommonMut<'a>, FoldCompoundAnonInnerMut<'a>);

impl<'a> FoldCompoundSplitMut<'a> {
    fn into_comparative(
        self,
    ) -> Option<(
        FoldCompoundCommonMut<'a>,
        FoldCompoundInnerComparativeMut<'a>,
    )> {
        // Clippy is wrong here: if I use `then_some` instead of `then`, the mutable reference to
        // `__bindgen_anon_2` is created even if the `type_` is not what we need. This is UB,
        // because the specific variant of the union is not "live" and we just cannot create a
        // reference (mutable or not) to it.
        #[allow(clippy::unnecessary_lazy_evaluations)]
        (*self.0.type_ == vrna_fc_type_e_VRNA_FC_TYPE_COMPARATIVE).then(|| {
            // Safety: union field related to comparative type accessed only after checking the
            // type
            (
                self.0,
                FoldCompoundInnerComparativeMut(unsafe { &mut self.1 .0.__bindgen_anon_2 }),
            )
        })
    }
}

#[derive(Clone)]
pub struct FoldCompoundCommon<'a> {
    pub type_: &'a vrna_fc_type_e,
    pub length: &'a c_uint,
    pub cutpoint: &'a c_int,
    pub strand_number: &'a *mut c_uint,
    pub strand_order: &'a *mut c_uint,
    pub strand_order_uniq: &'a *mut c_uint,
    pub strand_start: &'a *mut c_uint,
    pub strand_end: &'a *mut c_uint,
    pub strands: &'a c_uint,
    pub nucleotides: &'a *mut vrna_seq_t,
    pub alignment: &'a *mut vrna_msa_t,
    pub hc: &'a *mut vrna_hc_t,
    pub matrices: &'a *mut vrna_mx_mfe_t,
    pub exp_matrices: &'a *mut vrna_mx_pf_t,
    pub params: &'a *mut vrna_param_t,
    pub exp_params: &'a *mut vrna_exp_param_t,
    pub iindx: &'a *mut c_int,
    pub jindx: &'a *mut c_int,
    pub stat_cb: &'a vrna_callback_recursion_status,
    pub auxdata: &'a *mut c_void,
    pub free_auxdata: &'a vrna_callback_free_auxdata,
    pub domains_struc: &'a *mut vrna_sd_t,
    pub domains_up: &'a *mut vrna_ud_t,
    pub aux_grammar: &'a *mut vrna_gr_aux_t,
    pub max_d1: &'a c_uint,
    pub max_d2: &'a c_uint,
    pub reference_pt1: &'a *mut c_short,
    pub reference_pt2: &'a *mut c_short,
    pub reference_bps1: &'a *mut c_uint,
    pub reference_bps2: &'a *mut c_uint,
    pub bpdist: &'a *mut c_uint,
    pub mm1: &'a *mut c_uint,
    pub mm2: &'a *mut c_uint,
    pub window_size: &'a c_int,
    pub ptype_local: &'a *mut *mut c_char,
}

impl<'a> FoldCompoundCommon<'a> {
    #[inline]
    pub fn strand_number(&self) -> Option<&[c_uint]> {
        self.strand_number.is_null().not().then(|| {
            let len = self.length.checked_add(2).unwrap().try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.length + 2` (as in `sequence.c`).
            unsafe { slice::from_raw_parts(*self.strand_number, len) }
        })
    }

    #[inline]
    pub fn strand_order(&self) -> Option<&[c_uint]> {
        self.strand_order.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.strands + 1` or `2` (as in `sequence.c`) depending by type. This is
            //   guaranteed by `FoldCompoundCommon::strand_order_len`.
            unsafe { slice::from_raw_parts(*self.strand_order, self.strand_order_len()) }
        })
    }

    #[inline]
    pub fn strand_order_uniq(&self) -> Option<&[c_uint]> {
        self.strand_order_uniq.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.strands + 1` or `2` (as in `sequence.c`) depending by type. This is
            //   guaranteed by `FoldCompoundCommon::strand_order_len`.
            unsafe { slice::from_raw_parts(*self.strand_order_uniq, self.strand_order_len()) }
        })
    }

    #[inline]
    pub fn strand_start(&self) -> Option<&[c_uint]> {
        self.strand_start.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.strands + 1` or `2` (as in `sequence.c`) depending by type. This is
            //   guaranteed by `FoldCompoundCommon::strand_order_len`.
            unsafe { slice::from_raw_parts(*self.strand_start, self.strand_order_len()) }
        })
    }

    #[inline]
    pub fn strand_end(&self) -> Option<&[c_uint]> {
        self.strand_end.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.strands + 1` or `2` (as in `sequence.c`) depending by type. This is
            //   guaranteed by `FoldCompoundCommon::strand_order_len`.
            unsafe { slice::from_raw_parts(*self.strand_end, self.strand_order_len()) }
        })
    }

    #[inline]
    pub fn nucleotides(&self) -> Option<&[Sequence]> {
        self.nucleotides.is_null().not().then(|| {
            let len = usize::try_from(*self.strands).unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.strands` (as in `sequence.c`).
            // - `Sequence` is transparent over `vrna_seq_t`.
            unsafe { slice::from_raw_parts(self.nucleotides.cast(), len) }
        })
    }

    #[inline]
    pub fn alignment(&self) -> Option<&[Alignment]> {
        self.alignment.is_null().not().then(|| {
            let len = usize::try_from(*self.strands).unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - `len` is `fc.strands` (as in `sequence.c`).
            // - `Alignment` is transparent over `vrna_msa_t`.
            unsafe { slice::from_raw_parts(self.alignment.cast(), len) }
        })
    }

    #[inline]
    pub fn hc(&self) -> Option<HardConstraints<'_>> {
        self.hc.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `hc` is the same as self.
            let hc = unsafe { &**self.hc };

            HardConstraints {
                inner: hc,
                fold_compound: self,
            }
        })
    }

    #[inline]
    pub fn matrices(&self) -> Option<&'a MinimumFreeEnergyDynamicProgrammingMatrices> {
        self.matrices.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `matrices` is the same as self.
            // - `MinimumFreeEnergyDynamicProgrammingMatrices` is transparent over `vrna_mx_mfe_t`.
            unsafe { &*self.matrices.cast() }
        })
    }

    #[inline]
    pub fn exp_matrices(&self) -> Option<&'a PartitionFunctionDynamicProgrammingMatrices> {
        self.exp_matrices.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `exp_matrices` is the same as self.
            // - `PartitionFunctionDynamicProgrammingMatrices` is transparent over `vrna_mx_pf_t`.
            unsafe { &*self.exp_matrices.cast() }
        })
    }

    #[inline]
    pub fn params(&self) -> Option<&'a FreeEnergyParameters> {
        self.params.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `params` is the same as self.
            // - `FreeEnergyParameters` is transparent over `vrna_param_t`.
            unsafe { &*self.params.cast() }
        })
    }

    #[inline]
    pub fn exp_params(&self) -> Option<&'a BoltzmannFactor> {
        self.exp_params.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `exp_params` is the same as self.
            // - `BoltzmannFactor` is transparent over `vrna_exp_param_t`.
            unsafe { &*self.exp_params.cast() }
        })
    }

    #[inline]
    pub fn iindx(&self) -> Option<&'a [c_int]> {
        self.iindx.is_null().not().then(|| {
            let len = self.length.checked_add(1).unwrap().try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `iindx` is the same as self.
            // - length is `fc->length + 1` (as in `fold_compound.c`, initialized using
            //   `vrna_idx_row_wise`).
            unsafe { slice::from_raw_parts(*self.iindx, len) }
        })
    }

    #[inline]
    pub fn jindx(&self) -> Option<&'a [c_int]> {
        self.jindx.is_null().not().then(|| {
            let len = self.length.checked_add(1).unwrap().try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `jindx` is the same as self.
            // - length is `fc->length + 1` (as in `fold_compound.c`, initialized using
            //   `vrna_idx_col_wise`).
            unsafe { slice::from_raw_parts(*self.jindx, len) }
        })
    }

    #[inline]
    pub fn domains_up(&self) -> Option<&'a UnstructuredDomain> {
        self.domains_up.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `domains_up` is the same as self.
            // - `UnstructuredDomain` is transparent over `vrna_ud_t`.
            unsafe { &*self.domains_up.cast() }
        })
    }

    #[inline]
    pub fn reference_pt1(&self) -> Option<&'a [c_short]> {
        self.reference_pt1.is_null().not().then(|| {
            let len = self.length.checked_add(2).unwrap().try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `reference_pt1` is the same as self.
            // - length is `fc.length + 2` (as in `structured_utils.c` and fold_compound.c`).
            unsafe { slice::from_raw_parts(*self.reference_pt1, len) }
        })
    }

    #[inline]
    pub fn reference_pt2(&self) -> Option<&'a [c_short]> {
        self.reference_pt2.is_null().not().then(|| {
            let len = self.length.checked_add(2).unwrap().try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `reference_pt2` is the same as self.
            // - length is `fc.length + 2` (as in `structured_utils.c` and fold_compound.c`).
            unsafe { slice::from_raw_parts(*self.reference_pt2, len) }
        })
    }

    #[inline]
    pub fn reference_bps1(&self) -> Option<&'a [c_uint]> {
        self.reference_bps1.is_null().not().then(|| {
            let len = (self.length.checked_add(1).unwrap())
                .checked_mul(self.length.checked_add(2).unwrap())
                .unwrap()
                / 2;
            let len = len.try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `reference_bps1` is the same as self.
            // - length is `(len + 1) * (len + 2) / 2` (as in `structured_utils.c` and
            //   fold_compound.c`), where `len` is the first value of `reference_pt1`, which in
            //   turn is just `fc.length`.
            unsafe { slice::from_raw_parts(*self.reference_bps1, len) }
        })
    }

    #[inline]
    pub fn reference_bps2(&self) -> Option<&'a [c_uint]> {
        self.reference_bps2.is_null().not().then(|| {
            let len = (self.length.checked_add(1).unwrap())
                .checked_mul(self.length.checked_add(2).unwrap())
                .unwrap()
                / 2;
            let len = len.try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `reference_bps2` is the same as self.
            // - length is `(len + 1) * (len + 2) / 2` (as in `structured_utils.c` and
            //   fold_compound.c`), where `len` is the first value of `reference_pt2`, which in
            //   turn is just `fc.length`.
            unsafe { slice::from_raw_parts(*self.reference_bps2, len) }
        })
    }

    #[inline]
    pub fn bpdist(&self) -> Option<&'a [c_uint]> {
        self.bpdist.is_null().not().then(|| {
            let len = (self.length.checked_add(1).unwrap())
                .checked_mul(self.length.checked_add(2).unwrap())
                .unwrap()
                / 2;
            let len = len.try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `bpdist` is the same as self.
            // - length is `(len + 1) * (len + 2) / 2` (as in `structured_utils.c` and
            //   fold_compound.c`), where `len` is the first value of `reference_pt1`, which in
            //   turn is just `fc.length`.
            unsafe { slice::from_raw_parts(*self.bpdist, len) }
        })
    }

    #[inline]
    pub fn mm1(&self) -> Option<&'a [c_uint]> {
        self.mm1.is_null().not().then(|| {
            let len = self
                .length
                .checked_mul(self.length.checked_add(1).unwrap())
                .unwrap()
                / 2;
            // Just divided by 2, no need to check for overflow.
            let len = (len + 2).try_into().unwrap();

            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `mm1` is the same as self.
            // - length is `(length * (length + 1)) / 2 + 2` (as in `structured_utils.c` and
            //   fold_compound.c`), where `length` is the first value of the encoded sequence in
            //   `fc` with `how = 0`, which corresponds to the len of the string therefore to
            //   `fc.length`.
            unsafe { slice::from_raw_parts(*self.mm1, len) }
        })
    }

    #[inline]
    pub fn mm2(&self) -> Option<&'a [c_uint]> {
        self.mm2.is_null().not().then(|| {
            let len = self
                .length
                .checked_mul(self.length.checked_add(1).unwrap())
                .unwrap()
                / 2;
            // Just divided by 2, no need to check for overflow.
            let len = (len + 2).try_into().unwrap();

            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `mm2` is the same as self.
            // - length is `(length * (length + 1)) / 2 + 2` (as in `structured_utils.c` and
            //   fold_compound.c`), where `length` is the first value of the encoded sequence in
            //   `fc` with `how = 0`, which corresponds to the len of the string therefore to
            //   `fc.length`.
            unsafe { slice::from_raw_parts(*self.mm2, len) }
        })
    }

    #[inline]
    pub fn ptype_local(&self) -> Option<PairTypes<'a>> {
        self.ptype_local.is_null().not().then(|| {
            let outer_len = self.length.checked_add(1).unwrap().try_into().unwrap();
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `ptype_local` is the same as self, and it is passed to `PairTypes`.
            // - outer length is `fc.length + 1` (as in `alphabet.c`).
            let inner = unsafe { slice::from_raw_parts(self.ptype_local.cast(), outer_len) };
            let inner_len = usize::try_from(self.length.checked_add(5).unwrap())
                .unwrap()
                .min(self.window_size.checked_add(5).unwrap().try_into().unwrap());

            PairTypes {
                inner,
                len: inner_len,
            }
        })
    }

    #[inline]
    fn strand_order_len(&self) -> usize {
        let len = if *self.type_ == vrna_fc_type_e_VRNA_FC_TYPE_SINGLE {
            self.strands.checked_add(1).unwrap()
        } else if *self.type_ == vrna_fc_type_e_VRNA_FC_TYPE_COMPARATIVE {
            self.strands.checked_mul(2).unwrap()
        } else {
            unreachable!()
        };

        len.try_into().unwrap()
    }
}

impl PartialEq for FoldCompoundCommon<'_> {
    fn eq(&self, other: &Self) -> bool {
        (self.type_ == other.type_)
            & (self.length == other.length)
            & (self.cutpoint == other.cutpoint)
            & (self.strand_number() == other.strand_number())
            & (self.strand_order() == other.strand_order())
            & (self.strand_order_uniq() == other.strand_order_uniq())
            & (self.strand_start() == other.strand_start())
            & (self.strand_end() == other.strand_end())
            & (self.strands == other.strands)
            & (self.nucleotides() == other.nucleotides())
            & (self.alignment() == other.alignment())
            & (self.hc() == other.hc())
            & (self.matrices() == other.matrices())
            & (self.exp_matrices() == other.exp_matrices())
            & (self.params() == other.params())
            & (self.exp_params() == other.exp_params())
            & (self.iindx() == other.iindx())
            & (self.jindx() == other.jindx())
            & (self.domains_up() == other.domains_up())
            & (self.max_d1 == other.max_d1)
            & (self.max_d2 == other.max_d2)
            & (self.reference_pt1() == other.reference_pt1())
            & (self.reference_pt2() == other.reference_pt2())
            & (self.reference_bps1() == other.reference_bps2())
            & (self.bpdist() == other.bpdist())
            & (self.mm1() == other.mm1())
            & (self.mm2() == other.mm2())
            & (self.window_size == other.window_size)
            & (self.ptype_local() == other.ptype_local())
    }
}

impl fmt::Debug for FoldCompoundCommon<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FoldCompoundCommon")
            .field("type_", &self.type_)
            .field("length", &self.length)
            .field("cutpoint", &self.cutpoint)
            .field("strand_number", &self.strand_number())
            .field("strand_order", &self.strand_order())
            .field("strand_order_uniq", &self.strand_order_uniq())
            .field("strand_start", &self.strand_start())
            .field("strand_end", &self.strand_end())
            .field("strands", &self.strands)
            .field("nucleotides", &self.nucleotides())
            .field("alignment", &self.alignment())
            .field("hc", &self.hc())
            .field("matrices", &self.matrices())
            .field("exp_matrices", &self.exp_matrices())
            .field("params", &self.params())
            .field("exp_params", &self.exp_params())
            .field("iindx", &self.iindx())
            .field("jindx", &self.jindx())
            .field("stat_cb", &self.stat_cb)
            .field("auxdata", &self.auxdata)
            .field("free_auxdata", &self.free_auxdata)
            .field("domains_struc", &self.domains_struc)
            .field("domains_up", &self.domains_up())
            .field("aux_grammar", &self.aux_grammar)
            .field("max_d1", &self.max_d1)
            .field("max_d2", &self.max_d2)
            .field("reference_pt1", &self.reference_pt1())
            .field("reference_pt2", &self.reference_pt2())
            .field("reference_bps1", &self.reference_bps1())
            .field("reference_bps2", &self.reference_bps2())
            .field("bpdist", &self.bpdist())
            .field("mm1", &self.mm1())
            .field("mm2", &self.mm2())
            .field("window_size", &self.window_size)
            .field("ptype_local", &self.ptype_local())
            .finish()
    }
}

#[derive(Debug)]
pub struct FoldCompoundCommonMut<'a> {
    pub type_: &'a mut vrna_fc_type_e,
    pub length: &'a mut c_uint,
    pub cutpoint: &'a mut c_int,
    pub strand_number: &'a mut *mut c_uint,
    pub strand_order: &'a mut *mut c_uint,
    pub strand_order_uniq: &'a mut *mut c_uint,
    pub strand_start: &'a mut *mut c_uint,
    pub strand_end: &'a mut *mut c_uint,
    pub strands: &'a mut c_uint,
    pub nucleotides: &'a mut *mut vrna_seq_t,
    pub alignment: &'a mut *mut vrna_msa_t,
    pub hc: &'a mut *mut vrna_hc_t,
    pub matrices: &'a mut *mut vrna_mx_mfe_t,
    pub exp_matrices: &'a mut *mut vrna_mx_pf_t,
    pub params: &'a mut *mut vrna_param_t,
    pub exp_params: &'a mut *mut vrna_exp_param_t,
    pub iindx: &'a mut *mut c_int,
    pub jindx: &'a mut *mut c_int,
    pub stat_cb: &'a mut vrna_callback_recursion_status,
    pub auxdata: &'a mut *mut c_void,
    pub free_auxdata: &'a mut vrna_callback_free_auxdata,
    pub domains_struc: &'a mut *mut vrna_sd_t,
    pub domains_up: &'a mut *mut vrna_ud_t,
    pub aux_grammar: &'a mut *mut vrna_gr_aux_t,
    pub max_d1: &'a mut c_uint,
    pub max_d2: &'a mut c_uint,
    pub reference_pt1: &'a mut *mut c_short,
    pub reference_pt2: &'a mut *mut c_short,
    pub reference_bps1: &'a mut *mut c_uint,
    pub reference_bps2: &'a mut *mut c_uint,
    pub bpdist: &'a mut *mut c_uint,
    pub mm1: &'a mut *mut c_uint,
    pub mm2: &'a mut *mut c_uint,
    pub window_size: &'a mut c_int,
    pub ptype_local: &'a mut *mut *mut c_char,
}

impl<'a> FoldCompoundCommonMut<'a> {
    #[cfg(test)]
    #[inline]
    pub fn params(&mut self) -> Option<&'a mut FreeEnergyParameters> {
        self.params.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - lifetime of `params` is the same as self.
            // - `FreeEnergyParameters` is transparent over `vrna_param_t`.
            // - No other mutable references are alive because we need `&mut Self`.
            unsafe { &mut *self.params.cast() }
        })
    }
}

pub struct FoldCompoundAnonInner<'a>(&'a vrna_fc_s__bindgen_ty_1);

#[derive(Debug, PartialEq)]
pub enum FoldCompoundInner<'a> {
    Single(FoldCompoundInnerSingle<'a>),
    Comparative(FoldCompoundInnerComparative<'a>),
}

pub struct FoldCompoundInnerSingle<'a>(#[allow(unused)] &'a vrna_fc_s__bindgen_ty_1__bindgen_ty_1);

impl fmt::Debug for FoldCompoundInnerSingle<'_> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!()
    }
}

impl PartialEq for FoldCompoundInnerSingle<'_> {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

pub struct FoldCompoundInnerComparative<'a> {
    common: FoldCompoundCommon<'a>,
    inner: &'a vrna_fc_s__bindgen_ty_1__bindgen_ty_2,
}

impl fmt::Debug for FoldCompoundInnerComparative<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FoldCompoundInnerComparative")
            .field("sequences", &self.sequences())
            .field("n_seq", &self.inner.n_seq)
            .field("cons_seq", &self.cons_seq())
            .field("S_cons", &self.s_cons())
            .field("S", &self.s())
            .field("S5", &self.s5())
            .field("S3", &self.s3())
            .field("Ss", &self.ss())
            .field("a2s", &self.a2s())
            .field("pscore", &self.pscore())
            .field("pscore_local", &self.pscore_local())
            .field("pscore_pf_compat", &self.pscore_pf_compat())
            .field("scs", &self.scs())
            .finish()
    }
}

impl PartialEq for FoldCompoundInnerComparative<'_> {
    fn eq(&self, other: &Self) -> bool {
        (self.sequences() == other.sequences())
            & (self.cons_seq() == other.cons_seq())
            & (self.s_cons() == other.s_cons())
            & (self.s() == other.s())
            & (self.s5() == other.s5())
            & (self.s3() == other.s3())
            & (self.ss() == other.ss())
            & (self.a2s() == other.a2s())
            & (self.pscore() == other.pscore())
            & (self.pscore_local() == other.pscore_local())
            & (self.pscore_pf_compat() == other.pscore_pf_compat())
            & (self.scs() == other.scs())
    }
}

pub struct FoldCompoundAnonInnerMut<'a>(&'a mut vrna_fc_s__bindgen_ty_1);

pub struct FoldCompoundInnerComparativeMut<'a>(&'a mut vrna_fc_s__bindgen_ty_1__bindgen_ty_2);

impl<'a> FoldCompoundInnerComparative<'a> {
    pub fn sequences(&self) -> Option<CStrSlice<'_>> {
        self.inner.sequences.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked)
            // - length of full slice is `n_seq + 1`, the last pointer must be null 🤷
            // - each pointer except for the last one points to an allocated string
            let ptrs_slice = unsafe {
                slice::from_raw_parts(
                    self.inner.sequences as *const *const c_char,
                    usize::try_from(self.inner.n_seq)
                        .unwrap()
                        .checked_add(1)
                        .unwrap(),
                )
            };
            let (&null_ptr, ptrs_slice) = ptrs_slice.split_last().unwrap();
            assert!(null_ptr.is_null());

            // Safety:
            // All the required invariants have been just checked.
            unsafe { CStrSlice::from_slice_checked(ptrs_slice) }
        })
    }

    pub fn cons_seq(&self) -> Option<&CStr> {
        self.inner.cons_seq.is_null().not().then(|| {
            // Safety
            // - The pointer is not null (just checked).
            // - The pointer is a valid C string (as in fold_compound.c and msa_utils.c) when not
            //   null.
            // - lifetime invariances are guaranteed by the lifetime of self.
            unsafe { CStr::from_ptr(self.inner.cons_seq) }
        })
    }

    pub fn s_cons(&self) -> Option<&[c_short]> {
        self.inner.S_cons.is_null().not().then(|| {
            // S_cons should only be allocated when cons_seq is available.
            let cons_seq_len = self.cons_seq().unwrap().to_bytes().len();

            // Safety
            // - The pointer is not null (just checked).
            // - The pointer is a slice of len len(cons_seq) + 2 (as in fold_compound.c and
            //   msa_utils.c) when not null.
            // - lifetime invariances are guaranteed by the lifetime of self.
            unsafe {
                slice::from_raw_parts(self.inner.S_cons, cons_seq_len.checked_add(2).unwrap())
            }
        })
    }

    pub fn s(&self) -> Option<FoldCompoundSequenceData<'_, c_short>> {
        // Safety:
        // all invariances should be granted by the behavior of ViennaRNA in fold_compound.c.
        unsafe { self.get_inner_sequence_data(self.inner.S as *const *const c_short) }
    }

    pub fn s5(&self) -> Option<FoldCompoundSequenceData<'_, c_short>> {
        // Safety:
        // all invariances should be granted by the behavior of ViennaRNA in fold_compound.c.
        unsafe { self.get_inner_sequence_data(self.inner.S5 as *const *const c_short) }
    }

    pub fn s3(&self) -> Option<FoldCompoundSequenceData<'_, c_short>> {
        // Safety:
        // all invariances should be granted by the behavior of ViennaRNA in fold_compound.c.
        unsafe { self.get_inner_sequence_data(self.inner.S3 as *const *const c_short) }
    }

    pub fn ss(&self) -> Option<FoldCompoundSequenceData<'_, c_char>> {
        // Safety:
        // all invariances should be granted by the behavior of ViennaRNA in fold_compound.c.
        unsafe { self.get_inner_sequence_data(self.inner.Ss as *const *const c_char) }
    }

    pub fn a2s(&self) -> Option<FoldCompoundSequenceData<'_, c_uint>> {
        // Safety:
        // all invariances should be granted by the behavior of ViennaRNA in fold_compound.c.
        unsafe { self.get_inner_sequence_data(self.inner.a2s as *const *const c_uint) }
    }

    pub fn pscore(&self) -> Option<&[c_int]> {
        self.inner.pscore.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - length is calculated using pscore_len (as in fold_compound.c).
            unsafe { slice::from_raw_parts(self.inner.pscore, self.pscore_len()) }
        })
    }

    pub fn pscore_local(&self) -> Option<PScoreLocal<'_>> {
        self.inner.pscore_local.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked)
            // - length of slice is fc->length + 1 (as in fold_compound.c)
            let slice = unsafe {
                slice::from_raw_parts(
                    self.inner.pscore_local as *const *const c_int,
                    usize::try_from(*self.common.length)
                        .unwrap()
                        .checked_add(1)
                        .unwrap(),
                )
            };

            // From mfe_window.c
            let inner_len = usize::try_from(
                (*self.common.length).min(u32::try_from(*self.common.window_size).unwrap_or(0)),
            )
            .unwrap()
            .checked_add(5)
            .unwrap();

            PScoreLocal { slice, inner_len }
        })
    }

    pub fn pscore_pf_compat(&self) -> Option<&[c_short]> {
        self.inner.pscore_pf_compat.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - length is calculated using pscore_len (as in fold_compound.c).
            unsafe { slice::from_raw_parts(self.inner.pscore_pf_compat, self.pscore_len()) }
        })
    }

    #[inline]
    fn pscore_len(&self) -> usize {
        let fc_len = usize::try_from(*self.common.length).unwrap();
        // From fold_compound.c
        fc_len.checked_mul(fc_len.checked_add(1).unwrap()).unwrap() / 2 + 2
    }

    /// # Safety
    /// - ptr must be stored inside self, and its content must outlive self.
    /// - Length of slices pointed by ptr must have length `n_seq + 1`.
    /// - slice must have the same length of sequences, which must be non null.
    /// - each pointer in slice must be non-null and it must point to a slice of `n+2` elements,
    ///   where `n` is the length of the sequence with the same index.
    ///
    /// # Panics
    /// - The function panics if the last element of the slice is not a null pointer.
    unsafe fn get_inner_sequence_data<T>(
        &self,
        ptr: *const *const T,
    ) -> Option<FoldCompoundSequenceData<'_, T>> {
        ptr.is_null().not().then(|| {
            // Safety:
            // - pointer is not null (just checked).
            // - slice has a length of n_seq + 1 by fn invariances.
            let slice = unsafe {
                slice::from_raw_parts(
                    ptr,
                    usize::try_from(self.inner.n_seq)
                        .unwrap()
                        .checked_add(1)
                        .unwrap(),
                )
            };

            // Last element must be a null pointer 🤷
            let (null_ptr, slice) = slice.split_last().unwrap();
            assert!(null_ptr.is_null());

            // Safety:
            // From function invariants
            let sequences = unsafe { self.sequences().unwrap_unchecked() };

            // Safety:
            // All invariances are guaranteed by fn invariances.
            unsafe { FoldCompoundSequenceData::new(slice, sequences) }
        })
    }

    pub(crate) fn scs(&'a self) -> Option<OptionSoftConstraintSlice<'a>> {
        self.inner.scs.is_null().not().then(|| unsafe {
            // Safety:
            // - pointer is not null (just checked).
            // - it is safe to cast from *mut *mut T to *const *const T.
            // - scs is allocated with a length of `fc->n_seq + 1` (as in constraints/soft.c), but
            //   the last element is used as a null guard. Therefore, using a length of `fc->n_seq`
            //   is fine.
            // - lifetime invariances are guaranteed by the lifetime of self that is captured.
            let inner = slice::from_raw_parts(
                self.inner.scs as *const *const vrna_sc_s,
                self.inner
                    .n_seq
                    .try_into()
                    .expect("sequence too large for the current architecture"),
            );

            OptionSoftConstraintSlice {
                inner,
                fold_compound: self.common.clone(),
            }
        })
    }
}

pub struct PScoreLocal<'a> {
    slice: &'a [*const c_int],
    inner_len: usize,
}

impl<'a> PScoreLocal<'a> {
    #[inline]
    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        <&Self as IntoIterator>::into_iter(self)
    }

    #[inline]
    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        <Self as IntoIterator>::into_iter(self)
    }
}

impl fmt::Debug for PScoreLocal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl PartialEq for PScoreLocal<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other)
    }
}

impl Eq for PScoreLocal<'_> {}

impl<'a> IntoIterator for PScoreLocal<'a> {
    type Item = &'a [c_int];
    type IntoIter = PScoreLocalIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let Self { slice, inner_len } = self;
        let inner = slice.iter();
        PScoreLocalIter { inner, inner_len }
    }
}

impl<'a> IntoIterator for &'a PScoreLocal<'a> {
    type Item = &'a [c_int];
    type IntoIter = PScoreLocalIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let &PScoreLocal { slice, inner_len } = self;
        let inner = slice.iter();
        PScoreLocalIter { inner, inner_len }
    }
}

pub struct PScoreLocalIter<'a> {
    inner: slice::Iter<'a, *const c_int>,
    inner_len: usize,
}

impl<'a> Iterator for PScoreLocalIter<'a> {
    type Item = &'a [c_int];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|&ptr| {
            // Safety:
            // Guaranteed by struct invariances.
            unsafe { slice::from_raw_parts(ptr, self.inner_len) }
        })
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub(crate) enum AddShapeReactivityError {
    InvalidFoldCompoundType,
    InvalidContraintsNumber,
    Vienna,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SoftConstraintEnergyRef<'a> {
    Default(SoftConstraintDefaultEnergyRef<'a>),
    Window(SoftConstraintWindowEnergyRef<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SoftConstraintDefaultEnergyRef<'a> {
    pub energy_bp: Option<&'a [c_int]>,
    pub exp_energy_bp: Option<&'a [FLT_OR_DBL]>,
}

#[derive(Debug, Clone)]
pub struct SoftConstraintWindowEnergyRef<'a> {
    energy_bp_local: Option<&'a [*const c_int]>,
    exp_energy_bp_local: Option<&'a [*const FLT_OR_DBL]>,
    energy_len: usize,
    exp_energy_len: usize,
}

impl PartialEq for SoftConstraintWindowEnergyRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Safety:
        // - `len` must be the length of the slice pointed by the inner elements of both `a` and
        //   `b` if they are not `None` and the pointer is not null.
        #[inline]
        unsafe fn helper<T: PartialEq>(
            a: Option<&[*const T]>,
            b: Option<&[*const T]>,
            len: usize,
        ) -> bool {
            match (a, b) {
                (None, None) => true,
                (Some(_), None) | (None, Some(_)) => false,
                (Some(a), Some(b)) => {
                    #[inline]
                    fn to_opt_slice<T>(ptr: &*const T, len: usize) -> Option<&[T]> {
                        ptr.is_null().not().then(move || {
                            // Safety:
                            // - `ptr` is not null (just checked).
                            // - `len` is the length of the slice because it is an invariant of
                            //   the function.
                            unsafe { slice::from_raw_parts(*ptr, len) }
                        })
                    }

                    let a = a.iter().map(|ptr| to_opt_slice(ptr, len));
                    let b = b.iter().map(|ptr| to_opt_slice(ptr, len));

                    a.eq(b)
                }
            }
        }

        self.energy_len == other.energy_len
            && (
                // Safety:
                // - length of self is equal to length of other (just checked).
                // - `energy_bp_local` inner lengths must be `maxdist + 5` (as in mfe_window.c) when
                //   the pointer is not null, where `maxdist` is the minimum value between
                //   `fc.window_size` and `fc.window_size. This is guaranteed by
                //   `FullSoftConstraint::energy`.
                unsafe { helper(self.energy_bp_local, other.energy_bp_local, self.energy_len) }
            ) & (self.exp_energy_len == other.exp_energy_len &&
                // Safety:
                // - length of self is equal to length of other (just checked). -
                //   `exp_energy_bp_local` inner lengths must be `winSize + 1` (as in LPfold.c)
                //   when the pointer is not null, where `winSize` is `fc.window_size`. This is
                //   guaranteed by `FullSoftConstraint::energy`.
                unsafe {
                    helper(
                        self.exp_energy_bp_local,
                        other.exp_energy_bp_local,
                        self.exp_energy_len,
                    )
                })
    }
}

pub struct OptionSoftConstraintSlice<'a> {
    inner: &'a [*const vrna_sc_s],
    fold_compound: FoldCompoundCommon<'a>,
}

impl<'a> OptionSoftConstraintSlice<'a> {
    #[inline]
    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl fmt::Debug for OptionSoftConstraintSlice<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl PartialEq for OptionSoftConstraintSlice<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other)
    }
}

impl<'a> IntoIterator for &'a OptionSoftConstraintSlice<'a> {
    type Item = Option<SoftConstraint<'a>>;
    type IntoIter = OptionSoftConstraintIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let OptionSoftConstraintSlice {
            inner,
            fold_compound,
        } = self;
        let inner = inner.iter();

        OptionSoftConstraintIter {
            inner,
            fold_compound,
        }
    }
}

pub struct OptionSoftConstraintIter<'a> {
    inner: slice::Iter<'a, *const vrna_sc_s>,
    fold_compound: &'a FoldCompoundCommon<'a>,
}

impl<'a> Iterator for OptionSoftConstraintIter<'a> {
    type Item = Option<SoftConstraint<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            inner,
            fold_compound,
        } = self;

        inner.next().map(|&ptr| {
            ptr.is_null().not().then(|| {
                // Safety:
                // - pointer is not null (just checked).
                // - invariances are granted by upstream and downstream structures.
                let inner = unsafe { &*ptr };
                SoftConstraint {
                    inner,
                    fold_compound,
                }
            })
        })
    }
}

#[derive(Clone)]
pub struct SoftConstraint<'a> {
    inner: &'a vrna_sc_s,
    fold_compound: &'a FoldCompoundCommon<'a>,
}

impl SoftConstraint<'_> {
    #[inline]
    pub fn n(&self) -> c_uint {
        self.inner.n
    }

    #[inline]
    pub fn state(&self) -> SoftConstraintState {
        SoftConstraintState::from_bits(self.inner.state).unwrap()
    }

    pub fn energy(&self) -> SoftConstraintEnergyRef<'_> {
        if self.inner.type_ == vrna_sc_type_e_VRNA_SC_DEFAULT {
            let len: usize = energy_bp_len(*self.fold_compound.length)
                .try_into()
                .unwrap();
            // SAFETY: this is associated to the VRNA_SC_DEFAULT type
            let inner = unsafe { self.inner.__bindgen_anon_1.__bindgen_anon_1 };

            let energy_bp = inner.energy_bp.is_null().not().then(|| {
                // Safety:
                // - pointer is not null (just checked).
                // - `energy_bp` is an allocated slice of len `((n + 1) * (n + 2)) / 2`, where
                //   `n` is `fc.length`. This is guaranteed by `energy_bp_len`. See `soft.c`.
                unsafe { slice::from_raw_parts(inner.energy_bp, len) }
            });

            let exp_energy_bp = inner.exp_energy_bp.is_null().not().then(|| {
                // Safety:
                // - pointer is not null (just checked).
                // - `exp_energy_bp` is an allocated slice of len `((n + 1) * (n + 2)) / 2`, where
                //   `n` is `fc.length`. This is guaranteed by `energy_bp_len`. See `soft.c`.
                unsafe { slice::from_raw_parts(inner.exp_energy_bp, len) }
            });

            SoftConstraintEnergyRef::Default(SoftConstraintDefaultEnergyRef {
                energy_bp,
                exp_energy_bp,
            })
        } else if self.inner.type_ == vrna_sc_type_e_VRNA_SC_WINDOW {
            let len: usize = energy_bp_local_len(*self.fold_compound.length)
                .try_into()
                .unwrap();
            let window_size_offset: usize = self
                .fold_compound
                .window_size
                .checked_add(5)
                .unwrap()
                .try_into()
                .unwrap();

            // SAFETY: this is associated to the VRNA_SC_WINDOW type
            let inner = unsafe { self.inner.__bindgen_anon_1.__bindgen_anon_2 };

            let energy_bp_local = inner.energy_bp_local.is_null().not().then(|| {
                // Safety:
                // - pointer is not null (just checked).
                // - `energy_bp_local` is an allocated slice of len `n + 2`, where `n` is
                //   `fc.length`. This is guaranteed by `energy_bp_local_len`. See `soft.c`.
                unsafe { slice::from_raw_parts(inner.energy_bp_local.cast(), len) }
            });

            let exp_energy_bp_local = inner.exp_energy_bp_local.is_null().not().then(|| unsafe {
                // Safety:
                // - pointer is not null (just checked).
                // - `exp_energy_bp_local` is an allocated slice of len `n + 2`, where `n` is
                //   `fc.length`. This is guaranteed by `energy_bp_local_len`. See `soft.c`.
                slice::from_raw_parts(inner.exp_energy_bp_local.cast(), len)
            });

            SoftConstraintEnergyRef::Window(SoftConstraintWindowEnergyRef {
                energy_bp_local,
                exp_energy_bp_local,
                energy_len: window_size_offset.checked_sub(4).unwrap(),
                exp_energy_len: window_size_offset.min(len.checked_add(5).unwrap()),
            })
        } else {
            panic!("Invalid Vienna RNA soft constraints type");
        }
    }

    pub fn energy_up(&self) -> Option<EnergyUnpairedProbabilities<'_, c_int>> {
        self.inner.energy_up.is_null().not().then(|| {
            let len = self.unpaired_probabilities_len();
            // Safety:
            // - It's a slice of pointers related to unpaired probabilities.
            // - Not null (just checked).
            let data = unsafe { slice::from_raw_parts(self.inner.energy_up, len) };
            EnergyUnpairedProbabilities { data, len }
        })
    }

    pub fn exp_energy_up(&self) -> Option<EnergyUnpairedProbabilities<'_, FLT_OR_DBL>> {
        self.inner.exp_energy_up.is_null().not().then(|| {
            let len = self.unpaired_probabilities_len();
            // Safety:
            // - It's a slice of pointers related to unpaired probabilities.
            // - Not null (just checked).
            let data = unsafe { slice::from_raw_parts(self.inner.exp_energy_up, len) };
            EnergyUnpairedProbabilities { data, len }
        })
    }

    #[inline]
    fn unpaired_probabilities_len(&self) -> usize {
        usize::try_from(*self.fold_compound.length)
            .unwrap()
            .checked_add(2)
            .unwrap()
    }

    pub fn up_storage(&self) -> Option<&[c_int]> {
        self.inner.up_storage.is_null().not().then(|| {
            let len = self.unpaired_probabilities_len();
            // Safety:
            // - It's a slice of pointers related to unpaired probabilities.
            // - Not null (just checked).
            unsafe { slice::from_raw_parts(self.inner.up_storage, len) }
        })
    }

    pub fn bp_storage(&self) -> Option<BpStorage<'_>> {
        self.inner.bp_storage.is_null().not().then(|| {
            let len = self.unpaired_probabilities_len();
            // Safety:
            // - It's a slice of pointers related to unpaired probabilities.
            // - Not null (just checked).
            let data = unsafe { slice::from_raw_parts(self.inner.bp_storage, len) };
            BpStorage(data)
        })
    }

    pub fn energy_stack(&self) -> Option<&[c_int]> {
        self.inner.energy_stack.is_null().not().then(|| {
            let len = usize::try_from(*self.fold_compound.length).unwrap();

            // Safety:
            // - pointer is not null (just checked)
            // - when allocated, `energy_stack` is uses `fc.length + 1` as len (see soft.c)
            unsafe { slice::from_raw_parts(self.inner.energy_stack, len + 1) }
        })
    }

    pub fn exp_energy_stack(&self) -> Option<&[FLT_OR_DBL]> {
        self.inner.exp_energy_stack.is_null().not().then(|| {
            let len = usize::try_from(*self.fold_compound.length).unwrap();

            // Safety:
            // - pointer is not null (just checked)
            // - when allocated, `exp_energy_stack` is uses `fc.length + 1` as len (see soft.c)
            unsafe { slice::from_raw_parts(self.inner.exp_energy_stack, len + 1) }
        })
    }
}

impl fmt::Debug for SoftConstraint<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SoftConstraint")
            .field("type", &self.inner.type_)
            .field("n", &self.n())
            .field("state", &self.state())
            .field("energy_up", &self.energy_up())
            .field("exp_energy_up", &self.exp_energy_up())
            .field("up_storage", &self.up_storage())
            .field("bp_storage", &self.bp_storage())
            .field("energy", &self.energy())
            .field("energy_stack", &self.energy_stack())
            .field("exp_energy_stack", &self.exp_energy_stack())
            .field("f", &self.inner.f)
            .field("bt", &self.inner.bt)
            .field("exp_f", &self.inner.exp_f)
            .field("data", &self.inner.data)
            .field("free_data", &self.inner.free_data)
            .finish()
    }
}

impl PartialEq for SoftConstraint<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Note: callbacks and auxiliary data is explicitly not included

        (self.energy() == other.energy())
            & (self.n() == other.n())
            & (self.state() == other.state())
            & (self.energy_up() == other.energy_up())
            & (self.exp_energy_up() == other.exp_energy_up())
            & (self.up_storage() == other.up_storage())
            & (self.bp_storage() == other.bp_storage())
            & (self.energy() == other.energy())
            & (self.energy_stack() == other.energy_stack())
            & (self.exp_energy_stack() == other.exp_energy_stack())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptionalSliceElement<T> {
    NoSlice,
    NoElement,
    Some(T),
}

impl<T> OptionalSliceElement<T> {
    #[inline]
    pub fn from_element(element: Option<T>) -> Self {
        match element {
            Some(element) => Self::Some(element),
            None => Self::NoElement,
        }
    }
}

pub struct EnergyUnpairedProbabilities<'a, T> {
    /// # Note
    ///
    /// The first and the last elements point to just a c_int, the others point to allocated arrays
    /// of size `len - index + 2` (len == fc.length). Moreover, these can be null pointers.
    data: &'a [*mut T],
    len: usize,
}

impl<'a, T> EnergyUnpairedProbabilities<'a, T> {
    pub fn get(&self, index: usize) -> OptionalSliceElement<&[T]> {
        self.data
            .get(index)
            .copied()
            .map_or(OptionalSliceElement::NoSlice, |ptr| {
                // Safety:
                // - we are using the lifetime of the slice.
                // - index is related to the actual offset.
                // - len is `fc.length + 2` (see `FullSoftConstraint::energy_up`)
                OptionalSliceElement::from_element(unsafe {
                    energy_up_get_data(ptr, index, self.len)
                })
            })
    }
}

impl<T: fmt::Debug + Copy> fmt::Debug for EnergyUnpairedProbabilities<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<T> PartialEq for EnergyUnpairedProbabilities<'_, T>
where
    T: PartialEq + Copy,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other)
    }
}

impl<'a, T: Copy> EnergyUnpairedProbabilities<'a, T> {
    #[inline]
    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }

    #[inline]
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }
}

#[inline]
fn unpaired_probabilities_len_at_index(index: usize, len: usize) -> usize {
    if index == 0 {
        1
    } else if index == len - 1 {
        index
    } else {
        len - index
    }
}

/// # Safety
/// - `ptr` must be valid for at least `'a`.
/// - `index` must represent the offset from the start of the slice of pointers.
/// - `len` must be `fc.length + 2`
unsafe fn energy_up_get_data<'a, T>(ptr: *mut T, index: usize, len: usize) -> Option<&'a [T]> {
    ptr.is_null().not().then(|| {
        // Safety:
        // - ptr is not null
        // - length of the slice is evaluated directly from len (which is fc.length +
        //   2), derived from the C code in soft.c
        unsafe { slice::from_raw_parts(ptr, unpaired_probabilities_len_at_index(index, len)) }
    })
}

impl<'a, T: Copy> IntoIterator for EnergyUnpairedProbabilities<'a, T> {
    type Item = Option<&'a [T]>;
    type IntoIter = EnergyUnpairedProbabilitiesIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let Self { data, len } = self;
        let inner = data.iter();
        EnergyUnpairedProbabilitiesIter {
            inner,
            index: 0,
            len,
        }
    }
}

impl<'a, T: Copy> IntoIterator for &EnergyUnpairedProbabilities<'a, T> {
    type Item = Option<&'a [T]>;
    type IntoIter = EnergyUnpairedProbabilitiesIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let &EnergyUnpairedProbabilities { data, len } = self;
        let inner = data.iter();
        EnergyUnpairedProbabilitiesIter {
            inner,
            index: 0,
            len,
        }
    }
}

pub struct EnergyUnpairedProbabilitiesIter<'a, T> {
    inner: slice::Iter<'a, *mut T>,
    index: usize,
    len: usize,
}

impl<'a, T: Copy> Iterator for EnergyUnpairedProbabilitiesIter<'a, T> {
    type Item = Option<&'a [T]>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().copied().map(|ptr| {
            let index = self.index;
            self.index += 1;

            // Safety:
            // - we are using the lifetime of the slice.
            // - index is related to the actual offset (it increments with the iterations).
            // - len is `fc.length + 2` (see `<EnergyUnpairedProbabilities as IntoIterator>::into_iter`)
            unsafe { energy_up_get_data(ptr, index, self.len) }
        })
    }
}

#[derive(Clone, Copy)]
pub struct BpStorage<'a>(&'a [*mut vrna_sc_bp_storage_t]);

impl<'a> BpStorage<'a> {
    #[inline]
    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }

    #[inline]
    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }
}

impl fmt::Debug for BpStorage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl PartialEq for BpStorage<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && self.iter().eq(other)
    }
}

pub struct BpStorageIter<'a>(slice::Iter<'a, *mut vrna_sc_bp_storage_t>);

impl<'a> Iterator for BpStorageIter<'a> {
    type Item = Option<BasePairEnergyContributions<'a>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().copied().map(|ptr| {
            NonNull::new(ptr).map(|ptr| BasePairEnergyContributions {
                ptr,
                _marker: PhantomData,
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a> IntoIterator for BpStorage<'a> {
    type Item = Option<BasePairEnergyContributions<'a>>;
    type IntoIter = BpStorageIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        BpStorageIter(self.0.iter())
    }
}

impl<'a> IntoIterator for &'a BpStorage<'a> {
    type Item = Option<BasePairEnergyContributions<'a>>;
    type IntoIter = BpStorageIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        BpStorageIter(self.0.iter())
    }
}

#[repr(transparent)]
pub struct BasePairEnergyContributions<'a> {
    ptr: NonNull<vrna_sc_bp_storage_t>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> BasePairEnergyContributions<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        // Safety:
        // The inner value comes from `ViennaRNA` data (see `FullSoftConstraint::bp_storage`);
        unsafe { bp_energy_contributions_len(self.ptr) }
    }

    #[inline]
    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }

    #[inline]
    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }
}

impl fmt::Debug for BasePairEnergyContributions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl PartialEq for BasePairEnergyContributions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

/// # Safety
/// `ptr` must point to a slice of memory where at least one element must have an `interval_start`
/// equal to 0. This is how `ViennaRNA` handles the memory internally.
unsafe fn bp_energy_contributions_len(ptr: NonNull<vrna_sc_bp_storage_t>) -> usize {
    let mut len = 0;
    let mut ptr = ptr.as_ptr();
    loop {
        // Safety:
        // This must be guaranteed by the fact that the last base pair has an `interval_start`
        // field equal to 0, and all previous elements are initialized.
        let storage = unsafe { &*ptr };
        if storage.interval_start == 0 {
            break len;
        }

        // Safety:
        // The pointer must be inside a valid allocation.
        ptr = unsafe { ptr.add(1) };
        len += 1;
    }
}

impl<'a> IntoIterator for BasePairEnergyContributions<'a> {
    type Item = &'a BasePairEnergyContribution;
    type IntoIter = BasePairEnergyContributionsIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let ptr = self.ptr.as_ptr();
        BasePairEnergyContributionsIter {
            ptr,
            _marker: PhantomData,
        }
    }
}

impl<'a> IntoIterator for &'a BasePairEnergyContributions<'a> {
    type Item = &'a BasePairEnergyContribution;
    type IntoIter = BasePairEnergyContributionsIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let ptr = self.ptr.as_ptr();
        BasePairEnergyContributionsIter {
            ptr,
            _marker: PhantomData,
        }
    }
}

pub struct BasePairEnergyContributionsIter<'a> {
    ptr: *mut vrna_sc_bp_storage_t,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Iterator for BasePairEnergyContributionsIter<'a> {
    type Item = &'a BasePairEnergyContribution;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Safety:
        // - Pointer is not null because it comes from a `NonNull`.
        // - Data is valid because it comes from a slice of initialized data (see
        //   `<BasePairEnergyContributions as IntoIterator>::into_iter`).
        // - `BasePairEnergyContribution` is transparent and contains a `vrna_sc_bp_storage_t`
        //   (cast is valid).
        let data: &BasePairEnergyContribution = unsafe { &*self.ptr.cast() };

        if data.0.interval_start == 0 {
            None
        } else {
            // Safety:
            // - Pointer can be incremented because at least the last element must have
            // `interval_start` equal to 0.
            self.ptr = unsafe { self.ptr.add(1) };
            Some(data)
        }
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct BasePairEnergyContribution(pub vrna_sc_bp_storage_t);

impl Deref for BasePairEnergyContribution {
    type Target = vrna_sc_bp_storage_t;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<vrna_sc_bp_storage_t> for BasePairEnergyContribution {
    #[inline]
    fn as_ref(&self) -> &vrna_sc_bp_storage_t {
        &self.0
    }
}

impl PartialEq for BasePairEnergyContribution {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (self.0.interval_start == other.0.interval_start)
            & (self.0.interval_end == other.0.interval_end)
            & (self.0.e == other.0.e)
    }
}

impl Eq for BasePairEnergyContribution {}

/// # Safety
///
/// - The pointer must not outlive the slice
/// - Nothing must be changed through the pointers. The fact that the outer pointer is mutable is
///   only because of C interfaces. If the pointer is passed to a function that modifies the
///   content, then it is Undefined Behavior.
#[cfg(test)]
#[inline]
unsafe fn vrna_md_t_from_model_details(model_details: Option<&ModelDetails>) -> *mut vrna_md_t {
    unsafe { mem::transmute(model_details) }
}

impl fmt::Debug for FoldCompound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let split = self.split().into_checked();
        f.debug_struct("FoldCompound")
            .field("common", &split.common)
            .field("inner", &split.inner)
            .finish()
    }
}

impl Drop for FoldCompound {
    fn drop(&mut self) {
        // Safety: the pointer is not null, it is valid and it is not freed
        unsafe { vrna_fold_compound_free(self.inner.as_mut()) }
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum FoldCompoundError {
    NoSequences,
    UnequalSequences,
    Vienna,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FoldCompoundOptions: u32 {
        const DEFAULT = VRNA_OPTION_DEFAULT;
        const MFE = VRNA_OPTION_MFE;
        const PF = VRNA_OPTION_PF;
        const WINDOW = VRNA_OPTION_WINDOW;
    }
}

bitflags! {
    // Defined in soft.c
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SoftConstraintState: c_uchar {
        const CLEAN = 0;
        const DIRTY_UP_MFE = 1;
        const DIRTY_UP_PF = 2;
        const DIRTY_BP_MFE = 4;
        const DIRTY_BP_PF = 8;
    }
}

#[derive(Clone, Copy)]
pub struct HardConstraints<'a> {
    inner: &'a vrna_hc_t,
    fold_compound: &'a FoldCompoundCommon<'a>,
}

impl<'a> HardConstraints<'a> {
    pub fn ty(&self) -> HardConstraintsType {
        if self.inner.type_ == vrna_hc_type_e_VRNA_HC_DEFAULT {
            HardConstraintsType::Default
        } else if self.inner.type_ == vrna_hc_type_e_VRNA_HC_WINDOW {
            HardConstraintsType::Window
        } else {
            unreachable!()
        }
    }

    pub fn to_enum(self) -> HardConstraintsEnum<'a> {
        match self.ty() {
            HardConstraintsType::Default => {
                // Safety:
                // the variant of the union containing `mx` is acive only when `type` is `DEFAULT`.
                let mx = unsafe { self.inner.__bindgen_anon_1.__bindgen_anon_1.mx };
                HardConstraintsEnum::Default(HardConstraintsDefault { inner: self, mx })
            }
            HardConstraintsType::Window => {
                // Safety:
                // the variant of the union containing `matrix_local` is acive only when `type` is
                // `WINDOW`.
                let matrix_local =
                    unsafe { self.inner.__bindgen_anon_1.__bindgen_anon_2.matrix_local };
                HardConstraintsEnum::Window(HardConstraintsWindow {
                    inner: self,
                    matrix_local,
                })
            }
        }
    }

    #[inline]
    pub fn n(&self) -> c_uint {
        self.inner.n
    }

    pub fn up_ext(&self) -> Option<&[c_int]> {
        self.inner.up_ext.is_null().not().then(|| {
            let len = usize::try_from(self.inner.n.checked_add(2).unwrap()).unwrap();

            // Safety:
            // - `up_ext` is not null (just checked).
            // - Allocated slice has length of `n + 2`
            unsafe { slice::from_raw_parts(self.inner.up_ext, len) }
        })
    }

    pub fn up_hp(&self) -> Option<&[c_int]> {
        self.inner.up_hp.is_null().not().then(|| {
            let len = usize::try_from(self.inner.n.checked_add(2).unwrap()).unwrap();

            // Safety:
            // - `up_hp` is not null (just checked).
            // - Allocated slice has length of `n + 2`
            unsafe { slice::from_raw_parts(self.inner.up_hp, len) }
        })
    }

    pub fn up_int(&self) -> Option<&[c_int]> {
        self.inner.up_int.is_null().not().then(|| {
            let len = usize::try_from(self.inner.n.checked_add(2).unwrap()).unwrap();

            // Safety:
            // - `up_int` is not null (just checked).
            // - Allocated slice has length of `n + 2`
            unsafe { slice::from_raw_parts(self.inner.up_int, len) }
        })
    }

    pub fn up_ml(&self) -> Option<&[c_int]> {
        self.inner.up_ml.is_null().not().then(|| {
            let len = usize::try_from(self.inner.n.checked_add(2).unwrap()).unwrap();

            // Safety:
            // - `up_ml` is not null (just checked).
            // - Allocated slice has length of `n + 2`
            unsafe { slice::from_raw_parts(self.inner.up_ml, len) }
        })
    }

    pub fn depot(&self) -> Option<&'a HardConstraintDepot> {
        self.inner.depot.is_null().not().then(|| {
            // Safety:
            // - `depot` is not null (just checked).
            // - `HardConstraintDepot` is transparent over `vrna_hc_depot_t`.
            unsafe { &*self.inner.depot.cast() }
        })
    }

    fn partial_eq_common(&self, other: &Self) -> bool {
        (self.inner.type_ == other.inner.type_)
            & (self.inner.n == other.inner.n)
            & (self.inner.state == other.inner.state)
            & (self.up_ext() == other.up_ext())
            & (self.up_hp() == other.up_hp())
            & (self.up_int() == other.up_int())
            & (self.up_ml() == other.up_ml())
            & (self.depot() == other.depot())
    }
}

impl PartialEq for HardConstraints<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_enum() == other.to_enum()
    }
}

impl fmt::Debug for HardConstraints<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardConstraints")
            .field("type", &self.to_enum())
            .field("n", &self.inner.n)
            .field("state", &self.inner.state)
            .field("up_ext", &self.up_ext())
            .field("up_hp", &self.up_hp())
            .field("up_int", &self.up_int())
            .field("up_ml", &self.up_ml())
            .field("depot", &self.depot())
            .finish()
    }
}

#[derive(Debug, PartialEq)]
pub enum HardConstraintsEnum<'a> {
    Default(HardConstraintsDefault<'a>),
    Window(HardConstraintsWindow<'a>),
}

pub struct HardConstraintsDefault<'a> {
    inner: HardConstraints<'a>,
    mx: *mut c_uchar,
}

impl HardConstraintsDefault<'_> {
    pub fn mx(&self) -> &[c_uchar] {
        debug_assert!(self.mx.is_null().not());
        let len = self
            .inner
            .fold_compound
            .length
            .checked_add(1)
            .unwrap()
            .checked_pow(2)
            .unwrap()
            .checked_add(1)
            .unwrap()
            .try_into()
            .unwrap();

        // Safety:
        // - `mx` is always allocated when type is Default.
        // - allocated slice length is `(n + 1) * (n + 1) + 1`, where `n` is `fc->length`.
        unsafe { slice::from_raw_parts(self.mx, len) }
    }
}

impl PartialEq for HardConstraintsDefault<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.partial_eq_common(&other.inner) & (self.mx() == other.mx())
    }
}

impl fmt::Debug for HardConstraintsDefault<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardConstraintsDefault")
            .field("mx", &self.mx())
            .finish()
    }
}

pub struct HardConstraintsWindow<'a> {
    inner: HardConstraints<'a>,
    matrix_local: *mut *mut c_uchar,
}

impl HardConstraintsWindow<'_> {
    pub fn matrix_local(&self) -> HardConstraintsMatrixLocal<'_> {
        debug_assert!(self.matrix_local.is_null().not());
        let len = usize::try_from(self.inner.n().checked_add(2).unwrap()).unwrap();

        // Safety:
        // - `matrix_local` is always allocated.
        // - allocated slice is `n + 2`.
        let slice = unsafe { slice::from_raw_parts(self.matrix_local, len) };
        HardConstraintsMatrixLocal {
            inner: slice,
            fold_compound: self.inner.fold_compound,
        }
    }
}

impl PartialEq for HardConstraintsWindow<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.partial_eq_common(&other.inner) & (self.matrix_local() == other.matrix_local())
    }
}

impl fmt::Debug for HardConstraintsWindow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardConstraintsWindow")
            .field("matrix_local", &self.matrix_local())
            .finish()
    }
}

#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
#[repr(u32)]
pub enum HardConstraintsType {
    #[default]
    Default = vrna_hc_type_e_VRNA_HC_DEFAULT,
    Window = vrna_hc_type_e_VRNA_HC_WINDOW,
}

pub struct HardConstraintsMatrixLocal<'a> {
    inner: &'a [*mut c_uchar],
    fold_compound: &'a FoldCompoundCommon<'a>,
}

impl<'a> HardConstraintsMatrixLocal<'a> {
    pub fn get(&self, index: usize) -> OptionalSliceElement<&[c_uchar]> {
        self.inner
            .get(index)
            .copied()
            .map_or(OptionalSliceElement::NoSlice, |ptr| {
                // Safety:
                // - `ptr` comes from the slice of `local_matrix`.
                OptionalSliceElement::from_element(unsafe {
                    Self::slice_from_ptr(ptr, self.fold_compound)
                })
            })
    }

    #[inline]
    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        <Self as IntoIterator>::into_iter(self)
    }

    #[inline]
    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        <&Self as IntoIterator>::into_iter(self)
    }

    /// # Safety
    /// - `ptr` must have the same lifetime as `fold_compound`.
    /// - `ptr` must point inside to one element of `local_matrix` for a hard constraints structure
    ///   with the "window" type.
    unsafe fn slice_from_ptr(
        ptr: *mut c_uchar,
        fold_compound: &'a FoldCompoundCommon<'a>,
    ) -> Option<&'a [c_uchar]> {
        ptr.is_null().not().then(|| {
            let len = usize::try_from(fold_compound.length.checked_add(5).unwrap())
                .unwrap()
                .min(
                    fold_compound
                        .window_size
                        .checked_add(5)
                        .unwrap()
                        .try_into()
                        .unwrap(),
                );

            // Safety:
            // - `ptr` is not null (just checked).
            // - length of the slice is `MIN2(fc->window_size, fc->length) + 5`.
            // - lifetime of `ptr` must outlive `fold_compound` because of the function
            //   invariants.
            unsafe { slice::from_raw_parts(ptr, len) }
        })
    }
}

impl<'a> PartialEq for HardConstraintsMatrixLocal<'a> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len() && self.inner.iter().eq(other.inner)
    }
}

impl fmt::Debug for HardConstraintsMatrixLocal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for HardConstraintsMatrixLocal<'a> {
    type Item = Option<&'a [c_uchar]>;
    type IntoIter = HardConstraintsMatrixLocalIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let inner = self.inner.iter();
        HardConstraintsMatrixLocalIter {
            inner,
            fold_compound: self.fold_compound,
        }
    }
}

impl<'a> IntoIterator for &'a HardConstraintsMatrixLocal<'a> {
    type Item = Option<&'a [c_uchar]>;
    type IntoIter = HardConstraintsMatrixLocalIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let inner = self.inner.iter();
        HardConstraintsMatrixLocalIter {
            inner,
            fold_compound: self.fold_compound,
        }
    }
}

pub struct HardConstraintsMatrixLocalIter<'a> {
    inner: slice::Iter<'a, *mut c_uchar>,
    fold_compound: &'a FoldCompoundCommon<'a>,
}

impl<'a> Iterator for HardConstraintsMatrixLocalIter<'a> {
    type Item = Option<&'a [c_uchar]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().copied().map(|ptr| {
            // Safety:
            // - `ptr` comes from the slice of `local_matrix`.
            unsafe { HardConstraintsMatrixLocal::slice_from_ptr(ptr, self.fold_compound) }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[repr(transparent)]
pub struct HardConstraintDepot(vrna_hc_depot_t);

impl HardConstraintDepot {
    pub fn strands(&self) -> u32 {
        self.0.strands
    }

    pub fn up_size(&self) -> Option<&[c_size_t]> {
        (self.0.strands != 0).then(||
            // Safety:
            // - when strands is not zero, up_size is allocated by strands elements (as in
            //   hc_depot.inc)
            unsafe {
                std::slice::from_raw_parts(self.0.up_size, usize::try_from(self.0.strands).unwrap())
            })
    }

    pub fn up(&self) -> Option<HardConstraintNucs<'_>> {
        (self.0.strands != 0).then(|| {
            // Safety:
            // - when strands is not zero, up is allocated by strands elements (as in hc_depot.inc)
            let nucs = unsafe {
                std::slice::from_raw_parts(self.0.up, usize::try_from(self.0.strands).unwrap())
            };

            HardConstraintNucs {
                inner: nucs,
                depot: self,
            }
        })
    }

    pub fn bp_size(&self) -> Option<&[c_size_t]> {
        (self.0.strands != 0).then(||
            // Safety:
            // - when strands is not zero, bp_size is allocated by strands elements (as in
            //   hc_depot.inc)
            unsafe {
                std::slice::from_raw_parts(self.0.bp_size, usize::try_from(self.0.strands).unwrap())
            })
    }

    pub fn bp(&self) -> Option<HardConstraintBasePairs<'_>> {
        (self.0.strands != 0).then(|| {
            // Safety:
            // - when strands is not zero, bp is allocated by strands elements (as in hc_depot.inc)
            let bps = unsafe {
                std::slice::from_raw_parts(self.0.bp, usize::try_from(self.0.strands).unwrap())
            };

            HardConstraintBasePairs {
                inner: bps,
                depot: self,
            }
        })
    }
}

impl PartialEq for HardConstraintDepot {
    fn eq(&self, other: &Self) -> bool {
        (self.strands() == other.strands())
            & (self.up_size() == other.up_size())
            & (self.up() == other.up())
            & (self.bp_size() == other.bp_size())
            & (self.bp() == other.bp())
    }
}

impl fmt::Debug for HardConstraintDepot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardConstraintDepot")
            .field("strands", &self.strands())
            .field("up_size", &self.up_size())
            .field("up", &self.up())
            .field("bp_size", &self.bp_size())
            .field("bp", &self.bp())
            .finish()
    }
}

pub struct HardConstraintNucs<'a> {
    inner: &'a [*mut hc_nuc],
    depot: &'a HardConstraintDepot,
}

impl<'a> HardConstraintNucs<'a> {
    pub fn get(&self, index: usize) -> OptionalSliceElement<&[HardConstraintNuc]> {
        self.inner
            .get(index)
            .map_or(OptionalSliceElement::NoSlice, |&ptr| {
                let element = ptr.is_null().not().then(|| {
                    // Safety
                    // - ptr is not null (just checked).
                    // - up[index] is an allocated slice of up_size[index] + 1 elements.
                    unsafe { Self::from_ptr_and_size(ptr, self.depot.up_size().unwrap()[index]) }
                });

                OptionalSliceElement::from_element(element)
            })
    }

    /// # Safety
    /// - ptr must be not null and must point inside the `up` field of depot.
    /// - size must be taken from the relative `up_size` item for ptr.
    unsafe fn from_ptr_and_size(ptr: *mut hc_nuc, size: c_size_t) -> &'a [HardConstraintNuc] {
        // Safety
        // - ptr is not null (because of safety preconditions).
        // - up[index] is an allocated slice of up_size[index] + 1 elements.
        // - `HardContraintNuc` is transparent over `hc_nuc`.
        unsafe {
            std::slice::from_raw_parts(
                ptr.cast::<HardConstraintNuc>(),
                size.checked_add(1).unwrap(),
            )
        }
    }

    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }

    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }
}

impl PartialEq for HardConstraintNucs<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len() && self.iter().eq(other.iter())
    }
}

impl fmt::Debug for HardConstraintNucs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for HardConstraintNucs<'a> {
    type Item = Option<&'a [HardConstraintNuc]>;
    type IntoIter = HardConstraintNucsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let inner = self.inner.iter();
        let sizes = self.depot.up_size().unwrap().iter();

        HardConstraintNucsIter { inner, sizes }
    }
}

impl<'a> IntoIterator for &'a HardConstraintNucs<'a> {
    type Item = Option<&'a [HardConstraintNuc]>;
    type IntoIter = HardConstraintNucsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let inner = self.inner.iter();
        let sizes = self.depot.up_size().unwrap().iter();

        HardConstraintNucsIter { inner, sizes }
    }
}

pub struct HardConstraintNucsIter<'a> {
    inner: slice::Iter<'a, *mut hc_nuc>,
    sizes: slice::Iter<'a, c_size_t>,
}

impl<'a> Iterator for HardConstraintNucsIter<'a> {
    type Item = Option<&'a [HardConstraintNuc]>;

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = *self.inner.next()?;
        let size = *self.sizes.next().unwrap();

        let item = ptr.is_null().not().then(|| {
            // Safety
            // - ptr is not null (just checked).
            // - up[index] is an allocated slice of up_size[index] + 1 elements, and we are
            //   iterating both up and up_size.
            unsafe { HardConstraintNucs::from_ptr_and_size(ptr, size) }
        });
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> DoubleEndedIterator for HardConstraintNucsIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ptr = *self.inner.next_back()?;
        let size = *self.sizes.next_back().unwrap();

        let item = ptr.is_null().not().then(|| {
            // Safety
            // - ptr is not null (just checked).
            // - up[index] is an allocated slice of up_size[index] + 1 elements, and we are
            //   iterating both up and up_size.
            unsafe { HardConstraintNucs::from_ptr_and_size(ptr, size) }
        });
        Some(item)
    }
}

pub struct HardConstraintBasePairs<'a> {
    inner: &'a [*mut hc_basepair],
    depot: &'a HardConstraintDepot,
}

impl<'a> HardConstraintBasePairs<'a> {
    pub fn get(&self, index: usize) -> OptionalSliceElement<&[HardConstraintBasePair]> {
        self.inner
            .get(index)
            .map_or(OptionalSliceElement::NoSlice, |&ptr| {
                let element = ptr.is_null().not().then(|| {
                    // Safety
                    // - ptr is not null (just checked).
                    // - up[index] is an allocated slice of up_size[index] + 1 elements.
                    unsafe { Self::from_ptr_and_size(ptr, self.depot.up_size().unwrap()[index]) }
                });

                OptionalSliceElement::from_element(element)
            })
    }

    /// # Safety
    /// - ptr must be not null and must point inside the `up` field of depot.
    /// - size must be taken from the relative `up_size` item for ptr.
    unsafe fn from_ptr_and_size(
        ptr: *mut hc_basepair,
        size: c_size_t,
    ) -> &'a [HardConstraintBasePair] {
        // Safety
        // - ptr is not null (because of safety preconditions).
        // - up[index] is an allocated slice of up_size[index] + 1 elements.
        // - `HardContraintBasePair` is transparent over `hc_basepair`.
        unsafe {
            std::slice::from_raw_parts(
                ptr.cast::<HardConstraintBasePair>(),
                size.checked_add(1).unwrap(),
            )
        }
    }

    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }

    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        IntoIterator::into_iter(self)
    }
}

impl PartialEq for HardConstraintBasePairs<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len() && self.iter().eq(other.iter())
    }
}

impl fmt::Debug for HardConstraintBasePairs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for HardConstraintBasePairs<'a> {
    type Item = Option<&'a [HardConstraintBasePair]>;
    type IntoIter = HardConstraintBasePairsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let inner = self.inner.iter();
        let sizes = self.depot.up_size().unwrap().iter();

        HardConstraintBasePairsIter { inner, sizes }
    }
}

impl<'a> IntoIterator for &'a HardConstraintBasePairs<'a> {
    type Item = Option<&'a [HardConstraintBasePair]>;
    type IntoIter = HardConstraintBasePairsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let inner = self.inner.iter();
        let sizes = self.depot.up_size().unwrap().iter();

        HardConstraintBasePairsIter { inner, sizes }
    }
}

pub struct HardConstraintBasePairsIter<'a> {
    inner: slice::Iter<'a, *mut hc_basepair>,
    sizes: slice::Iter<'a, c_size_t>,
}

impl<'a> Iterator for HardConstraintBasePairsIter<'a> {
    type Item = Option<&'a [HardConstraintBasePair]>;

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = *self.inner.next()?;
        let size = *self.sizes.next().unwrap();

        let item = ptr.is_null().not().then(|| {
            // Safety
            // - ptr is not null (just checked).
            // - up[index] is an allocated slice of up_size[index] + 1 elements, and we are
            //   iterating both up and up_size.
            unsafe { HardConstraintBasePairs::from_ptr_and_size(ptr, size) }
        });
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> DoubleEndedIterator for HardConstraintBasePairsIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ptr = *self.inner.next_back()?;
        let size = *self.sizes.next_back().unwrap();

        let item = ptr.is_null().not().then(|| {
            // Safety
            // - ptr is not null (just checked).
            // - up[index] is an allocated slice of up_size[index] + 1 elements, and we are
            //   iterating both up and up_size.
            unsafe { HardConstraintBasePairs::from_ptr_and_size(ptr, size) }
        });
        Some(item)
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct HardConstraintNuc(hc_nuc);

impl PartialEq for HardConstraintNuc {
    fn eq(&self, other: &Self) -> bool {
        (self.0.direction == other.0.direction)
            & (self.0.context == other.0.context)
            & (self.0.nonspec == other.0.nonspec)
    }
}

#[repr(transparent)]
pub struct HardConstraintBasePair(hc_basepair);

impl HardConstraintBasePair {
    #[inline]
    pub fn list_size(&self) -> c_size_t {
        self.0.list_size
    }

    #[inline]
    pub fn list_mem(&self) -> c_size_t {
        self.0.list_mem
    }

    pub fn j(&self) -> Option<&[c_uint]> {
        self.0.j.is_null().not().then(|| {
            // Safety:
            // - ptr is not null (just checked).
            // - allocated slice has a length of list_mem + 1 (as in hc_depot.inc).
            unsafe { std::slice::from_raw_parts(self.0.j, self.0.list_mem.checked_add(1).unwrap()) }
        })
    }

    pub fn strand_j(&self) -> Option<&[c_uint]> {
        self.0.strand_j.is_null().not().then(|| {
            // Safety:
            // - ptr is not null (just checked).
            // - allocated slice has a length of list_mem + 1 (as in hc_depot.inc).
            unsafe {
                std::slice::from_raw_parts(self.0.strand_j, self.0.list_mem.checked_add(1).unwrap())
            }
        })
    }

    pub fn context(&self) -> Option<&[c_uchar]> {
        self.0.context.is_null().not().then(|| {
            // Safety:
            // - ptr is not null (just checked).
            // - allocated slice has a length of list_mem + 1 (as in hc_depot.inc).
            unsafe {
                std::slice::from_raw_parts(self.0.context, self.0.list_mem.checked_add(1).unwrap())
            }
        })
    }
}

impl PartialEq for HardConstraintBasePair {
    fn eq(&self, other: &Self) -> bool {
        (self.list_size() == other.list_size())
            & (self.list_mem() == other.list_mem())
            & (self.j() == other.j())
            & (self.strand_j() == other.strand_j())
            & (self.context() == other.context())
    }
}

impl fmt::Debug for HardConstraintBasePair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardConstraintBasePair")
            .field("list_size", &self.list_size())
            .field("list_mem", &self.list_mem())
            .field("j", &self.j())
            .field("strand_j", &self.strand_j())
            .field("context", &self.context())
            .finish()
    }
}

#[repr(transparent)]
pub struct MinimumFreeEnergyDynamicProgrammingMatrices(vrna_mx_mfe_t);

impl PartialEq for MinimumFreeEnergyDynamicProgrammingMatrices {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl fmt::Debug for MinimumFreeEnergyDynamicProgrammingMatrices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MinimumFreeEnergyDynamicProgrammingMatrices")
            .field(&"[UNIMPLEMENTED DEBUG]")
            .finish()
    }
}

#[repr(transparent)]
pub struct PartitionFunctionDynamicProgrammingMatrices(vrna_mx_pf_t);

impl PartialEq for PartitionFunctionDynamicProgrammingMatrices {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl fmt::Debug for PartitionFunctionDynamicProgrammingMatrices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PartitionFunctionDynamicProgrammingMatrices")
            .field(&"[UNIMPLEMENTED DEBUG]")
            .finish()
    }
}

#[repr(transparent)]
pub struct FreeEnergyParameters(vrna_param_t);

impl FreeEnergyParameters {
    #[cfg(test)]
    #[inline]
    pub fn id_mut(&mut self) -> &mut i32 {
        &mut self.0.id
    }
}

impl PartialEq for FreeEnergyParameters {
    fn eq(&self, other: &Self) -> bool {
        (self.0.id == other.0.id)
            & (self.0.stack == other.0.stack)
            & (self.0.hairpin == other.0.hairpin)
            & (self.0.bulge == other.0.bulge)
            & (self.0.internal_loop == other.0.internal_loop)
            & (self.0.mismatchExt == other.0.mismatchExt)
            & (self.0.mismatchI == other.0.mismatchI)
            & (self.0.mismatch1nI == other.0.mismatch1nI)
            & (self.0.mismatch23I == other.0.mismatch23I)
            & (self.0.mismatchH == other.0.mismatchH)
            & (self.0.mismatchM == other.0.mismatchM)
            & (self.0.dangle5 == other.0.dangle5)
            & (self.0.dangle3 == other.0.dangle3)
            & (self.0.int11 == other.0.int11)
            & (self.0.int21 == other.0.int21)
            & (self.0.int22 == other.0.int22)
            & (self.0.ninio == other.0.ninio)
            & (self.0.lxc == other.0.lxc)
            & (self.0.MLbase == other.0.MLbase)
            & (self.0.MLintern == other.0.MLintern)
            & (self.0.MLclosing == other.0.MLclosing)
            & (self.0.TerminalAU == other.0.TerminalAU)
            & (self.0.DuplexInit == other.0.DuplexInit)
            & (self.0.Tetraloop_E == other.0.Tetraloop_E)
            & (self.0.Tetraloops == other.0.Tetraloops)
            & (self.0.Triloop_E == other.0.Triloop_E)
            & (self.0.Triloops == other.0.Triloops)
            & (self.0.Hexaloop_E == other.0.Hexaloop_E)
            & (self.0.Hexaloops == other.0.Hexaloops)
            & (self.0.TripleC == other.0.TripleC)
            & (self.0.MultipleCA == other.0.MultipleCA)
            & (self.0.MultipleCB == other.0.MultipleCB)
            & (self.0.gquad == other.0.gquad)
            & (self.0.gquadLayerMismatch == other.0.gquadLayerMismatch)
            & (self.0.gquadLayerMismatchMax == other.0.gquadLayerMismatchMax)
            & (self.0.temperature == other.0.temperature)
            & (ModelDetails::from_ref(&self.0.model_details)
                == ModelDetails::from_ref(&other.0.model_details))
            & (self.0.param_file == other.0.param_file)
    }
}

impl fmt::Debug for FreeEnergyParameters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeEnergyParameters")
            .field("id", &self.0.id)
            .field("stack", &self.0.stack)
            .field("hairpin", &self.0.hairpin)
            .field("bulge", &self.0.bulge)
            .field("internal_loop", &self.0.internal_loop)
            .field("mismatchExt", &self.0.mismatchExt)
            .field("mismatchI", &self.0.mismatchI)
            .field("mismatch1nI", &self.0.mismatch1nI)
            .field("mismatch23I", &self.0.mismatch23I)
            .field("mismatchH", &self.0.mismatchH)
            .field("mismatchM", &self.0.mismatchM)
            .field("dangle5", &self.0.dangle5)
            .field("dangle3", &self.0.dangle3)
            .field("int11", &self.0.int11)
            .field("int21", &self.0.int21)
            .field("int22", &self.0.int22)
            .field("ninio", &self.0.ninio)
            .field("lxc", &self.0.lxc)
            .field("MLbase", &self.0.MLbase)
            .field("MLintern", &self.0.MLintern)
            .field("MLclosing", &self.0.MLclosing)
            .field("TerminalAU", &self.0.TerminalAU)
            .field("DuplexInit", &self.0.DuplexInit)
            .field("Tetraloop_E", &self.0.Tetraloop_E)
            .field("Tetraloops", &self.0.Tetraloops)
            .field("Triloop_E", &self.0.Triloop_E)
            .field("Triloops", &self.0.Triloops)
            .field("Hexaloop_E", &self.0.Hexaloop_E)
            .field("Hexaloops", &self.0.Hexaloops)
            .field("TripleC", &self.0.TripleC)
            .field("MultipleCA", &self.0.MultipleCA)
            .field("MultipleCB", &self.0.MultipleCB)
            .field("gquad", &self.0.gquad)
            .field("gquadLayerMismatch", &self.0.gquadLayerMismatch)
            .field("gquadLayerMismatchMax", &self.0.gquadLayerMismatchMax)
            .field("temperature", &self.0.temperature)
            .field(
                "model_details",
                ModelDetails::from_ref(&self.0.model_details),
            )
            .field("param_file", &self.0.param_file)
            .finish()
    }
}

#[repr(transparent)]
pub struct BoltzmannFactor(vrna_exp_param_t);

impl PartialEq for BoltzmannFactor {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl fmt::Debug for BoltzmannFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BoltzmannFactor")
            .field(&"[UNIMPLEMENTED DEBUG]")
            .finish()
    }
}

#[repr(transparent)]
pub struct UnstructuredDomain(vrna_ud_t);

impl PartialEq for UnstructuredDomain {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl fmt::Debug for UnstructuredDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("UnstructuredDomain")
            .field(&"[UNIMPLEMENTED DEBUG]")
            .finish()
    }
}

#[derive(Debug)]
#[allow(unused)]
pub struct PairTypes<'a> {
    inner: &'a [*mut c_char],
    len: usize,
}

impl PartialEq for PairTypes<'_> {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

#[repr(transparent)]
pub struct Sequence(pub vrna_seq_t);

#[allow(unused)]
#[repr(u32)]
pub enum SequenceType {
    Unknown = vrna_seq_type_e_VRNA_SEQ_UNKNOWN,
    Rna = vrna_seq_type_e_VRNA_SEQ_RNA,
    Dna = vrna_seq_type_e_VRNA_SEQ_DNA,
}

impl Sequence {
    #[allow(unused)]
    pub fn ty(&self) -> SequenceType {
        if self.type_ == vrna_seq_type_e_VRNA_SEQ_UNKNOWN {
            SequenceType::Unknown
        } else if self.type_ == vrna_seq_type_e_VRNA_SEQ_RNA {
            SequenceType::Rna
        } else if self.type_ == vrna_seq_type_e_VRNA_SEQ_DNA {
            SequenceType::Dna
        } else {
            unreachable!()
        }
    }

    pub fn name(&self) -> Option<&CStr> {
        self.name.is_null().not().then(|| {
            // Safety:
            // - ptr is not null (just checked)
            // - `name` can be a null pointer or a C string (see sequence.c).
            unsafe { CStr::from_ptr(self.name) }
        })
    }

    pub fn string(&self) -> Option<&CStr> {
        self.string.is_null().not().then(|| {
            // Safety:
            // - ptr is not null (just checked)
            // - `string` is a C string (see sequence.c).
            unsafe { CStr::from_ptr(self.string) }
        })
    }

    pub fn encoding(&self) -> Option<&[c_short]> {
        self.encoding.is_null().not().then(|| {
            let len = usize::try_from(self.length.checked_add(2).unwrap()).unwrap();

            // Safety:
            // - ptr is not null (just checked)
            // - `encoding` is an allocated slice of `length + 2` C shorts (as in sequence.c and
            //   alphabet.c), where `length` is the inner field, which in turn is the length of
            //   `string`.
            unsafe { slice::from_raw_parts(self.encoding, len) }
        })
    }

    pub fn encoding5(&self) -> Option<&[c_short]> {
        self.encoding5.is_null().not().then(|| {
            let len = usize::try_from(self.length.checked_add(1).unwrap()).unwrap();

            // Safety:
            // - ptr is not null (just checked)
            // - `encoding5` is an allocated slice of `length + 1` C shorts (as in sequence.c),
            //   where `length` is the inner field, which in turn is the length of `string`.
            unsafe { slice::from_raw_parts(self.encoding5, len) }
        })
    }

    pub fn encoding3(&self) -> Option<&[c_short]> {
        self.encoding3.is_null().not().then(|| {
            let len = usize::try_from(self.length.checked_add(1).unwrap()).unwrap();

            // Safety:
            // - ptr is not null (just checked)
            // - `encoding3` is an allocated slice of `length + 1` C shorts (as in sequence.c),
            //   where `length` is the inner field, which in turn is the length of `string`.
            unsafe { slice::from_raw_parts(self.encoding3, len) }
        })
    }

    #[inline]
    pub fn length(&self) -> c_uint {
        self.0.length
    }
}

impl PartialEq for Sequence {
    fn eq(&self, other: &Self) -> bool {
        (self.0.type_ == other.0.type_)
            & (self.name() == other.name())
            & (self.string() == other.string())
            & (self.encoding() == other.encoding())
            & (self.encoding5() == other.encoding5())
            & (self.encoding3() == other.encoding3())
            & (self.0.length == other.0.length)
    }
}

impl Eq for Sequence {}

impl Deref for Sequence {
    type Target = vrna_seq_t;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<vrna_seq_t> for Sequence {
    fn as_ref(&self) -> &vrna_seq_t {
        &self.0
    }
}

impl fmt::Debug for Sequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequence")
            .field("type_", &self.0.type_)
            .field("name", &self.name())
            .field("string", &self.string())
            .field("encoding", &self.encoding())
            .field("encoding5", &self.encoding5())
            .field("encoding3", &self.encoding3())
            .field("length", &self.0.length)
            .finish()
    }
}

#[repr(transparent)]
pub struct Alignment(pub vrna_msa_t);

impl Alignment {
    pub fn sequences(&self) -> &[Sequence] {
        // Note: this is based on `sequence.c`
        assert!(self.sequences.is_null().not());

        // Safety:
        // - pointer is not null (just checked).
        // - the allocation contains `n_seq` sequences (as in sequence.c).
        // - `Sequence` is transparent over `vrna_seq_t`.
        unsafe {
            slice::from_raw_parts(self.sequences.cast(), usize::try_from(self.n_seq).unwrap())
        }
    }

    pub fn gapfree_seq(&self) -> Option<OptionCStrSlice<'_>> {
        self.0.gapfree_seq.is_null().not().then(|| {
            // Safety:
            // - `gapfree_seq` is not null (just checked).
            // - We are taking a non mutable reference to self, no changes to the underlying data
            //   are allowed.
            // - `gapfree_seq` is a slice of `n_seq` possibly null C strings (as in `sequence.c`).
            // - The lifetime of the created structure is relative to `self`.
            unsafe {
                let slice = slice::from_raw_parts(
                    self.0.gapfree_seq.cast(),
                    usize::try_from(self.n_seq).unwrap(),
                );
                OptionCStrSlice::from_slice(slice)
            }
        })
    }

    pub fn gapfree_size(&self) -> Option<&[c_uint]> {
        self.0.gapfree_size.is_null().not().then(|| {
            // Safety:
            // - `gapfree_size` is not null (just checked).
            // - allocated slice has a len equal to `n_seq` (as in `sequence.c`).
            unsafe {
                slice::from_raw_parts(self.gapfree_size, usize::try_from(self.n_seq).unwrap())
            }
        })
    }

    pub fn genome_size(&self) -> Option<&[c_ulonglong]> {
        self.0.genome_size.is_null().not().then(|| {
            // Safety:
            // - `genome_size` is not null (just checked).
            // - allocated slice has a len equal to `n_seq` (as in `sequence.c`).
            unsafe { slice::from_raw_parts(self.genome_size, usize::try_from(self.n_seq).unwrap()) }
        })
    }

    pub fn start(&self) -> Option<&[c_ulonglong]> {
        self.0.start.is_null().not().then(|| {
            // Safety:
            // - `start` is not null (just checked).
            // - allocated slice has a len equal to `n_seq` (as in `sequence.c`).
            unsafe { slice::from_raw_parts(self.start, usize::try_from(self.n_seq).unwrap()) }
        })
    }

    pub fn orientation(&self) -> Option<&[c_uchar]> {
        self.0.orientation.is_null().not().then(|| {
            // Safety:
            // - `orientation` is not null (just checked).
            // - allocated slice has a len equal to `n_seq` (as in `sequence.c`).
            unsafe { slice::from_raw_parts(self.orientation, usize::try_from(self.n_seq).unwrap()) }
        })
    }

    pub fn a2s(&self) -> Option<AlignmentA2S<'_>> {
        self.0.a2s.is_null().not().then(|| {
            // - `a2s` is not null (just checked).
            // - allocated slice has a len equal to `n_seq` (as in `sequence.c`).
            // - each pointer inside `a2s` is not null (checked inside the block), it is safe to
            //   transmute for a slice of pointers to a slice of `NonNull`.
            // - We are using the lifetime of self.
            let a2s = unsafe {
                let slice = slice::from_raw_parts(self.0.a2s, usize::try_from(self.n_seq).unwrap());
                assert!(slice.iter().all(|ptr| ptr.is_null().not()));
                &*(slice as *const _ as *const [std::ptr::NonNull<u32>])
            };

            let sequences = self.sequences();
            AlignmentA2S { a2s, sequences }
        })
    }
}

impl PartialEq for Alignment {
    fn eq(&self, other: &Self) -> bool {
        (self.sequences() == other.sequences())
            & (self.gapfree_seq() == other.gapfree_seq())
            & (self.gapfree_size() == other.gapfree_size())
            & (self.genome_size() == other.genome_size())
            & (self.start() == other.start())
            & (self.orientation() == other.orientation())
            & (self.a2s() == other.a2s())
    }
}

impl Eq for Alignment {}

impl fmt::Debug for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Alignment")
            .field("sequences", &self.sequences())
            .field("gapfree_seq", &self.gapfree_seq())
            .field("gapfree_size", &self.gapfree_size())
            .field("genome_size", &self.genome_size())
            .field("start", &self.start())
            .field("orientation", &self.orientation())
            .field("a2s", &self.a2s())
            .finish()
    }
}

pub struct AlignmentA2S<'a> {
    a2s: &'a [NonNull<c_uint>],
    sequences: &'a [Sequence],
}

impl<'a> AlignmentA2S<'a> {
    #[inline]
    pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        <Self as IntoIterator>::into_iter(self)
    }

    #[inline]
    pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl PartialEq for AlignmentA2S<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.a2s.len() == other.a2s.len() && self.iter().eq(other)
    }
}

impl Eq for AlignmentA2S<'_> {}

impl fmt::Debug for AlignmentA2S<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<'a> IntoIterator for AlignmentA2S<'a> {
    type Item = &'a [c_uint];
    type IntoIter = AlignmentA2SIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let a2s = self.a2s.iter();
        let sequences = self.sequences.iter();
        AlignmentA2SIter { a2s, sequences }
    }
}

impl<'a> IntoIterator for &'a AlignmentA2S<'a> {
    type Item = &'a [c_uint];
    type IntoIter = AlignmentA2SIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let a2s = self.a2s.iter();
        let sequences = self.sequences.iter();
        AlignmentA2SIter { a2s, sequences }
    }
}

pub struct AlignmentA2SIter<'a> {
    a2s: slice::Iter<'a, NonNull<c_uint>>,
    sequences: slice::Iter<'a, Sequence>,
}

impl<'a> Iterator for AlignmentA2SIter<'a> {
    type Item = &'a [c_uint];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.a2s.next().map(|ptr| {
            let sequence = self.sequences.next().unwrap();
            let len = usize::try_from(sequence.length().checked_add(1).unwrap()).unwrap();

            // Safety:
            // - `ptr` is not null (because of `NonNull`).
            // - length of the slice is the relative sequence length + 1 (as in `sequence.c`).
            unsafe { slice::from_raw_parts(ptr.as_ptr(), len) }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.a2s.size_hint()
    }
}

impl Deref for Alignment {
    type Target = vrna_msa_t;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<vrna_msa_t> for Alignment {
    fn as_ref(&self) -> &vrna_msa_t {
        &self.0
    }
}

mod inner {
    use std::{
        ffi::{c_char, CStr},
        fmt,
        ops::{Index, Not},
        ptr::NonNull,
        slice,
    };

    use super::OptionalSliceElement;

    #[derive(Clone, Copy)]
    pub struct OptionCStrSlice<'a>(&'a [*const c_char]);

    impl<'a> OptionCStrSlice<'a> {
        /// Create a new instance for a slice of pointers.
        ///
        /// # Safety
        /// - Each of the pointers must be either null or a valid null-terminated C string.
        /// - The valid pointed strings must outlive the slice and, by consequence, the new
        ///   instance.
        /// - The pointed strings are not allowed to change during the lifetime of the new
        ///   instance.
        pub unsafe fn from_slice(slice: &'a [*const c_char]) -> Self {
            Self(slice)
        }

        #[inline]
        pub fn get(&self, index: usize) -> OptionalSliceElement<&'a CStr> {
            self.0
                .get(index)
                .copied()
                .map_or(OptionalSliceElement::NoSlice, |ptr| {
                    let element = ptr.is_null().not().then(|| {
                        unsafe {
                            // Safety:
                            // - `ptr` is not null (just checked).
                            // - The validity of `ptr` being a null terminated C string is guaranteed by the
                            //   invariances of the struct, for which the builder is responsible.
                            CStr::from_ptr(ptr)
                        }
                    });

                    OptionalSliceElement::from_element(element)
                })
        }

        #[inline]
        pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
            <Self as IntoIterator>::into_iter(self)
        }

        #[inline]
        pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
            <&Self as IntoIterator>::into_iter(self)
        }
    }

    impl PartialEq for OptionCStrSlice<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.0.len() == other.0.len() && self.iter().eq(other)
        }
    }

    impl Eq for OptionCStrSlice<'_> {}

    impl fmt::Debug for OptionCStrSlice<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_list().entries(self.iter()).finish()
        }
    }

    impl<'a> IntoIterator for OptionCStrSlice<'a> {
        type Item = Option<&'a CStr>;
        type IntoIter = OptionCStrSliceIter<'a>;

        #[inline]
        fn into_iter(self) -> Self::IntoIter {
            OptionCStrSliceIter(self.0.iter())
        }
    }

    impl<'a> IntoIterator for &'a OptionCStrSlice<'a> {
        type Item = Option<&'a CStr>;
        type IntoIter = OptionCStrSliceIter<'a>;

        #[inline]
        fn into_iter(self) -> Self::IntoIter {
            OptionCStrSliceIter(self.0.iter())
        }
    }

    pub struct OptionCStrSliceIter<'a>(slice::Iter<'a, *const c_char>);

    impl<'a> Iterator for OptionCStrSliceIter<'a> {
        type Item = Option<&'a CStr>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().copied().map(|ptr| {
                ptr.is_null().not().then(|| {
                    unsafe {
                        // Safety:
                        // - `ptr` is not null (just checked).
                        // - The validity of `ptr` being a null terminated C string is guaranteed by the
                        //   invariances of the struct, for which the builder is responsible.
                        CStr::from_ptr(ptr)
                    }
                })
            })
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    impl<'a> DoubleEndedIterator for OptionCStrSliceIter<'a> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            self.0.next_back().copied().map(|ptr| {
                ptr.is_null().not().then(|| {
                    unsafe {
                        // Safety:
                        // - `ptr` is not null (just checked).
                        // - The validity of `ptr` being a null terminated C string is guaranteed by the
                        //   invariances of the struct, for which the builder is responsible.
                        CStr::from_ptr(ptr)
                    }
                })
            })
        }
    }

    impl<'a> ExactSizeIterator for OptionCStrSliceIter<'a> {
        #[inline]
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    #[derive(Clone, Copy)]
    pub struct CStrSlice<'a>(&'a [NonNull<c_char>]);

    impl<'a> CStrSlice<'a> {
        /// Create a new instance for a slice of pointers.
        ///
        /// # Panics
        /// The function panics if one of the pointers is null.
        ///
        /// # Safety
        /// - Each of the pointers must be a valid null-terminated C string.
        /// - The valid pointed strings must outlive the slice and, by consequence, the new
        ///   instance.
        /// - The pointed strings are not allowed to change during the lifetime of the new
        ///   instance.
        pub unsafe fn from_slice_checked(slice: &'a [*const c_char]) -> Self {
            for ptr in slice {
                assert!(ptr.is_null().not());
            }

            // Safety:
            // - a slice of pointers has the same layout as a slice of NonNull pointers.
            // - we just checked that none of the pointers are null.
            unsafe { Self(&*(slice as *const [*const c_char] as *const [NonNull<c_char>])) }
        }

        #[inline]
        pub fn get(&self, index: usize) -> Option<&'a CStr> {
            self.0.get(index).copied().map(|ptr| {
                unsafe {
                    // Safety:
                    // - `ptr` is not null.
                    // - The validity of `ptr` being a null terminated C string is guaranteed by the
                    //   invariances of the struct, for which the builder is responsible.
                    CStr::from_ptr(ptr.as_ptr())
                }
            })
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        #[inline]
        pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
            <Self as IntoIterator>::into_iter(self)
        }

        #[inline]
        pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
            <&Self as IntoIterator>::into_iter(self)
        }
    }

    impl PartialEq for CStrSlice<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.0.len() == other.0.len() && self.iter().eq(other)
        }
    }

    impl Eq for CStrSlice<'_> {}

    impl fmt::Debug for CStrSlice<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_list().entries(self.iter()).finish()
        }
    }

    impl Index<usize> for CStrSlice<'_> {
        type Output = CStr;

        #[inline]
        fn index(&self, index: usize) -> &Self::Output {
            self.get(index).unwrap()
        }
    }

    impl<'a> IntoIterator for CStrSlice<'a> {
        type Item = &'a CStr;
        type IntoIter = CStrSliceIter<'a>;

        #[inline]
        fn into_iter(self) -> Self::IntoIter {
            CStrSliceIter(self.0.iter())
        }
    }

    impl<'a> IntoIterator for &'a CStrSlice<'a> {
        type Item = &'a CStr;
        type IntoIter = CStrSliceIter<'a>;

        #[inline]
        fn into_iter(self) -> Self::IntoIter {
            CStrSliceIter(self.0.iter())
        }
    }

    pub struct CStrSliceIter<'a>(slice::Iter<'a, NonNull<c_char>>);

    impl<'a> Iterator for CStrSliceIter<'a> {
        type Item = &'a CStr;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().copied().map(|ptr| {
                unsafe {
                    // Safety:
                    // - `ptr` is not null.
                    // - The validity of `ptr` being a null terminated C string is guaranteed by the
                    //   invariances of the struct, for which the builder is responsible.
                    CStr::from_ptr(ptr.as_ptr())
                }
            })
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    impl<'a> DoubleEndedIterator for CStrSliceIter<'a> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            self.0.next_back().copied().map(|ptr| {
                unsafe {
                    // Safety:
                    // - `ptr` is not null.
                    // - The validity of `ptr` being a null terminated C string is guaranteed by the
                    //   invariances of the struct, for which the builder is responsible.
                    CStr::from_ptr(ptr.as_ptr())
                }
            })
        }
    }

    impl<'a> ExactSizeIterator for CStrSliceIter<'a> {
        #[inline]
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    pub struct FoldCompoundSequenceData<'a, T> {
        slice: &'a [*const T],
        sequences: CStrSlice<'a>,
    }

    impl<'a, T> FoldCompoundSequenceData<'a, T> {
        /// # Safety
        /// - slice must have the same length of sequences.
        /// - each pointer in slice must be non-null and it must point to a slice of `n+2`
        ///   elements, where `n` is the length of the sequence with the same index.
        pub unsafe fn new(slice: &'a [*const T], sequences: CStrSlice<'a>) -> Self {
            debug_assert_eq!(slice.len(), sequences.len());
            Self { slice, sequences }
        }

        pub fn get(&self, index: usize) -> Option<&'a [T]> {
            self.slice.get(index).map(|&ptr| {
                let len = self.sequences[index]
                    .to_bytes()
                    .len()
                    .checked_add(2)
                    .unwrap();

                // Safety:
                // - validity of pointer and slice length are guaranteed by struct invariances.
                unsafe { slice::from_raw_parts(ptr, len) }
            })
        }

        #[inline]
        pub fn iter(&'a self) -> <&'a Self as IntoIterator>::IntoIter {
            <&Self as IntoIterator>::into_iter(self)
        }

        #[inline]
        pub fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
            <Self as IntoIterator>::into_iter(self)
        }
    }

    impl<T: fmt::Debug> fmt::Debug for FoldCompoundSequenceData<'_, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_list().entries(self).finish()
        }
    }

    impl<T: PartialEq> PartialEq for FoldCompoundSequenceData<'_, T> {
        fn eq(&self, other: &Self) -> bool {
            self.iter().eq(other)
        }
    }

    impl<T: Eq> Eq for FoldCompoundSequenceData<'_, T> {}

    impl<'a, T> Index<usize> for FoldCompoundSequenceData<'a, T> {
        type Output = [T];

        #[inline]
        fn index(&self, index: usize) -> &Self::Output {
            self.get(index).expect("index out of bound")
        }
    }

    impl<'a, T> IntoIterator for FoldCompoundSequenceData<'a, T> {
        type Item = &'a [T];
        type IntoIter = FoldCompoundSequenceDataIter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            let Self { slice, sequences } = self;
            let inner = slice.iter();
            let sequences = sequences.into_iter();

            FoldCompoundSequenceDataIter { inner, sequences }
        }
    }

    impl<'a, T> IntoIterator for &'a FoldCompoundSequenceData<'a, T> {
        type Item = &'a [T];
        type IntoIter = FoldCompoundSequenceDataIter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            let FoldCompoundSequenceData { slice, sequences } = self;
            let inner = slice.iter();
            let sequences = sequences.iter();

            FoldCompoundSequenceDataIter { inner, sequences }
        }
    }

    pub struct FoldCompoundSequenceDataIter<'a, T> {
        inner: slice::Iter<'a, *const T>,
        sequences: CStrSliceIter<'a>,
    }

    impl<'a, T> Iterator for FoldCompoundSequenceDataIter<'a, T> {
        type Item = &'a [T];

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().map(|&ptr| {
                let len = self
                    .sequences
                    .next()
                    .unwrap()
                    .to_bytes()
                    .len()
                    .checked_add(2)
                    .unwrap();

                // Safety:
                // - validity of pointer and slice length are guaranteed by struct invariances.
                unsafe { slice::from_raw_parts(ptr, len) }
            })
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io, ops::Not, path::Path};

    use tempfile::tempdir;
    use viennarna_mfe_sys::vrna_sc_add_SHAPE_deigan_ali;

    use crate::{
        aligner::AlignedSequence,
        fasta,
        gapped_reactivity::GappedReactivity,
        gapped_sequence::{GappedSequence, StatefulBaseOrGap},
        Sequence,
    };

    use super::*;

    #[test]
    fn miri_vrna_md_t_from_model_details() {
        let model_details = vrna_md_t {
            temperature: Default::default(),
            betaScale: Default::default(),
            pf_smooth: Default::default(),
            dangles: Default::default(),
            special_hp: Default::default(),
            noLP: Default::default(),
            noGU: Default::default(),
            noGUclosure: Default::default(),
            logML: Default::default(),
            circ: Default::default(),
            gquad: Default::default(),
            uniq_ML: Default::default(),
            energy_set: Default::default(),
            backtrack: Default::default(),
            backtrack_type: Default::default(),
            compute_bpp: Default::default(),
            nonstandards: [0; 64],
            max_bp_span: Default::default(),
            min_loop_size: Default::default(),
            window_size: Default::default(),
            oldAliEn: Default::default(),
            ribo: Default::default(),
            cv_fact: Default::default(),
            nc_fact: Default::default(),
            sfact: Default::default(),
            rtype: Default::default(),
            alias: Default::default(),
            pair: Default::default(),
            pair_dist: Default::default(),
        };
        let model_details = ModelDetails(model_details);
        let c_ptr = unsafe { vrna_md_t_from_model_details(Some(&model_details)) };
        assert!(c_ptr.is_null().not());

        let c_ptr = unsafe { vrna_md_t_from_model_details(None) };
        assert!(c_ptr.is_null());
    }

    #[allow(clippy::too_many_lines)]
    #[test]
    fn add_shape_reactivity_for_deigan_consensus_structure_prediction() {
        use std::io::Write;

        const SLOPE: f32 = 0.32;
        const INTERCEPT: f32 = 0.7;

        let sequences = {
            use crate::Base::{A, C, G, T};
            [
                [A, C, G, C, T, C, C, C, C, A, A, T],
                [C, T, G, T, T, T, T, A, T, C, G, G],
            ]
        };
        let sequences = [
            Sequence {
                bases: &sequences[0],
                molecule: Molecule::Rna,
            },
            Sequence {
                bases: &sequences[1],
                molecule: Molecule::Rna,
            },
        ];

        let alignments = {
            use crate::aligner::BaseOrGap::{Base, Gap};
            [
                AlignedSequence(vec![
                    Base, Base, Gap, Base, Base, Base, Base, Gap, Gap, Base, Base, Base, Base,
                    Base, Base,
                ]),
                AlignedSequence(vec![
                    Base, Base, Base, Base, Base, Gap, Gap, Base, Base, Base, Base, Base, Base,
                    Gap, Base,
                ]),
            ]
        };
        let sequences = [
            GappedSequence::new(sequences[0], &alignments[0]),
            GappedSequence::new(sequences[1], &alignments[1]),
        ];
        let reactivities = [
            [
                0.23, 0.15, 0.85, 0.54, 0.72, 0.14, 0.96, 0.86, 0.43, 0.23, 0.01, 0.15,
            ],
            [
                0.65, 0.23, 0.78, 0.29, 0.65, 0.72, 0.12, 0.89, 0.18, 0.84, 0.23, 0.68,
            ],
        ];
        let gapped_reacitvities = [
            GappedReactivity {
                reactivity: reactivities[0].as_slice(),
                alignment: alignments[0].to_ref(),
            },
            GappedReactivity {
                reactivity: reactivities[1].as_slice(),
                alignment: alignments[1].to_ref(),
            },
        ];

        let model_details = ModelDetails::default();
        let create_fold_compound = || {
            FoldCompound::new_comparative(
                &sequences,
                Some(&model_details),
                FoldCompoundOptions::DEFAULT,
            )
            .unwrap()
        };

        let temp_dir = tempdir().expect("cannot create temporary directory");

        let sequences_path = temp_dir.path().join("sequence.fasta");
        let seq1_shape_path = temp_dir.path().join("seq1.shape");
        let seq2_shape_path = temp_dir.path().join("seq2.shape");
        let mut sequence_file = File::create(sequences_path).unwrap();
        write!(
            sequence_file,
            "{}\n{}",
            fasta::Entry {
                description: "seq1",
                sequence: sequences[0].sequence,
                alignment: Some(&alignments[0])
            },
            fasta::Entry {
                description: "seq2",
                sequence: sequences[1].sequence,
                alignment: Some(&alignments[1])
            },
        )
        .unwrap();
        drop(sequence_file);
        create_shape_file(&seq1_shape_path, reactivities[0].as_slice(), sequences[0]).unwrap();
        create_shape_file(&seq2_shape_path, reactivities[1].as_slice(), sequences[1]).unwrap();

        let mut fold_compound = create_fold_compound();
        let shape_files = [
            CString::new(seq1_shape_path.to_str().unwrap()).unwrap(),
            CString::new(seq2_shape_path.to_str().unwrap()).unwrap(),
        ];
        let shape_files = [
            shape_files[0].as_ptr(),
            shape_files[1].as_ptr(),
            ptr::null(),
        ];
        let shape_file_associations = [0, 1, -1];
        let ret = unsafe {
            #[allow(clippy::cast_lossless)]
            vrna_sc_add_SHAPE_deigan_ali(
                fold_compound.as_mut(),
                shape_files.as_ptr().cast_mut().cast(),
                shape_file_associations.as_ptr(),
                SLOPE as FLT_OR_DBL,
                INTERCEPT as FLT_OR_DBL,
                FoldCompoundOptions::DEFAULT.bits(),
            )
        };
        assert_eq!(ret, 1);

        temp_dir
            .close()
            .expect("unable to delete temporary directory");

        let mut fold_compound_impl = create_fold_compound();
        fold_compound_impl
            .add_shape_reactivity_for_deigan_consensus_structure_prediction(
                &gapped_reacitvities,
                SLOPE.into(),
                INTERCEPT.into(),
                FoldCompoundOptions::DEFAULT,
            )
            .unwrap();

        assert!(fold_compound.is_comparative());

        // We need to set the id for the parameters to 0, because ViennaRNA uses a global id which
        // is incremented every time `get_scaled_params` is called.
        if let Some(params) = fold_compound_impl
            .split_mut()
            .into_comparative()
            .unwrap()
            .0
            .params()
        {
            *params.id_mut() = 0;
        }

        assert_eq!(fold_compound, fold_compound_impl);
    }

    fn create_shape_file(
        path: &Path,
        reactivities: &[Reactivity],
        gapped_sequence: GappedSequence,
    ) -> io::Result<()> {
        use std::io::Write;

        assert_eq!(reactivities.len(), gapped_sequence.sequence.bases.len());
        let mut file = File::create(path)?;
        gapped_sequence
            .into_iter()
            .filter_map(StatefulBaseOrGap::to_base)
            .enumerate()
            .zip(reactivities)
            .try_for_each(|((index, base), &reactivity)| {
                writeln!(
                    file,
                    "{} {} {reactivity}",
                    index + 1,
                    char::from(base.to_byte(Molecule::Rna))
                )
            })
    }
}
