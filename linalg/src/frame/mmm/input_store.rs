use downcast_rs::{impl_downcast, Downcast};
use dyn_clone::DynClone;
use dyn_hash::DynHash;
use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;
use tract_data::internal::*;

pub trait MMMInputFormat: Downcast + Debug + DynHash + DynClone + Send + Sync + Display {
    fn prepare_tensor(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>>;
    fn r(&self) -> usize;
    fn k_alignment(&self) -> usize;
    fn same_as(&self, other: &dyn MMMInputFormat) -> bool;
    fn mem_size(&self, k: TDim, mn: TDim) -> TDim;
}
dyn_clone::clone_trait_object!(MMMInputFormat);
impl_downcast!(MMMInputFormat);
dyn_hash::hash_trait_object!(MMMInputFormat);

pub trait MMMInputValue: DynClone + Debug + DynHash + Send + Sync + Display + Downcast {
    fn format(&self) -> &dyn MMMInputFormat;
    fn scratch_panel_buffer_layout(&self) -> Option<Layout>;
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8>;
    fn panels_count(&self) -> usize {
        self.mn().divceil(self.format().r())
    }
    fn mn(&self) -> usize;
    fn k(&self) -> usize;
    fn opaque_fact(&self) -> &dyn OpaqueFact;
    fn same_as(&self, other: &dyn MMMInputValue) -> bool;
}
dyn_clone::clone_trait_object!(MMMInputValue);
impl_downcast!(MMMInputValue);
dyn_hash::hash_trait_object!(MMMInputValue);

impl From<Box<dyn MMMInputValue>> for Opaque {
    fn from(value: Box<dyn MMMInputValue>) -> Self {
        Opaque(Arc::new(value))
    }
}

impl OpaquePayload for Box<dyn MMMInputValue> {
    fn same_as(&self, other: &dyn OpaquePayload) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| (*self).same_as(other))
    }
}

#[derive(Clone, Hash, Debug)]
pub struct PackedOpaqueFact {
    pub format: Box<dyn MMMInputFormat>,
    pub mn: usize,
    pub k: usize,
}

impl Display for PackedOpaqueFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Eager {} tensor (mn={} k={})",
            self.format, self.mn, self.k
        )
    }
}

impl OpaqueFact for PackedOpaqueFact {
    fn mem_size(&self) -> TDim {
        self.format.mem_size(self.k.to_dim(), self.mn.to_dim())
    }

    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }
}

impl PartialEq for PackedOpaqueFact {
    fn eq(&self, other: &Self) -> bool {
        self.format.same_as(&*other.format) && self.mn == other.mn && self.k == other.k
    }
}

#[derive(Clone, Hash)]
pub struct EagerPackedInput {
    pub fact: PackedOpaqueFact,
    pub packed: Arc<Blob>,
    pub panel_bytes: usize,
}

impl MMMInputValue for EagerPackedInput {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel_bytes(&self, i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        unsafe { Ok(self.packed.as_ptr().add(i * self.panel_bytes)) }
    }
    fn k(&self) -> usize {
        self.fact.k
    }
    fn mn(&self) -> usize {
        self.fact.mn
    }
    fn format(&self) -> &dyn MMMInputFormat {
        &*self.fact.format
    }
    fn opaque_fact(&self) -> &dyn OpaqueFact {
        &self.fact
    }
    fn same_as(&self, other: &dyn MMMInputValue) -> bool {
        other.downcast_ref::<Self>().is_some_and(|other| {
            self.fact.same_as(&other.fact)
                && self.packed == other.packed
                && self.panel_bytes == other.panel_bytes
        })
    }
}

impl Display for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (&self.fact as &dyn Display).fmt(f)
    }
}

impl Debug for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}
