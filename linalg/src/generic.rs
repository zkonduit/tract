pub mod by_scalar;
pub mod erf;
pub mod leaky_relu;
pub mod lut;
pub mod mmm;
pub mod reduce;
pub mod rounding;
pub mod sigmoid;
pub mod tanh;
pub mod unicast;

use tract_data::prelude::DatumType;

use crate::frame::reduce::ReduceKer;
use crate::{BinOp, ByScalarKer, InsertOpExt, Linalg1Registry, LinalgRegistry, UnicastKer};

pub use self::by_scalar::{HMulByScalar8, SMulByScalar4};
pub use self::erf::SErf4;
pub use self::leaky_relu::{HLeakyRelu8, SLeakyRelu4};
pub use self::lut::GenericLut8;
pub use self::reduce::softmax_l2::SSoftMaxL2;
pub use self::rounding::{ScaleShiftAndRound, Scaler};
pub use self::sigmoid::{HSigmoid8, SSigmoid4};
pub use self::tanh::{HTanh8, STanh4};

pub(crate) fn register_all_unicast(registry: &mut LinalgRegistry) {
    registry.insert_op((BinOp::Mul, DatumType::F32), Box::new(|| unicast::SUnicastMul4::bin()));
    registry.insert_op((BinOp::Mul, DatumType::F16), Box::new(|| unicast::HUnicastMul8::bin()));
    registry.insert_op((BinOp::Add, DatumType::F32), Box::new(|| unicast::SUnicastAdd4::bin()));
    registry.insert_op((BinOp::Add, DatumType::F16), Box::new(|| unicast::HUnicastAdd8::bin()));
    registry.insert_op((BinOp::Sub, DatumType::F32), Box::new(|| unicast::SUnicastSub4::bin()));
    registry.insert_op((BinOp::Sub, DatumType::F16), Box::new(|| unicast::HUnicastSub8::bin()));
    registry.insert_op((BinOp::SubF, DatumType::F32), Box::new(|| unicast::SUnicastSubF4::bin()));
    registry.insert_op((BinOp::SubF, DatumType::F16), Box::new(|| unicast::HUnicastSubF8::bin()));
    registry.insert_op((BinOp::Min, DatumType::F32), Box::new(|| unicast::SUnicastMin4::bin()));
    registry.insert_op((BinOp::Min, DatumType::F16), Box::new(|| unicast::HUnicastMin8::bin()));
    registry.insert_op((BinOp::Max, DatumType::F32), Box::new(|| unicast::SUnicastMax4::bin()));
    registry.insert_op((BinOp::Max, DatumType::F16), Box::new(|| unicast::HUnicastMax8::bin()));
}

pub(crate) fn register_all_by_scalar(registry: &mut LinalgRegistry) {
    registry.insert_op((BinOp::Mul, DatumType::F32), Box::new(|| by_scalar::SMulByScalar4::bin()));
    registry.insert_op((BinOp::Mul, DatumType::F16), Box::new(|| by_scalar::HMulByScalar8::bin()));
    registry.insert_op((BinOp::Add, DatumType::F32), Box::new(|| by_scalar::SAddByScalar4::bin()));
    registry.insert_op((BinOp::Add, DatumType::F16), Box::new(|| by_scalar::HAddByScalar8::bin()));
    registry.insert_op((BinOp::Sub, DatumType::F32), Box::new(|| by_scalar::SSubByScalar4::bin()));
    registry.insert_op((BinOp::Sub, DatumType::F16), Box::new(|| by_scalar::HSubByScalar8::bin()));
    registry.insert_op((BinOp::SubF, DatumType::F32), Box::new(|| by_scalar::SSubFByScalar4::bin()));
    registry.insert_op((BinOp::SubF, DatumType::F16), Box::new(|| by_scalar::HSubFByScalar8::bin()));
    registry.insert_op((BinOp::Min, DatumType::F32), Box::new(|| by_scalar::SMinByScalar4::bin()));
    registry.insert_op((BinOp::Min, DatumType::F16), Box::new(|| by_scalar::HMinByScalar8::bin()));
    registry.insert_op((BinOp::Max, DatumType::F32), Box::new(|| by_scalar::SMaxByScalar4::bin()));
    registry.insert_op((BinOp::Max, DatumType::F16), Box::new(|| by_scalar::HMaxByScalar8::bin()));
}

pub(crate) fn register_all_reducer(registry: &mut Linalg1Registry) {
    registry.insert_op((BinOp::Max, DatumType::F32), Box::new(|| reduce::max::SMax4::red_1()));
    registry.insert_op((BinOp::Max, DatumType::F16), Box::new(|| reduce::max::HMax8::red_1()));
    registry.insert_op((BinOp::Add, DatumType::F32), Box::new(|| reduce::sum::SSum4::red_1()));
    registry.insert_op((BinOp::Add, DatumType::F16), Box::new(|| reduce::sum::HSum8::red_1()));
}
