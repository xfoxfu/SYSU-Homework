use crate::case::*;

#[macro_use]
mod macro_utils {
    #[macro_export]
    macro_rules! assert_feq {
        ($lhs:expr,$rhs:expr) => {
            let lv = $lhs;
            let rv = $rhs;
            assert!((lv - rv).abs() < f64::EPSILON, "expected {} ~= {}", lv, rv);
        };
    }
}

pub fn make_case(b: Buying, m: Maint, l: Label) -> Case {
    Case::new(
        b,
        m,
        Doors::Two,
        Persons::Two,
        LugBoot::Big,
        Safety::High,
        l,
    )
}
