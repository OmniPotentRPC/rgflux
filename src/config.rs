//! Chainable `with_*` setters for the public configuration structs.
//!
//! Every configuration struct is `#[non_exhaustive]`: a caller outside
//! this crate starts from [`Default::default`] or a named constructor
//! and chains setters, so a field added later is not a breaking change.
//! An `opt` field takes `impl Into<Option<T>>`, so `with_field(value)`
//! sets `Some(value)` and `with_field(None)` clears it.

macro_rules! with_setters {
    ($ty:ident { $($setter:ident => $field:ident : $kind:ident $fty:ty),* $(,)? }) => {
        impl $ty {
            $( with_setters!(@one $setter $field $kind $fty); )*
        }
    };
    (@one $setter:ident $field:ident val $fty:ty) => {
        #[doc = concat!("Set [`Self::", stringify!($field), "`].")]
        #[must_use]
        pub fn $setter(mut self, $field: $fty) -> Self {
            self.$field = $field;
            self
        }
    };
    (@one $setter:ident $field:ident opt $fty:ty) => {
        #[doc = concat!("Set [`Self::", stringify!($field), "`].")]
        ///
        /// A bare value sets `Some`; `None` clears it.
        #[must_use]
        pub fn $setter(mut self, $field: impl Into<Option<$fty>>) -> Self {
            self.$field = $field.into();
            self
        }
    };
}
