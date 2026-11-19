//! Layout: `u32 max_new`, `u32 n`, `u32 prompt[n]`, little-endian.

use alloc::vec::Vec;

use crate::read_u32;

#[derive(Clone, Debug, PartialEq)]
pub struct GenParams {
    pub prompt: Vec<u32>,
    pub max_new: u32,
}

impl GenParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + 4 * self.prompt.len());
        b.extend_from_slice(&self.max_new.to_le_bytes());
        b.extend_from_slice(&(self.prompt.len() as u32).to_le_bytes());
        for t in &self.prompt {
            b.extend_from_slice(&t.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let n = read_u32(b, 4) as usize;
        assert_eq!(b.len(), 8 + 4 * n, "malformed generation parameters");
        GenParams {
            max_new: read_u32(b, 0),
            prompt: (0..n).map(|i| read_u32(b, 8 + 4 * i)).collect(),
        }
    }
}
