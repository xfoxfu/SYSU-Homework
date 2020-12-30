pub const S_BOX: &[u8;16*16] = 
    b"\x63\x7C\x77\x7B\xF2\x6B\x6F\xC5\x30\x01\x67\x2B\xFE\xD7\xAB\x76\xCA\x82\xC9\x7D\xFA\x59\x47\xF0\xAD\xD4\xA2\xAF\x9C\xA4\x72\xC0\xB7\xFD\x93\x26\x36\x3F\xF7\xCC\x34\xA5\xE5\xF1\x71\xD8\x31\x15\x04\xC7\x23\xC3\x18\x96\x05\x9A\x07\x12\x80\xE2\xEB\x27\xB2\x75\x09\x83\x2C\x1A\x1B\x6E\x5A\xA0\x52\x3B\xD6\xB3\x29\xE3\x2F\x84\x53\xD1\x00\xED\x20\xFC\xB1\x5B\x6A\xCB\xBE\x39\x4A\x4C\x58\xCF\xD0\xEF\xAA\xFB\x43\x4D\x33\x85\x45\xF9\x02\x7F\x50\x3C\x9F\xA8\x51\xA3\x40\x8F\x92\x9D\x38\xF5\xBC\xB6\xDA\x21\x10\xFF\xF3\xD2\xCD\x0C\x13\xEC\x5F\x97\x44\x17\xC4\xA7\x7E\x3D\x64\x5D\x19\x73\x60\x81\x4F\xDC\x22\x2A\x90\x88\x46\xEE\xB8\x14\xDE\x5E\x0B\xDB\xE0\x32\x3A\x0A\x49\x06\x24\x5C\xC2\xD3\xAC\x62\x91\x95\xE4\x79\xE7\xC8\x37\x6D\x8D\xD5\x4E\xA9\x6C\x56\xF4\xEA\x65\x7A\xAE\x08\xBA\x78\x25\x2E\x1C\xA6\xB4\xC6\xE8\xDD\x74\x1F\x4B\xBD\x8B\x8A\x70\x3E\xB5\x66\x48\x03\xF6\x0E\x61\x35\x57\xB9\x86\xC1\x1D\x9E\xE1\xF8\x98\x11\x69\xD9\x8E\x94\x9B\x1E\x87\xE9\xCE\x55\x28\xDF\x8C\xA1\x89\x0D\xBF\xE6\x42\x68\x41\x99\x2D\x0F\xB0\x54\xBB\x16";

pub fn sbox_get(b:u8) -> u8{
    S_BOX[b as usize]
}

pub const INV_S_BOX: &[u8;16*16] = 
    b"\x52\x09\x6A\xD5\x30\x36\xA5\x38\xBF\x40\xA3\x9E\x81\xF3\xD7\xFB\x7C\xE3\x39\x82\x9B\x2F\xFF\x87\x34\x8E\x43\x44\xC4\xDE\xE9\xCB\x54\x7B\x94\x32\xA6\xC2\x23\x3D\xEE\x4C\x95\x0B\x42\xFA\xC3\x4E\x08\x2E\xA1\x66\x28\xD9\x24\xB2\x76\x5B\xA2\x49\x6D\x8B\xD1\x25\x72\xF8\xF6\x64\x86\x68\x98\x16\xD4\xA4\x5C\xCC\x5D\x65\xB6\x92\x6C\x70\x48\x50\xFD\xED\xB9\xDA\x5E\x15\x46\x57\xA7\x8D\x9D\x84\x90\xD8\xAB\x00\x8C\xBC\xD3\x0A\xF7\xE4\x58\x05\xB8\xB3\x45\x06\xD0\x2C\x1E\x8F\xCA\x3F\x0F\x02\xC1\xAF\xBD\x03\x01\x13\x8A\x6B\x3A\x91\x11\x41\x4F\x67\xDC\xEA\x97\xF2\xCF\xCE\xF0\xB4\xE6\x73\x96\xAC\x74\x22\xE7\xAD\x35\x85\xE2\xF9\x37\xE8\x1C\x75\xDF\x6E\x47\xF1\x1A\x71\x1D\x29\xC5\x89\x6F\xB7\x62\x0E\xAA\x18\xBE\x1B\xFC\x56\x3E\x4B\xC6\xD2\x79\x20\x9A\xDB\xC0\xFE\x78\xCD\x5A\xF4\x1F\xDD\xA8\x33\x88\x07\xC7\x31\xB1\x12\x10\x59\x27\x80\xEC\x5F\x60\x51\x7F\xA9\x19\xB5\x4A\x0D\x2D\xE5\x7A\x9F\x93\xC9\x9C\xEF\xA0\xE0\x3B\x4D\xAE\x2A\xF5\xB0\xC8\xEB\xBB\x3C\x83\x53\x99\x61\x17\x2B\x04\x7E\xBA\x77\xD6\x26\xE1\x69\x14\x63\x55\x21\x0C\x7D";

pub fn inv_sbox_get(b:u8) -> u8{
    INV_S_BOX[b as usize]
}

pub const RCON: &[u8; 11] = b"\x00\x01\x02\x04\x08\x10\x20\x40\x80\x1B\x36";

pub fn rcon_get(pos:usize) -> u32 {
    (RCON[pos] as u32) << 24
}

#[cfg(test)]
#[test]
fn test_sbox() {
    assert_eq!(S_BOX[0x58], 0x6A);
    assert_eq!(sbox_get(0x58), 0x6A);
    assert_eq!(S_BOX[0xC1], 0x78);
    assert_eq!(sbox_get(0xC1), 0x78);
    assert_eq!(INV_S_BOX[0x6A], 0x58);
    assert_eq!(inv_sbox_get(0x6A), 0x58);
    assert_eq!(INV_S_BOX[0x78], 0xC1);
    assert_eq!(inv_sbox_get(0x78), 0xC1);
}

#[cfg(test)]
#[test]
fn test_rcon() {
    assert_eq!(RCON[0x07], 0x40);
    assert_eq!(rcon_get(0x07), 0x40000000);
}
