@0xbf9bfb4bcd258283;

interface Store {
  put @0 (key: Text, value: Data);
  get @1 (key: Text) -> (present: Bool, value: Data);
  del @2 (key: Text) -> (present: Bool);
}
