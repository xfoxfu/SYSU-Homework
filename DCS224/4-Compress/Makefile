CC=g++
IDIR=
CFLAGS=-Wall -Wextra -std=c++17 -O2 -g $(patsubst %,-I %,$(IDIR))
FFLAGS=
ODIR=.
PROGRAMS=StructSave StructRead FilePack FileUnpack

all: $(PROGRAMS)
StructSave: StructSave.cpp Person.hpp
	$(CC) -o $@ $< $(CFLAGS) $(FFLAGS)
StructRead: StructRead.cpp Person.hpp
	$(CC) -o $@ $< $(CFLAGS) $(FFLAGS)
FilePack: FilePack.cpp FileStruct.hpp
	$(CC) -o $@ $< $(CFLAGS) $(FFLAGS)
FileUnpack: FileUnpack.cpp FileStruct.hpp
	$(CC) -o $@ $< $(CFLAGS) $(FFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS)
