CC=g++
IDIR=
CFLAGS=-Wall -Wextra -std=c++17 -O2 -g $(patsubst %,-I %,$(IDIR))
FFLAGS=
ODIR=.
PROGRAMS=ftpget

all: $(PROGRAMS)
ftpget: ftpget.cpp socket.hpp
	$(CC) -o $@ $< $(CFLAGS) $(FFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS)
