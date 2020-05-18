CC=g++
IDIR=
CFLAGS=-Wall -Wextra -std=c++17 -lpthread -g $(patsubst %,-I %,$(IDIR))
FFLAGS=
ODIR=.
PROGRAMS=telnet

all: telnet
telnet: client.cpp socket.hpp
	$(CC) -o telnet $< $(CFLAGS) $(FFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS)
