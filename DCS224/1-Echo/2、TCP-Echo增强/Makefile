CC=g++
IDIR=
CFLAGS=-Wall -Wextra -std=c++17 -g $(patsubst %,-I %,$(IDIR))
FFLAGS=
ODIR=.
PROGRAMS=server client

all: server client
server: server.cpp socket.hpp
	$(CC) -o server $< $(CFLAGS) $(FFLAGS)
client: client.cpp socket.hpp
	$(CC) -o client $< $(CFLAGS) $(FFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS)
