CC=g++
IDIR=
CFLAGS=-Wall -Wextra -std=c++17 -O2 -g $(patsubst %,-I %,$(IDIR))
FFLAGS=
ODIR=.
PROGRAMS=server client
OBJS=socket.o common.o

all: $(PROGRAMS)
%.o: %.cpp %.hpp
	$(CC) -c -o $@ $< $(CFLAGS) $(FFLAGS)
server: server.cpp $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(FFLAGS)
client: client.cpp $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(FFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) $(OBJS)
