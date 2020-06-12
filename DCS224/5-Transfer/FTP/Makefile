CC=g++
IDIR=
CFLAGS=-Wall -Wextra -std=c++17 -O2 -g $(patsubst %,-I %,$(IDIR))
FFLAGS=
ODIR=.
PROGRAMS=ftpget
OBJS=socket_stream.o socket.o

all: $(PROGRAMS)
%.o: %.cpp %.hpp
	$(CC) -c -o $@ $< $(CFLAGS) $(FFLAGS)
ftpget: ftpget.cpp $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(FFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) $(OBJS)
