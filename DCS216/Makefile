CC=nasm
CFLAGS=-f bin
SDIR=src
ODIR=dist
BIN=$(ODIR)/os.bin

_OBJ = BOOT.BIN LT.BIN RT.BIN LB.BIN RB.BIN
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.BIN: $(SDIR)/%.asm | $(ODIR)/
	$(CC) $(CFLAGS) -o $@ $<

$(BIN): $(OBJ)

.PHONY: clean run

$(ODIR)/:
	mkdir -p $(ODIR)

clean:
	rm -f $(OBJ) $(BIN)

run: $(BIN)
	./$(BIN)
