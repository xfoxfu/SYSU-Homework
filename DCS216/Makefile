CC=nasm
CFLAGS=-f bin
SDIR=src
ODIR=dist
BIN=$(ODIR)/os.img

_OBJ = boot.bin display_bottom_left.bin display_bottom_right.bin display_top_left.bin display_top_right.bin
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.bin: %.asm | $(ODIR)/
	$(CC) $(CFLAGS) -o $@ $<

$(BIN): $(OBJ)
	dd if=$(ODIR)/boot.bin of=$@ bs=512 count=1 seek=0 conv=notrunc
	dd if=$(ODIR)/display_bottom_left.bin of=$@ bs=512 count=1 seek=1 conv=notrunc
	dd if=$(ODIR)/display_bottom_right.bin of=$@ bs=512 count=1 seek=2 conv=notrunc
	dd if=$(ODIR)/display_top_left.bin of=$@ bs=512 count=1 seek=3 conv=notrunc
	dd if=$(ODIR)/display_top_right.bin of=$@ bs=512 count=1 seek=4 conv=notrunc

.PHONY: clean run

$(ODIR)/:
	mkdir -p $(ODIR)

clean:
	rm -f $(OBJ) $(BIN)

run: $(BIN)
	qemu-system-x86_64 $(BIN)
