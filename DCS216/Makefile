CC=x86_64-elf-gcc
CPP=x86_64-elf-g++
AS=nasm
CFLAGS=-std=gnu17 -O0 -nostdlib -m16 -masm=intel -ffreestanding -Wl,--nmagic,--script=linker.ld -Xlinker -melf_i386
SDIR=src
ODIR=dist
BIN=$(ODIR)/os.img

_OBJ = bootloader kernel.bin crt0.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/bootloader: boot.asm | $(ODIR)/
	$(AS) -f bin -o $@ $<

$(ODIR)/crt0.o: crt0.asm linker.ld | $(ODIR)/
	$(AS) -f elf32 -o $@ $<

$(ODIR)/%.bin: %.c $(ODIR)/crt0.o | $(ODIR)/
	$(CC) $(CFLAGS) -o $@ $^

$(BIN): $(OBJ)
	rm -rf $(BIN)
	dd if=$(ODIR)/bootloader of=$@ bs=512 count=1 seek=0 conv=notrunc
	dd if=$(ODIR)/kernel.bin of=$@ bs=512 count=6 seek=1 conv=notrunc

.PHONY: clean run

$(ODIR)/:
	mkdir -p $(ODIR)

clean:
	rm -f $(OBJ) $(BIN)

run: $(BIN)
	qemu-system-x86_64 -fda $(BIN)
