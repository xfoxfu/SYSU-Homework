CC=x86_64-elf-gcc
CPP=x86_64-elf-g++
AS=nasm
CFLAGS=-std=gnu17 -Os -nostdlib -m16 -masm=intel -ffreestanding -Wl,--nmagic,--script=linker.ld -Xlinker -melf_i386
SDIR=src
ODIR=sysroot
BIN=$(ODIR)/os.img

_OBJ = bootloader kernel.com crt0.o \
	display_bottom_left.com display_bottom_right.com \
	display_top_left.com display_top_right.com
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_COMMONC = biosfn.c

.DEFAULT_GOAL := $(BIN)

$(ODIR)/bootloader: boot.asm | $(ODIR)/
	$(AS) -f bin -o $@ $<

$(ODIR)/crt0.o: crt0.asm linker.ld | $(ODIR)/
	$(AS) -f elf32 -o $@ $<

$(ODIR)/%.com: %.c $(ODIR)/crt0.o $(_COMMONC) | $(ODIR)/
	$(CC) $(CFLAGS) -o $@ $^

$(BIN): $(OBJ)
	dd if=/dev/zero                        of=$@ bs=512 count=30
	dd if=$(ODIR)/bootloader               of=$@ bs=512 count=1 seek=0  conv=notrunc
	dd if=$(ODIR)/kernel.com               of=$@ bs=512 count=6 seek=1  conv=notrunc
	dd if=$(ODIR)/display_top_left.com     of=$@ bs=512 count=3 seek=7  conv=notrunc
	dd if=$(ODIR)/display_top_right.com    of=$@ bs=512 count=3 seek=10 conv=notrunc
	dd if=$(ODIR)/display_bottom_left.com  of=$@ bs=512 count=3 seek=13 conv=notrunc
	dd if=$(ODIR)/display_bottom_right.com of=$@ bs=512 count=3 seek=16 conv=notrunc

.PHONY: clean run

$(ODIR)/:
	mkdir -p $(ODIR)

clean:
	rm -f $(OBJ) $(BIN)

run: $(BIN)
	qemu-system-x86_64 -fda $(BIN)

debug: $(BIN)
	qemu-system-x86_64 -fda $(BIN) -s -S
