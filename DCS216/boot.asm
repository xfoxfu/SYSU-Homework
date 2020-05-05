; this is a NASM code

bits 16
org  0x7c00 ; expected to be loaded at 0x7c00 (bootloader)

LOAD_SECTION equ 0x800
SECTION_OFFSET equ 0x100

_start:
    ; set correct segment base address
    mov ax, cs
    mov ds, ax
    mov es, ax
    xor ax, ax
    mov ss, ax
    mov sp, bp

another:
    ; mov byte[bp], '0'
    mov word[bpp], bp
    mov bp, message
    mov ax, 1301h
    mov bx, 000Fh
    mov cx, len
    mov dx, 0000h
    int 10h
    mov bp, word[bpp]

    ; read character input
    mov ah, 0x00
    int 0x16

    ; compute disk sector id
    mov cl, al
    sub cl, 'a' - 3

read:
    ; DOS 1+ - SELECT DEFAULT DRIVE
    ; 
    ; AH = 0Eh
    ; DL = new default drive (00h = A:, 01h = B:, etc)
    mov ah, 0Eh
    int 21h
    ; DISK - READ SECTOR(S) INTO MEMORY
    ; 
    ; AH = 02h
    ; AL = number of sectors to read (must be nonzero)
    ; CH = low eight bits of cylinder number
    ; CL = sector number 1-63 (bits 0-5)
    ; high two bits of cylinder (bits 6-7, hard disk only)
    ; DH = head number
    ; DL = drive number (bit 7 set for hard disk)
    ; ES:BX -> data buffer
    ; 
    mov ax, LOAD_SECTION
    mov es, ax
    mov bx, SECTION_OFFSET
    mov ax, 0201h
    mov dh, 00h
    ; DL read by others
    mov ch, 00h
    int 13h

    ; execute user program
    ; mov ax, [another]
    call (LOAD_SECTION << 4 + SECTION_OFFSET)

    xor ax, ax
    mov es, ax
    mov bp, message
    mov ax, 1301h
    mov bx, 000Fh
    mov cx, len
    mov dx, 0000h
    int 10h
    jmp another

    hlt

    message db "Enter a,b,c,d to run different program"
    len     equ ($ -message)
    bpp dw 0
times 510 - ($-$$) db 0 ; pad remaining 510 bytes with zeroes
dw 0xaa55 ; magic bootloader magic - marks this 512 byte sector bootable!
