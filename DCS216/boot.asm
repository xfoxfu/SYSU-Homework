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
    push 0

    mov ah, 0x00
    mov al, 0x03  ; text mode 80x25 16 colours
    int 0x10

another:
    ; set default drive id into DL
    mov ah, 0Eh
    int 21h
    mov [drive], dl
    
    ; read executable list into memory
    mov ax, LOAD_SECTION
    mov es, ax
    mov bx, SECTION_OFFSET
    mov ax, 0201h
    mov dh, 00h
    ; mov dl, 02h
    mov ch, 00h
    mov cl, 02h
    int 13h

    mov cl, [es:bx]
    sub cl, '0' - 1 ; calculate file count

    pop ax
    cmp ax, 0 ; ax <= 0
    jle read_cmd ; if command not present
    cmp al, 'a'
    jl another
    add cl, 'a'
    cmp al, cl ; ax <= cl
    jle read
    jmp another

read_cmd:
    ; read command
    push 0

    imul cx, 32 ; calculate char count
    mov byte [list_len], al
    ; print executable list
    mov [bpp], bp ; protect bp
    mov bp, bx
    inc bp
    mov ax, 1301h
    mov bx, 000Fh
    mov dx, 0000h
    int 10h
    mov bp, [bpp] ; restore bp

    ; VIDEO - WRITE CHARACTER ONLY AT CURSOR POSITION
    ; 
    ; AH = 0Ah
    ; AL = character to display
    ; BH = page number (00h to number of pages - 1) (see #00010)
    ; background color in 256-color graphics modes (ET4000)
    ; BL = attribute (PCjr, Tandy 1000 only) or color (graphics mode)
    ; if bit 7 set in <256-color graphics mode, character is XOR'ed
    ; onto screen
    ; CX = number of times to write character
    mov ah, 0Ah
    mov al, 20h
    mov bx, 0000h
    mov cx, 80
    int 10h

    mov dx, 0000h
cmd_rd:
    ; read character input
    mov ah, 0x00
    int 0x16
    cmp al, 0x0D ; CR
    je cmd_fin
    cmp al, 0x0A ; LF
    je cmd_fin
    push ax

    ; VIDEO - TELETYPE OUTPUT
    ; 
    ; AH = 0Eh
    ; AL = character to write
    ; BH = page number
    ; BL = foreground color (graphics modes only)
    mov ah, 0Eh
    mov bx, 0000h
    int 10h
    inc dx

    cmp al, 0x08 ; backslash
    jne cmd_rd

    pop ax
    pop ax
    jmp cmd_rd
cmd_fin:
    jmp another

read:
    ; compute disk sector id
    mov cl, al
    sub cl, 'a' - 3

    mov dl, byte [drive]
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
    ; mov dh, 00h
    mov dx, 0000h
    ; DL read by others
    mov ch, 00h
    int 13h

    ; execute user program
    ; mov ax, [another]
    call (LOAD_SECTION << 4 + SECTION_OFFSET)

    xor cx, cx
    xor dx, dx
    jmp another

    hlt

    bpp dw 0 ; used to protect bp
    drive db 0
    list_len db 0
    com_len  db 0
    commands db 0

times 510 - ($-$$) db 0 ; pad remaining 510 bytes with zeroes
dw 0xaa55 ; magic bootloader magic - marks this 512 byte sector bootable!
