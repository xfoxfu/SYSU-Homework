bits 16
org  0x7c00

LOAD_SECTION equ 0x800
DRIVER_OFFSET equ 0x100

_start:
	mov ax, cs ; 通过AX中转, 将CS的值传送给DS和ES
	mov ds, ax
	mov es, ax

    ; ; Set SS and SP as they may get used by BIOS calls.
    ; xor ax, ax
    ; mov ss, ax
    ; mov sp, 0x0000


    ; cli


    ; Get input to %al
    mov ah, 0x00
    int 0x16

    ; mov cl, al
    ; sub cl, 'a' - 2
    ; inc al
	cmp al ,'a'
	je pa
	cmp al ,'b'
	je pb
	cmp al ,'c'
	je pc
	cmp al ,'d'
	je pd
pa:
	mov cl,02h
	jmp read
pb:
	mov cl,03h
	jmp read
pc:
	mov cl,04h
	jmp read
pd:
	mov cl,05h
	jmp read	

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
	mov bx, DRIVER_OFFSET
	mov ax,0201h
	; mov dx,0080h
    mov dh, 00h
    ; DL read by others
	mov ch,00h
	
	int 13h
    ; mov ax, LOAD_SECTION
    ; mov es, ax
	; jmp [es:100h]
    jmp (LOAD_SECTION << 4 + DRIVER_OFFSET)

    hlt

times 510 - ($-$$) db 0 ; pad remaining 510 bytes with zeroes
dw 0xaa55 ; magic bootloader magic - marks this 512 byte sector bootable!
