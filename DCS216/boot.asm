; this is a NASM code

bits 16
org  0x7c00 ; expected to be loaded at 0x7c00 (bootloader)

; load kernel at 800:100 (0x8000)
KERNEL_SEGMNT equ 800h
KERNEL_OFFSET equ 100h
KERNEL_SECLEN equ 200h
KERNEL_SECEND equ KERNEL_OFFSET + 200h * 6

; section .boot
boot:
	xor ax, ax
    mov ds, ax
    mov ss, ax
    mov es, ax ; setup segments
    mov bp, 0x7C00
    mov sp, bp ; setup stack

    ; set display mode
    mov ah, 0x00
    mov al, 0x03  ; text mode 80x25 16 colours
    int 0x10 ; out al

	; set default drive id into DL
    mov ah, 0Eh
    int 21h ; out dl
    mov [disk], dl

    mov ax, KERNEL_SEGMNT
    mov es, ax
    mov bx, KERNEL_OFFSET
    ; mov dh, 00h
    mov dx, 0000h
    mov dl, byte [disk]
    mov cx, 0002h
; load:
    mov ax, 0206h
    int 13h ; OUT CF AH AL
    jc error ; error handling

kstart:
    lea si, [msg_ok]
    call print_str
    call KERNEL_SEGMNT:KERNEL_OFFSET

    ; restore segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    lea si, [msg_exit]
    call print_str

    mov dx, 0x4240
    mov cx, 0x000F
    mov ax, 0x8600
    int 0x15

    mov ax, 0x1000
    mov ax, ss
    mov sp, 0xf000
    mov ax, 0x5307
    mov bx, 0x0001
    mov cx, 0x0003
    int 0x15
    jmp $

error:
    lea si, [msg_err]
    call print_str
    jmp $

print_str: ; IN si start address
    lodsb
    cmp al, 0
    je .done
    mov ah, 0xE
    mov bx, 7
    int 10h
    jmp print_str
    .done:
        ret

msg_err:  db "Error loading kernel", 0x0D, 0x0A, 0x00 
msg_ok:   db "Kernel loaded......", 0x0D, 0x0A, 0x00
msg_exit: db "Shutting down......", 0x0D, 0x0A, 0x00
disk:     db 0x0

times 510 - ($-$$) db 0
dw 0xaa55
