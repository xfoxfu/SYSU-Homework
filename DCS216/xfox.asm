bits 16 ; tell NASM this is 16 bit code
org 0x7c00 ; tell NASM to start outputting stuff at offset 0x7c00

DISP_WIDTH  equ 80
DISP_HEIGHT equ 25
DISP_BASE   equ 0xB800

boot:
    mov  bp, 7C00h
    xor  ax, ax
    mov  ds, ax
    mov  es, ax
    mov  ss, ax      ; \  Keep these close together
    mov  sp, bp      ; / 

    mov ax, word DISP_BASE
    mov es, ax ; set ES at base address of VGA buffer
;   int col = 0;
    mov cl, 0
;   int row = 0;
    mov ch, 0
;   int col_incr = 1;
    mov dl, 1
;   int row_incr = 1;
    mov dh, 1
;   int color = 1;
    mov ax, 0x0F
    mov byte [color], al
;   for (;;) {
loop:
print:
;     table[row][col] = '*';
    xor bx, bx      ; bx = 0
    xor ax, ax      ; ax = 0
    mov al, ch      ; ax = [0:ch]
    add bx, ax      ; bx += ax
    imul bx, 80     ; bx *= 80
    mov word [sbx], bx
    add word [sbx], 80
    shl word [sbx], 1
    mov al, cl      ; ax = [0:cl]
    mov [scx], cx
    mov [sdx], dx
    add bx, ax      ; bx += ax
    shl bx, 1       ; bx *= 2
    ; mov ah, cl    ; 
    mov ah, byte [color]
    and ah, 0x0F
    or ah, 0x08
    mov al, 'A'     ; ax = [0F: *]
    mov [last], bx
    mov [es:bx], ax ;
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect0], cx
    mov al, '1'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect1], cx
    mov al, '7'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect2], cx
    mov al, '3'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect3], cx
    mov al, '4'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect4], cx
    mov al, '1'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect5], cx
    mov al, '0'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect6], cx
    mov al, '3'
    mov [es:bx], ax
    add bx, 2
    cmp bx, [sbx]
    jge print_exit
    mov cx, [es:bx]
    mov [protect7], cx
    mov al, '9'
    mov [es:bx], ax
print_exit:
    mov dx, [sdx]
    mov cx, [scx]
;     col += col_incr;
    add cl, dl
;     row += row_incr;
    add ch, dh
;     if (col == 0)
    mov ax, 0
    cmp cl, al
    jne fwd1
;       col_incr = 1;
    mov dl, 1
    inc byte [color]
fwd1:
;     if (col == 79)
    mov ax, DISP_WIDTH - 1
    cmp cl, al
    jne fwd2
;       col_incr = -1;
    mov dl, -1
    inc byte [color]
fwd2:
;     if (row == 0)
    mov ax, 0
    cmp ch, al
    jne fwd3
;       row_incr = 1;
    mov dh, 1
    inc byte [color]
fwd3:
;     if (row == 23)
    mov ax, DISP_HEIGHT - 1
    cmp ch, al
    jne fwd4
;       row_incr = -1;
    mov dh, -1
    inc byte [color]
fwd4:
    mov [scx], cx
    mov [sdx], dx
;     usleep(1000 * 1000 * 0.1);
    mov cx, 0x0000
    mov dx, 0xC350
    ; mov cx, 0x0001
    ; mov dx, 0x86A0
    mov AH, 0x86
    int 0x15
    mov dx, [sdx]
    mov cx, [scx]

    mov [scx], cx
    mov bx, [last]
    mov cx, [protect0]
    mov [es:bx+2], cx
    mov cx, [protect1]
    mov [es:bx+4], cx
    mov cx, [protect2]
    mov [es:bx+6], cx
    mov cx, [protect3]
    mov [es:bx+8], cx
    mov cx, [protect4]
    mov [es:bx+10], cx
    mov cx, [protect5]
    mov [es:bx+12], cx
    mov cx, [protect6]
    mov [es:bx+14], cx
    mov cx, [protect7]
    mov [es:bx+16], cx
    mov cx, [scx]

    jmp loop
;   }

    jmp $
data:
    color db 0
    sax dw 0
    sbx dw 0
    scx dw 0
    sdx dw 0
    last dw 0
    protect0 dw 0
    protect1 dw 0
    protect2 dw 0
    protect3 dw 0
    protect4 dw 0
    protect5 dw 0
    protect6 dw 0
    protect7 dw 0

times 510 - ($-$$) db 0 ; pad remaining 510 bytes with zeroes
dw 0xaa55 ; magic bootloader magic - marks this 512 byte sector bootable!
