bits 16

section .entry
global start
start:
    mov  ax,  cs
	mov  ds,  ax
	mov  es,  ax

    push hello
    extern dosmain
    call dword dosmain ; call dword is essential
    add sp, 4
    mov cx, ax
    mov dx, prompt
    mov ah, 0x09
    int 0x21
    mov dx, hello
    int 0x21
    mov dx, prompt2
    int 0x21

    mov dl, cl
    add dl, '0'
    mov ah, 2  ; 2 is the function number of output char in the DOS Services.
    int 21h    ; calls DOS Services

    mov   al, 0x00
    mov   ah, 0x4C
    int   0x21 ; exit dos

section .data
prompt:  db "Total count of `*` in `$"
prompt2: db "`: $"
hello:  db "Lo**m Ip*um*dolo**sit a*e*$"

section .text
