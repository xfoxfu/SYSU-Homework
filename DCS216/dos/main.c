char message[12] = "AaBbCcDdEe$";
char progress[3] = ".$";

void print(char *string) {
  asm volatile("mov   $0x09, %%ah\n"
               "int   $0x21\n"
               : /* no output */
               : "d"(string)
               : "ah");
}

void to_upper(char *str) {
  for (int i = 0; i < 12; i++) { // str[i] != '$' && str[i] != '\0'; i++) {
    print(progress);
    if (str[i] >= 'a' && str[i] <= 'z') {
      str[i] += 'A' - 'a';
    }
  }
}

void dosmain() {
  to_upper(message);
  // print("Hello, World!\n$");
}
