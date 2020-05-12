char message[12] = "Aa*bCc*d**$";
char progress[3] = ".$";

short count(char *str) {
  short count = 0;
  for (short i = 0; str[i] != '$' && str[i] != '\0'; i++) {
    if (str[i] == '*')
      count++;
  }
  return count;
}

short dosmain(char *str) { return count(str); }
