#include <cstdint>
#include <ctime>

#define BUF_LEN 100
#define USER_NAME_LEN 20
#define EMAIL_LEN 80
#define TIME_BUF_LEN 30

typedef struct Person {
  int8_t username[USER_NAME_LEN]; // 员工名
  int32_t level;                  // 工资级别
  int8_t email[EMAIL_LEN];        // email地址
  uint32_t sendtime;              // 发送时间
  uint32_t regtime;               // 注册时间
} Person;
