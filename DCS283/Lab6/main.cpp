#include <cstring>
#include <iostream>
#include <mysql.h>
#include <string>

void mysql_check_error(MYSQL &db);
void create_course_table(MYSQL &db);
void insert_rows_into_course_table(MYSQL &db);

int main(int, char **) {
  try {
    MYSQL db;
    mysql_init(&db);
    mysql_check_error(db); // check for error
    // connect to database
    mysql_real_connect(&db, "localhost", "root", "", "jxgl", 0, nullptr, 0);
    mysql_check_error(db);

    bool exit_flag = false;
    while (!exit_flag) {
      std::cout << "Choose an operation:" << std::endl;
      std::cout << "0 - exit" << std::endl;
      std::cout << "1 - create course table" << std::endl;
      std::cout << "2 - insert into course table" << std::endl;

      // read operation
      std::string op_code_str;
      std::getline(std::cin, op_code_str);
      int op_code = std::stoi(op_code_str);

      // perform operation
      try {
        switch (op_code) {
        case 0:
          exit_flag = true;
          break;
        case 1:
          create_course_table(db);
          break;
        case 2:
          insert_rows_into_course_table(db);
          break;
        default:
          std::cout << "Invalid operation." << std::endl;
          break;
        }
      } catch (std::logic_error &err) {
        std::cout << "An error occurred: " << err.what() << std::endl;
        return -1;
      }
    }
  } catch (std::logic_error &err) {
    std::cout << "An error occurred: " << err.what() << std::endl;
    return -1;
  }
  return 0;
}

// check if any error occurred in the last query
void mysql_check_error(MYSQL &db) {
  if (mysql_errno(&db) != 0) {
    throw std::logic_error(mysql_error(&db));
  }
}

void create_course_table(MYSQL &db) {
  // check if table already exists
  auto list_res = mysql_list_tables(&db, "course");

  if (list_res->row_count > 0) {
    std::cout
        << "`course` table already exists, do you want to delete it first?"
        << std::endl;
    std::cout << "[Y/N]:" << std::flush;
    std::string confirm;
    std::getline(std::cin, confirm);

    if (confirm == "Y" || confirm == "y") {
      // remove existing table
      mysql_query(&db, "DROP TABLE course;");
      mysql_check_error(db);
    }
  }

  // create table
  mysql_query(&db, "CREATE TABLE `course` (\n"
                   "  `cno` BIGINT NOT NULL AUTO_INCREMENT,\n"
                   "  `cname` VARCHAR(64) DEFAULT NULL,\n"
                   "  `cpno` BIGINT DEFAULT NULL,\n"
                   "  `credit` SMALLINT DEFAULT NULL,\n"
                   "  PRIMARY KEY (`cno`),\n"
                   "  CONSTRAINT `course_cpno_fk` FOREIGN KEY (`cpno`) "
                   "REFERENCES `course` (`cno`)\n"
                   ") DEFAULT CHARSET=utf8mb4;");
  mysql_check_error(db);
  std::cout << "Created table successfully;" << std::endl;
}

void insert_rows_into_course_table(MYSQL &db) {
  std::cout << "Inserting into `course` table" << std::endl;
  // read course name (name)
  std::string name;
  std::cout << "course name: " << std::flush;
  std::getline(std::cin, name);
  // read course credit (credit)
  std::cout << "course credit: " << std::flush;
  std::string credit_str;
  std::getline(std::cin, credit_str);
  short credit = std::stoi(credit_str);
  // read course parent (cpno)
  std::string parent_str;
  std::cout << "course parent (empty for null): " << std::flush;
  std::getline(std::cin, parent_str);

  // prepare statement
  MYSQL_STMT *query = mysql_stmt_init(&db);
  mysql_check_error(db);
  const char *str = "INSERT INTO course (cname, credit, cpno) VALUES (?, ?, "
                    "?);";
  mysql_stmt_prepare(query, str, strlen(str));
  mysql_check_error(db);

  // bind parameters
  MYSQL_BIND params[3];
  memset(params, 0, sizeof(params));
  // cname
  params[0].buffer_type = MYSQL_TYPE_VAR_STRING;
  params[0].buffer = const_cast<char *>(name.c_str());
  params[0].is_null = nullptr;
  unsigned long name_len = name.length();
  params[0].length = &name_len;
  // credit
  params[1].buffer_type = MYSQL_TYPE_SHORT;
  params[1].buffer = &credit;
  params[1].is_null = nullptr;
  params[1].length = nullptr;
  // cpno
  params[2].buffer_type = MYSQL_TYPE_SHORT;
  short parent_val = !parent_str.empty() ? atoi(parent_str.c_str()) : 0;
  params[2].buffer = &parent_val;
  bool is_null = parent_str.empty();
  params[2].is_null = &is_null;
  params[2].length = nullptr;
  mysql_stmt_bind_param(query, params);
  mysql_check_error(db);

  // execute
  mysql_stmt_execute(query);
  mysql_check_error(db);
  // check results
  std::cout << mysql_stmt_affected_rows(query) << " rows affected."
            << std::endl;
}
