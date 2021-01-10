# 平台的问题
windows平台，在`src`目录下，g++编译命令示例如下
```shell
g++ main.cpp sell.cpp stock.cpp refund.cpp mysqlclient.cpp libmysql.dll .\table_printer.cpp view.cpp ..\vendor\fmt\src\format.cc ..\vendor\fmt\src\os.cc .\controller\book.cpp .\controller\provider.cpp .\controller\offer.cpp ui.cpp -I../include -I../vendor/fmt/include -o main.exe
```
cmake一直过不了:joy:
# 数据库结构和数据
`db/bookstore.sql`是fdl的，`db/bookstore_with_data.sql`后面加了某些字段的自动增长和数据的，导入时使用`db/bookstore_with_data.sql`。