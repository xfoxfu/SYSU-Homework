# mysqlclient的问题
windows平台，在`src`目录下，编译命令示例如下
```shell
g++ main.cpp mysqlclient.cpp libmysql.dll -o main.exe
```
cmake一直过不了:joy:
# 数据库结构和数据
`db/bookstore.sql`是fdl的，`db/bookstore_with_data.sql`后面加了某些字段的自动增长和数据的，导入时使用`db/bookstore_with_data.sql`。