# ID3/C4.5/CART 决策树程序

## 编译方法

```
cargo build --release
```

## 运行方法

将输入数据集放置在 `lab2_dataset/car_train.csv`，划分后会生成

```
lab2_dataset/car_train_tr.csv
lab2_dataset/car_train_va.csv
lab2_dataset/car_train_te.csv
```

然后就可以运行算法，自动输出验证集准确率，结果在 `test.csv` 中，也会构建树图形在 `tree.dot`。

**数据集划分**

```
cargo run --release -- sample
```

**ID3/C4.5/CART算法**

```
cargo run --release -- id3
cargo run --release -- c45
cargo run --release -- cart
```

## 代码结构

```
.
|----     Cargo.toml 构建系统配置
|----     Cargo.lock
|----     learn.py Python实现
|----     src
| |----   selector
| | |---- mod.rs 选择器通用代码
| | |---- c45_selector.rs C4.5选择器
| | |---- cart_selector.rs CART选择器
| | |---- id3_selector.rs ID3选择器
| |----   conf.rs 配置
| |----   run.rs 运行ID3/C4.5/CART
| |----   test_utils.rs 测试用辅助函数
| |----   main.rs 主入口
| |----   case.rs 样例结构
| |----   sample.rs 数据集划分
| |----   builder
| | |---- builder_impl.rs 构建树
| | |---- binary_tree.rs 二叉树&遍历
| | |---- mod.rs
| | |---- print.rs 打印树
```

