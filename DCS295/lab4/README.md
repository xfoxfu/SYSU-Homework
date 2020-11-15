# Lab 4

## 编译方法

```
mkdir -p build
cd build
cmake ..
make
```

## 运行方法

不加参数运行时，会有提示。

### 矩阵乘法

```
./matrix 1024 1024 1024
```

会输出使用五种方法的运行时间，并且进行计算准确性检查。

- 默认调度 OpenMP
- 静态调度 OpenMP
- 动态调度 OpenMP
- `parallel_for`
- 朴素矩阵乘法