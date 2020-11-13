# Lab 3

## 编译方法

```
mkdir -p build
cd build
cmake ..
make
```

若要使用基于 `atomic` 的求和，可以使用

```
cmake -DSUM_USE_ATOMIC=true ..
```

## 运行方法

不加参数运行时，会有提示。

### 矩阵乘法

```
./matrix 1024 1024 1024 8
```

### 数组求和

```
./sum 10 8
```

第一个参数是每次获得的元素数量，第二个参数是并行线程数。

### 求解一元二次方程

```
./equation 1 2 1
```

分别是 $a, b, c$ 三个参数。

### Monte-Carlo 方法求面积

```
./area 1000000 8
```

分别是迭代次数和线程数。
