---
title: PythonNote
mathjax:true
---
# 零碎的Python笔记
[TOC]


## 字符串

### 转义字符串`\`
- 字符串可以用`' '`和`" "`和`'''`表示
- 在这一章节我们解决，假如字符串中本身含有'和"时的表示方法
#### 表示字符串
1. 当字符串中含有`'`时，可以用双引号`"`括起来
> 例如，"I'm OK"
2. 当字符串中含有`"`时，可以用单引号`'`括起来
> 例如，'Summer said,"So where is it？"'
3. 如果既包含`'`又包含`"`,则需要对字符进行转义
> 例如，'Bob said \"I\'m OK\".'
> (被markdown吞掉了，请看源码)
> 想要显示出来，应该是'Bob said \\"I\\'m OK\\".'
4. 常用的转义字符

| 符号|	表示意义|
|------|:--------|
|`\n`	|表示换行|
|`\t`	|表示一个制表符|
| `\\`	|表示 `\ `字符本身|
#### raw字符串
当一句话里有很多需要转义的字符，一个一个转就很麻烦；这时我们要把整句话当做raw字符串.有两种表示方法,`r'your_character'`或者`r"your_character"`
> print r"I'm OK"

但是，`r'your_character'`或者`r"your_character"`表示法不适用于同时出现`'`和`"`的情况；也不能表示多行字符串

#### 多行字符串
要表示多行字符串，可以用'''your_character'''表示，例如
> '''这是第一行
> 第二行可以分开哦
> 第三行很棒呢'''

当然，我们也可以这样表示
> '这是第一行\n第二行可以分开哦\n第三行很棒呢'

两种输出是一样的。
在多行字符前也可以加上r，使多行字符成为raw字符串
> r'''I'm very goog.
"Higenbana" is not good.'''
### 字符串的运算
下面来详细介绍一下字符串运算符

|运算符	|描述|
|--------|:----------------|
|+	|连接符|
|*	|重复输出|
| []	|通过索引获取字符串中字符|
|[ : ]	|获取字符串中的一部分,左开右闭|
|in	|字符串中是否包含指定字符|
|not in	|字符串中是否不包含指定字符|
|`r` 或者 `R` |字符串原样输出|

输入
> s1 = 'Hello'
s2 = 'Python'
print('s1 + s2 -->', s1 + s2)
print('s1 * 2 -->', s1 * 2)
print('s1[0] -->', s1[0])
print('s1[0:2] -->',s1[0:2])
print('"H" in s1 -->','H' in s1)
print('"H" not in s1 -->','H' not in s1)
print('\\r -->', R'\r')

输出结果
> s1 + s2 --> HelloPython
s1 * 2 --> HelloHello
s1[0] --> H
s1[0:2] --> He
"H" in s1 --> True
"H" not in s1 --> False
\r --> \r
### 格式化、占位符

当我们需要输出的内容中含有变量时，比如：`Hello xxx`，其中`xxx` 为变量，此时便需要一种格式化字符串的方式，Python 使用 % 格式化字符串，常用占位符如下表所示：
|占位符|描述|
|------|--------|
|%s	|格式化字符串|
|%d	|格式化整数|
|%f	|格式化浮点数|

以字符串为例,输入
> print('Hello %s' % 'Python')

我们也可以使用字符串的 `format()`方法进行格式化
> print('{0} {1}'.format('Hello', 'Python'))

这种方式是用传入的参数依次替换字符串内的占位符{0}、{1} ...


## 序列

序列的定义：序列是一块可存放多个值的连续内存空间，所有值按一定顺序排列，每个值所在位置都有一个编号，称其为索引，我们可以通过索引访问其对应值





## list列表
list是一种有序的集合，可以随时添加和删除其中的元素
### 创建list
```[element(1) ,element(2) , … , element(n)]```
### 访问list
#### 按照索引访问list
1. 序列是Python中最基本的数据结构。序列中的每个元素都分配一个数字 - 它的位置，或索引，第一个索引是0，第二个索引是1，依此类推
```
classmate = ['chenchen' , 'qiutian' , 'zhouzihan']
输入classmate[0] 代表 'chenchen' 
```
2. 截取一段list
```
list_name = [1,2,3,4,5,6,7,8]
输入list_name[2:6] 输出是[3,4,5,6]
```
注意！！ [2:6]表示的是列表中的第2个下标到第6个下标，左闭右开区间

#### 倒叙访问
可以根据倒数下标进行索引，在Python中倒数的表示符号为`-`号，倒数第一，下标就为`-1`
```
list_name = [1,2,3,4,5,6,7,8]
输入list_name[-1] 输出是 8
```
### list增删查改
#### 修改
找到对应的下标然后通过索引到该位置进行赋值
```
L=[1,2,3,4,5,6,7,8]
L[3]=0
```
#### 增加
依靠方法append来实现，调用方式`变量名.append(元素)`,添加方式是尾插，即list最后面插入

#### 删除
当list要删除某个元素的时候需要借助方法del来实现，调用方式`del 变量名[下标位置]`,选择的下标位置的元素将被删除
### list常用函数

|函数名 |作用|
|---------|:--------------------|
|cmp(list1,list2)	|比较两个list的元素，返回值等于0时表示两个list相等|
|len(list_name)	|求取list的长度|
|max(list_name)	|返回list的最大值|
|min(list_name_)	|返回list的最小值|
|list(seq_name)|将元组转化为列表|

### python包含的函数

|函数名 |作用|
|---------|:--------------------|
|list_name.append(obj)	|尾插新对象|
|list_name.count(obj)	|统计对象在列表中出现的次数|
|list_name.extend(list1)	|用新列表扩展现有的列表（list末尾一次性追加一个list1）|
|list_name.insert(index,obj)	|将对象obj插入到列表的index位置中去|
|list_name.pop()	|若括号内为空，则删除列表中最后一个位置的元素，并返回删除的元素值（尾删）；若括号内为`n`，则删除索引为`n`的值|
|list_name.remove(obj)	|删除列表中第一次出现的对象obj(相同元素，删除最前面的那一个，如果只有一个就直接删除此obj)|
|list_name.reverse()	|逆序操作（反向列表中的所有元素）|
|list_name.sort()	|默认升序，list.sort(reverse=True)降序，事实上sort有三个参数cmp, key, reverse，后面遇到再讨论|



## 元组(Tuple)
- tuple是另一种有序的列表，也称为“ 元组 ”。tuple 和 list 非常类似，但是，tuple一旦创建完毕，就不能修改了。
同样是创建班级同学的名称
```
t = ('Zhangsan', 'Lisi', 'Wangwu')
```
- 创建tuple和创建list唯一不同之处是用( )替代了[ ]。
- 创建好了 t 就不能改变了，tuple没有 append()方法，也没有insert()和pop()方法。所以，新同学没法直接往 tuple 中添加，老同学想退出 tuple 也不行。
- 获取 tuple 元素的方式和 list 是一模一样的，我们可以正常使用 t[0]，t[-1]等索引方式访问元素，但是不能赋值成别的元素。






## `try` `except1`语句的用法,异常类型

- 把可能错误的语句放在try中，用except处理异常
- 每一个try，都必须至少有一个except
### 1 万能异常Exception
```
s1 = 'hello'
try:
    int(s1)
except Exception as e:
    print(e)
```

### 2 异常类只能来处理指定的异常情况，如果非指定异常则无法处理

```
s1 = 'hello'
try:
    int(s1)
except IndexError as e: # 未捕获到异常，程序直接报错
    print (e)
```
#### 注：Python的异常类型
```
BaseException       所有异常的基类     
 +-- SystemExit       解释器请求退出
 +-- KeyboardInterrupt     用户中断执行(通常是输入^C)
 +-- GeneratorExit      生成器(generator)发生异常来通知退出
 +-- Exception        常规错误的基类
   +-- StopIteration       迭代器没有更多值 
   +-- StopAsyncIteration       必须通过异步迭代器对象的__anext__()方法引发以停止迭代
   +-- ArithmeticError         所有数值计算错误的基类
   |  +-- FloatingPointError       浮点计算错误
   |  +-- OverflowError         数值运算超出最大限制
   |  +-- ZeroDivisionError       除(或取模)零 (所有数据类型
   +-- AssertionError         断言语句失败
   +-- AttributeError         对象没有这个属性
   +-- BufferError          与缓冲区相关的操作时引发
   +-- EOFError            没有内建输入,到达EOF 标记
   +-- ImportError           导入失败
   |  +-- ModuleNotFoundError    找不到模块
   +-- LookupError           无效数据查询的基类
   |  +-- IndexError           序列中没有此索引(index)
   |  +-- KeyError            映射中没有这个键
   +-- MemoryError           内存溢出错误
   +-- NameError            未声明、初始化对象
   |  +-- UnboundLocalError       访问未初始化的本地变量
   +-- OSError             操作系统错误，
   |  +-- BlockingIOError        操作将阻塞对象设置为非阻塞操作
   |  +-- ChildProcessError       子进程上的操作失败
   |  +-- ConnectionError        与连接相关的异常的基类
   |  |  +-- BrokenPipeError       在已关闭写入的套接字上写入
   |  |  +-- ConnectionAbortedError   连接尝试被对等方中止
   |  |  +-- ConnectionRefusedError   连接尝试被对等方拒绝
   |  |  +-- ConnectionResetError    连接由对等方重置
   |  +-- FileExistsError        创建已存在的文件或目录
   |  +-- FileNotFoundError       请求不存在的文件或目录
   |  +-- InterruptedError       系统调用被输入信号中断
   |  +-- IsADirectoryError       在目录上请求文件操作
   |  +-- NotADirectoryError      在不是目录的事物上请求目录操作
   |  +-- PermissionError       在没有访问权限的情况下运行操作
   |  +-- ProcessLookupError      进程不存在
   |  +-- TimeoutError         系统函数在系统级别超时
   +-- ReferenceError        弱引用试图访问已经垃圾回收了的对象
   +-- RuntimeError         一般的运行时错误
   |  +-- NotImplementedError   尚未实现的方法
   |  +-- RecursionError      解释器检测到超出最大递归深度
   +-- SyntaxError          Python 语法错误
   |  +-- IndentationError     缩进错误
   |     +-- TabError     Tab 和空格混用
   +-- SystemError       一般的解释器系统错误
   +-- TypeError        对类型无效的操作
   +-- ValueError       传入无效的参数
   |  +-- UnicodeError       Unicode 相关的错误
   |     +-- UnicodeDecodeError   Unicode 解码时的错误
   |     +-- UnicodeEncodeError   Unicode 编码时错误
   |     +-- UnicodeTranslateError Unicode 转换时错误
   +-- Warning            警告的基类
      +-- DeprecationWarning     关于被弃用的特征的警告
      +-- PendingDeprecationWarning  关于构造将来语义会有改变的警告
      +-- RuntimeWarning      可疑的运行行为的警告
      +-- SyntaxWarning      可疑的语法的警告
      +-- UserWarning       用户代码生成的警告
      +-- FutureWarning      有关已弃用功能的警告的基类
      +-- ImportWarning      模块导入时可能出错的警告的基类
      +-- UnicodeWarning      与Unicode相关的警告的基类
      +-- BytesWarning       bytes和bytearray相关的警告的基类
      +-- ResourceWarning      与资源使用相关的警告的基类

```

### 2 多分支

```
s1 = 'hello'
try:
    int(s1)
except IndexError as e:
    print(e)
except KeyError as e:
    print(e)
except ValueError as e:
    print(e)
```
### 3 多分支+Exception
```
s1 = 'hello'
try:
    int(s1)
except IndexError as e:
    print(e)
except KeyError as e:
    print(e)
except ValueError as e:
    print(e)
except Exception as e:
    print(e)
```

## Numpy库
### np.dot 和np.matmul 的区别
在`NumPy`中，`np.matmul`和`np.dot`都用于执行矩阵乘法操作
- `np.matmul`：主要用于执行矩阵的乘法运算，它严格遵循线性代数中的矩阵乘法规则。
- `np.dot`：用于计算两个数组的点积，功能更为通用，既可以处理一维数组的点积，也可以处理多维数组的乘法。
**一般而言,**
- 当处理二维矩阵乘法时，`np.matmul` 和 `np.dot` 功能基本相同。
- 处理一维数组时，使用 `np.dot `计算点积。
- 处理多维数组时，`np.matmul` 更适合进行批量矩阵乘法，而 `np.dot `适用于需要对最后两个维度进行点积的场景。
- 进行标量乘法时，使用 `np.dot`。


## TensorFlow库
### Epochs(轮次) and batches(批次)
1. Epoch轮次
一个 `Epoch` 表示整个训练数据集完整地通过模型一次的过程。在训练模型时，通常不会只让数据集通过模型一次，而是会进行多个 `Epoch` 的训练。因为只训练一个 `Epoch` 可能不足以让模型学习到数据中的所有模式和特征，通过多个 `Epoch` 的训练，模型可以不断调整其内部的参数，从而逐渐提高性能。
- ==适当增加 `Epochs` 数量==通常可以提高模型的性能，因为模型有更多机会学习数据中的模式。
- 但如果 `Epochs` 数量过多，==模型可能会出现过拟合现象==，即在训练集上表现很好，但在测试集上表现不佳。
- ==`Epochs` 数量越多，训练所需的时间就越长。==
例如
```
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0

# 构建简单的神经网络模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，设置 epochs 为 5
model.fit(x_train, y_train, epochs=5)
```
2. batches(批次)
在训练过程中，由于内存限制，通常不会将整个训练数据集一次性输入到模型中进行训练，而是将数据集分成若干个小的子集，每个子集就称为一个 Batch（批次）。模型会依次对每个 Batch 进行训练，更新模型的参数。
- Tensorflow中批处理的默认大小为32
- 在一个 Epoch 中，模型会依次处理所有的 Batch。假设训练数据集有 1000 个样本，Batch Size 为 100，那么一个 Epoch 中模型会进行 10 次 Batch 的训练（1000 ÷ 100 = 10）。如果设置 Epochs 为 5，那么整个训练过程中模型会对这 10 个 Batch 重复训练 5 次。









## copy库
### 浅拷贝 和 深拷贝
==修改浅拷贝,原始列表也修改;修改深拷贝,原始列表不改变==
- 浅拷贝（`copy.copy`）：只复制对象本身，而不复制其包含的子对象。也就是说，新对象和原对象虽然是不同的对象，但它们内部的子对象可能是共享的。
- 深拷贝（`copy.deepcopy`）：会递归地复制对象及其所有子对象，创建一个完全独立的新对象，新对象和原对象没有任何共享的部分

例如代码
```
import copy

# 定义一个嵌套列表
original_list = [[1, 2, 3], [4, 5, 6]]
# 浅拷贝
shallow_copy = copy.copy(original_list)
# 深拷贝
deep_copy = copy.deepcopy(original_list)
# 修改浅拷贝中的子列表
shallow_copy[0][0] = 99
# 修改深拷贝中的子列表
deep_copy[0][0] = 100

print("原始列表:", original_list)
print("浅拷贝:", shallow_copy)
print("深拷贝:", deep_copy)
```
输出结果
```
原始列表: [[99, 2, 3], [4, 5, 6]]
浅拷贝: [[99, 2, 3], [4, 5, 6]]
深拷贝: [[100, 2, 3], [4, 5, 6]]
```
总结:
- 修改 `shallow_copy` 中的子列表元素，发现原始列表 `original_list` 也被修改了，这是因为浅拷贝只复制了外层列表，内部子列表是共享的。
- 修改 `deep_copy`中的子列表元素，原始列表 `original_list` 不受影响，说明深拷贝创建了一个完全独立的新对象

#### 浅拷贝深拷贝:如何使用?
Python中的赋值默认是浅拷贝，也就是引用传递。对于不可变对象（如整数、浮点数、字符串等），直接赋值不会有问题，因为修改时会创建新的对象。而像列表或数组这样的可变对象，直接赋值会导致新旧变量指向同一内存地址，修改其中一个会影响另一个



