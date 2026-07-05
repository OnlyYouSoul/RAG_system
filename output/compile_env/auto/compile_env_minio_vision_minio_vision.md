## 编译运行环境说明

## 整体说明

1. 参赛选手所提交的源码压缩包，由系统解压后自动编译运行，生成竞赛结果。

2. 运行参赛选手源码的操作系统为x86\_64 的Linux 操作系统；

## 二、 编译环境

1. C/C++

make：GNU Make 4.3

gcc：gcc 11.4.0

g++：g++ 11.4.0

2. Java

OpenJDK 1.8.0\_492

3. Python

Python：python 3.10.12(numpy=2.2.6)

## 三、 SDK目录必须包含内容

## 1. C语言

<table><tr><td rowspan=1 colspan=1>文件名</td><td rowspan=1 colspan=1>是否可以修改</td><td rowspan=1 colspan=1>是否需要上传</td><td rowspan=1 colspan=1>说明</td></tr><tr><td rowspan=1 colspan=1>main.c</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>main文件</td></tr><tr><td rowspan=1 colspan=1>results. csv</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>AI模型预测结果</td></tr><tr><td rowspan=1 colspan=1>Model文件夹</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>模型训练和测试的代码(要求可复现results.csv)</td></tr></table>

2. C++
<table><tr><td rowspan=1 colspan=1>文件名</td><td rowspan=1 colspan=1>是否可以修改</td><td rowspan=1 colspan=1>是否需要上传</td><td rowspan=1 colspan=1>说明</td></tr><tr><td rowspan=1 colspan=1> main.cpp</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>main文件</td></tr><tr><td rowspan=1 colspan=1>results. csv</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>AI模型预测结果</td></tr><tr><td rowspan=1 colspan=1>Model文件夹</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>模型训练和测试的代码(要求可复现results.csv)</td></tr></table>

## 3. Java

<table><tr><td rowspan=1 colspan=1>文件名</td><td rowspan=1 colspan=1>是否可以修改</td><td rowspan=1 colspan=1>是否需要上传</td><td rowspan=1 colspan=1>说明</td></tr><tr><td rowspan=1 colspan=1>Main.java</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>main文件</td></tr><tr><td rowspan=1 colspan=1>results. csv</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>AI模型预测结果</td></tr><tr><td rowspan=1 colspan=1>Model文件夹</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>模型训练和测试的代码(要求可复现results.csv)</td></tr></table>

## 4. Python

<table><tr><td rowspan=1 colspan=1>文件名</td><td rowspan=1 colspan=1>是否可以修改</td><td rowspan=1 colspan=1>是否需要上传</td><td rowspan=1 colspan=1>说明</td></tr><tr><td rowspan=1 colspan=1>main.py</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>main文件</td></tr><tr><td rowspan=1 colspan=1>results. csv</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>AI模型预测结果</td></tr><tr><td rowspan=1 colspan=1>Model文件夹</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>模型训练和测试的代码(要求可复现results.csv)</td></tr></table>

说明：  
1. 注意代码文件名main文件和AI模型预测结果文件严格与上述要求文件名一致。  
2. 每种支持的语言限定特定的文件后缀名，请严格保持一致。

C：.c, .h C++: .cpp, .h Java: .java Python: .py

3. 支持多文件编译，C&CPP项目不支持子目录。所有源文件请保持和main文件在同一级目录下。

## 四、 压缩文件说明

1. 选手提交的压缩包应为zip格式，请采用常用的压缩软件压缩。

2. 压缩包直接打开即为项目文件，请注意不要有额外一层目录嵌套。平台不对嵌套目录进行处理。（请参考提供的语言示例提交代码）

例如：

注意：如果只做第一题必须包含results.csv+model文件夹，如果只做第二题必须包含main.c文件。

![image](http://127.0.0.1:9000/mineru-images/compile_env/be2cb8b5f4ca4ff9a3864d3a2fb99eda.jpg)

## 五、 补充说明

1. C/C++

⚫ C/C++工程均采用Makefile编译(系统解压文件后，自动在目录下添加，不需要选手提供)，编译优化级别为O2

⚫ 选手新增代码文件需在一级目录下，否则会编译失败。

不允许引用第三方库。

⚫ C++语言编写的源代码支持C++17特性。

第3页, 共4页

## 2. Java

不允许引用第三方库。

系统支持jdk1.8版本，请在此版本下进行源码开发。

## 3. Python

系统支持python3.10.12版本，请在该版本下进行源码开发。

除numpy外，不允许引用其他第三方库。

参赛选手提交的源码会在系统上直接运行，因此请保证main.py在一级目录下，不可修改该文件名称。

## 4. 其他

提交csv文件支持的编码为UTF-8-SIG，其他文件支持的编码为UTF-8。

压缩包中不能带有任何运行时调用的文件，包括可执行程序、动态库、数据表。平台在运行选手程序之前会清理所有文件，只保留最终程序,故运行时对压缩包中的文件调用均会失败。

⚫ 上传源码压缩包中所包含的目录不得含有以下合法字符集以外的任何字符；

目录名合法的命名字符集：英文大写字母“A-Z”、英文小写字母“a-z”、数字“0-9”、 英文短横线“-”、 英文下划线“\_”。

上传到源码压缩包中所包含的文件名不得含有以下合法字符集以外的任何字符，且“.”符号不得连续出现。

文件名合法的命名字符集：英文大写字母“A-Z”、英文小写字母“a-z”、数字“0-9”、 英文短横线“-”、英文下划线“\_”、英文点“.”。

选手程序的编译运行均在linux下，选手提交前可尝试在linux平台下编译成功，以免造成编译失败。