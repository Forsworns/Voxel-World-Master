创建一个新的工程将src目录下文件拷贝至VS2017项目目录下，在源文件中导入src下文件。

在VS 2017中配置项目属性

- Debug x64（注意配置时与编译时环境均为此）

- C/C++  常规

  “附加包含目录” 中添加(前两个为该项目文件，后面的两个为cuda的头文件，具体位置与cuda环境有关)：

  - .\vendor\tclap-1.2.1\include;
  - .\includes;    
  - \usr\local\cuda\include; 
  - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;

- 链接器

  ”输入“ 中添加：

  - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\cudart_static.lib;
  - kernel32.lib;user32.lib;gdi32.lib;winspool.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;comdlg32.lib;advapi32.lib

最后对项目选择生成依赖项，勾选Cuda 10.0，在main.cu的属性中设置项类型为CUDA C/C++。

编译执行。