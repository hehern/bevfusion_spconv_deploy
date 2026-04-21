# 编译器设置
CC = nvcc
CFLAGS = -std=c++14 -O3

# 目标文件和可执行文件名
TARGET = device
SOURCE = device.cpp

# 默认构建目标
all: $(TARGET)

# 编译规则
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $<

# 清理生成文件
clean:
	rm -f $(TARGET)

.PHONY: all clean