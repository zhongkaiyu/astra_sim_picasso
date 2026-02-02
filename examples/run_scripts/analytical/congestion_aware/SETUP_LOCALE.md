# 终端中文显示配置指南

## 问题说明
如果在终端中看到中文乱码（如 `������`），这是因为系统缺少中文 locale。

## 解决方案

### 方案 1: 安装中文 Locale（推荐）

```bash
# 1. 安装中文语言包
sudo apt-get update
sudo apt-get install -y language-pack-zh-hans

# 2. 生成中文 locale
sudo locale-gen zh_CN.UTF-8

# 3. 更新 locale
sudo update-locale LANG=zh_CN.UTF-8

# 4. 设置当前会话的 locale
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8
```

### 方案 2: 使用 C.UTF-8（临时方案）

```bash
# 使用系统自带的 UTF-8 locale
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

### 方案 3: 永久设置（添加到 ~/.bashrc）

```bash
# 将以下内容添加到 ~/.bashrc 文件末尾
echo 'export LANG=zh_CN.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=zh_CN.UTF-8' >> ~/.bashrc

# 重新加载配置
source ~/.bashrc
```

## 验证设置

```bash
# 查看当前 locale 设置
locale

# 测试中文显示
echo "测试中文显示"

# 运行脚本测试
./run_DP4_H100.sh
```

## 注意事项

1. 如果在 Docker 容器中，可能需要在 Dockerfile 中添加：
   ```dockerfile
   RUN apt-get update && apt-get install -y locales \
       && locale-gen zh_CN.UTF-8
   ENV LANG=zh_CN.UTF-8
   ENV LC_ALL=zh_CN.UTF-8
   ```

2. 如果没有 sudo 权限，请联系系统管理员安装语言包

3. 脚本已更新为纯英文输出（注释仍为中英双语），以避免显示问题

