# Git & GitHub 基础教程

## 为什么需要 Git

### 遇到的问题

在没有版本控制工具之前，程序员面临这些问题：

1. **代码备份混乱**：`final.py`, `final_v2.py`, `final_final.py`, `final_really_final.py`...
2. **协作困难**：多人修改同一个文件，不知道谁改了什么，容易互相覆盖
3. **无法回退**：改坏了代码，想回到之前的版本，但找不到了
4. **分支开发困难**：想尝试新功能，但又怕破坏现有代码

### Git 的解决方案

Git 是一个**本地版本控制系统**，安装在你的电脑上，用于：
- 记录每次代码修改的历史
- 随时回退到任何历史版本
- 创建分支进行实验性开发
- 合并不同分支的代码

**重点**：Git 是本地工具，即使没有网络，你也可以使用 Git 管理代码。

## Git vs GitHub：核心区别

### Git（本地工具）
- 安装在你的电脑上
- 管理本地代码的版本历史
- 离线也能使用
- 免费开源

### GitHub（远程托管平台）
- 在线的代码托管服务
- 存储代码的远程备份
- 提供团队协作功能
- 需要网络访问（国内需要梯子）

**类比**：Git 就像你电脑上的 Word，GitHub 就像 Google Docs（在线文档服务）。

## 安装和配置 Git

### 安装 Git
```bash
# Linux (Ubuntu/Debian)
sudo apt install git

# macOS
brew install git

# Windows
# 下载安装包：https://git-scm.com/download/win
```

### 配置用户信息
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### 查看配置
```bash
git config --list
```

## 拿到一个 GitHub 仓库后的流程

### 场景：你在 GitHub 上发现了一个有用的项目

### 1. Fork 还是直接 Clone？

**直接 Clone（克隆）**
- 适用场景：只是想学习代码，不打算贡献
- 操作简单，直接下载到本地

```bash
git clone https://github.com/原作者/仓库名.git
```

**Fork（复刻）**
- 适用场景：想要修改代码并贡献给原项目，或者基于原项目做自己的开发
- 会在你的 GitHub 账号下创建一个副本
- 你可以自由修改你的副本，不影响原仓库

Fork 流程：
1. 在 GitHub 网页上点击右上角的 "Fork" 按钮
2. 等待 GitHub 复制仓库到你的账号下
3. Clone 你自己的副本到本地

```bash
git clone https://github.com/你的用户名/仓库名.git
```

**什么时候选择 Fork？**
- 你想保留一份自己可控的副本
- 你想给开源项目提交代码（Pull Request）
- 你想基于别人的项目做自己的修改

## Git 基本工作流程

### 1. Clone（克隆仓库）

```bash
git clone https://github.com/用户名/仓库名.git
cd 仓库名
```

### 2. 查看状态

```bash
git status
```

这个命令会告诉你：
- 哪些文件被修改了
- 哪些文件在暂存区
- 当前在哪个分支

### 3. Add（添加到暂存区）

修改代码后，需要先把文件添加到"暂存区"：

```bash
git add 文件名.py          # 添加单个文件
git add .                  # 添加所有修改的文件
git add *.py               # 添加所有 .py 文件
```

**暂存区的作用**：可以选择性地提交文件，不是所有修改都要一次性提交。

### 4. Commit（提交到本地仓库）

```bash
git commit -m "描述你做了什么修改"
```

**好的提交信息示例**：
- "添加用户登录功能"
- "修复数据库连接错误"
- "优化图片加载速度"

**不好的提交信息**：
- "修改"
- "update"
- "aaa"

### 5. Push（推送到远程仓库）

```bash
git push
```

第一次推送可能需要：
```bash
git push -u origin main
```

**Push 的作用**：把你本地的提交上传到 GitHub，这样：
- 代码有了远程备份
- 其他人可以看到你的修改
- 团队成员可以拉取你的代码

### 6. Pull（拉取远程更新）

```bash
git pull
```

**Pull 的作用**：
- 获取其他人推送的代码
- 保持本地代码与远程同步

**重要**：在开始工作前，先 `git pull`，避免代码冲突。

## 分支管理（Branch）

### 为什么需要分支？

想象你在开发一个网站：
- 主分支（main）是稳定的生产代码
- 你想添加一个新功能，但不确定会不会成功
- 创建一个新分支，在上面实验，不影响主分支
- 功能开发成功后，合并回主分支

### 查看分支

```bash
git branch              # 查看本地分支
git branch -a           # 查看所有分支（包括远程）
```

### 创建并切换分支

```bash
git checkout -b 新分支名
```

例如：
```bash
git checkout -b feature-login    # 创建并切换到 feature-login 分支
```

### 切换分支

```bash
git checkout 分支名
```

### 合并分支

```bash
git checkout main              # 切换到主分支
git merge feature-login        # 把 feature-login 合并到 main
```

### 删除分支

```bash
git branch -d 分支名
```

## 完整的协作流程示例

### 场景：给开源项目贡献代码

1. **Fork 原仓库**（在 GitHub 网页操作）

2. **Clone 你的 Fork**
```bash
git clone https://github.com/你的用户名/仓库名.git
cd 仓库名
```

3. **创建新分支**
```bash
git checkout -b fix-bug-123
```

4. **修改代码**
```bash
# 编辑文件...
```

5. **提交修改**
```bash
git add .
git commit -m "修复了 issue #123 的 bug"
```

6. **推送到你的 Fork**
```bash
git push -u origin fix-bug-123
```

7. **在 GitHub 上创建 Pull Request**
- 访问你的 Fork 页面
- 点击 "Pull Request"
- 选择你的分支，提交给原仓库

8. **等待原作者审核和合并**

## 常用命令速查

```bash
git status              # 查看状态
git add .               # 添加所有修改
git commit -m "说明"    # 提交
git push                # 推送到远程
git pull                # 拉取远程更新
git log                 # 查看提交历史
git log --oneline       # 简洁的提交历史
git diff                # 查看修改内容
git checkout -b 分支名  # 创建并切换分支
git branch              # 查看分支
git merge 分支名        # 合并分支
```

## .gitignore 文件

告诉 Git 哪些文件不需要跟踪：

```
# Python
__pycache__/
*.pyc
*.pyo
.env
venv/
.venv/

# IDE
.vscode/
.idea/

# 操作系统
.DS_Store
Thumbs.db

# 敏感信息
config.ini
secrets.json
```

## 注意事项

1. **不要提交敏感信息**：密码、API密钥、私钥等
2. **提交前先 pull**：避免代码冲突
3. **提交信息要清晰**：让别人知道你做了什么
4. **经常提交**：不要等到改了一大堆才提交
5. **使用分支**：不要直接在 main 分支上开发
6. **GitHub 需要梯子**：国内访问 GitHub 非常不稳定，需要科学上网工具

## 学习资源

- Git 官方文档：https://git-scm.com/doc
- GitHub 官方教程：https://docs.github.com/cn
- 交互式学习：https://learngitbranching.js.org/
