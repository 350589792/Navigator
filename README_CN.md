# 交通事故案例分析系统 - 部署和使用说明

## 目录
1. [系统要求](#系统要求)
2. [安装步骤](#安装步骤)
3. [使用说明](#使用说明)
4. [常见问题](#常见问题)

## 系统要求

### 后端要求
- Python 3.12 或更高版本
- pip（Python包管理器）
- 建议使用虚拟环境

### 前端要求
- Node.js 18.0 或更高版本
- pnpm（推荐）或 npm

## 安装步骤

### 后端部署（Windows）

1. 安装 Python
   - 访问 [Python官网](https://www.python.org/downloads/)
   - 下载并安装 Python 3.12
   - 安装时勾选"Add Python to PATH"

2. 创建虚拟环境
   ```cmd
   cd backend/traffic_case_api
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. 安装依赖
   ```cmd
   pip install poetry
   poetry install
   ```

### 前端部署（Windows）

1. 安装 Node.js
   - 访问 [Node.js官网](https://nodejs.org/)
   - 下载并安装 Node.js 18 LTS版本

2. 安装 pnpm
   ```cmd
   npm install -g pnpm
   ```

3. 安装依赖
   ```cmd
   cd frontend/traffic_case_ui
   pnpm install
   ```

### 启动应用

1. 启动后端服务
   ```cmd
   cd backend/traffic_case_api
   .\venv\Scripts\activate
   python -m uvicorn app.main:app --reload
   ```

2. 启动前端服务
   ```cmd
   cd frontend/traffic_case_ui
   pnpm dev
   ```

完成后，可以通过浏览器访问 http://localhost:5173 使用系统。

## 使用说明

### 主要功能
1. 案例输入
   - 在文本框中输入交通事故案例描述
   - 案例描述应包含事故发生的具体情况、涉事方信息等

2. 法条匹配
   - 系统会自动分析案例内容
   - 显示相关法律条款及其适用说明
   - 支持多个违法行为的组合分析（如酒驾+超速）

3. 相似案例推荐
   - 展示相似度最高的历史案例
   - 提供案例详细信息和判决结果
   - 可用于参考类似案件的处理方式

### 操作流程
1. 打开系统主页
2. 在输入框中输入或粘贴案例描述
3. 点击"分析"按钮
4. 查看分析结果：
   - 左侧显示相关法条
   - 右侧显示相似案例

## 常见问题

### 1. 安装依赖失败
- 检查网络连接
- 确保已安装最新版本的 pip 和 poetry
- 尝试使用管理员权限运行命令

### 2. npm安装依赖出现ECONNRESET错误
- 尝试切换npm镜像源：
  ```cmd
  npm config set registry https://registry.npmmirror.com/
  ```
- 或使用其他包管理器：
  ```cmd
  # 使用pnpm
  npm install -g pnpm
  pnpm install

  # 或使用yarn
  npm install -g yarn
  yarn install
  ```
- 如果仍然失败，检查防火墙设置和网络代理配置

### 3. 启动服务失败
- 确保所有依赖都已正确安装
- 检查端口 5173 和 8000 是否被占用
- 查看控制台错误信息

### 4. 系统响应缓慢
- 检查网络连接状况
- 确保系统内存充足
- 可能是首次加载模型较慢，请耐心等待

### 5. Windows特定问题
- 如遇到"'vite' 不是内部或外部命令"错误：
  ```cmd
  # 确保在正确目录下
  cd frontend/traffic_case_ui
  # 重新安装依赖
  pnpm install
  # 启动开发服务器
  pnpm dev
  # 或使用npx运行
  npx vite
  ```
- Node.js环境变量问题：
  1. 右键"此电脑" -> "属性" -> "高级系统设置" -> "环境变量"
  2. 检查Path变量中是否包含Node.js安装路径
  3. 如果没有，添加Node.js安装目录（通常在C:\Program Files\nodejs）
- 使用管理员权限运行命令提示符可能解决某些权限问题

如遇其他问题，请查看控制台输出的错误信息，或联系技术支持。
