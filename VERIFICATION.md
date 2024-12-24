# 本地部署验证步骤（Windows系统）

## 前置条件检查

1. Python环境要求
   - 安装Python 3.12（推荐）或更高版本
   - 下载地址：https://www.python.org/downloads/
   - 安装时勾选"Add Python to PATH"

2. 端口可用性检查
   ```cmd
   # 检查所有需要的端口
   netstat -ano | findstr LISTENING | findstr "5000"
   netstat -ano | findstr LISTENING | findstr "5173"
   netstat -ano | findstr LISTENING | findstr "8000"
   
   # 或者一次性检查所有端口
   netstat -ano | findstr LISTENING | findstr /C:"5000" /C:"5173" /C:"8000"
   ```
   
   注意：应用需要以下端口：
   - 5000：主应用服务器
   - 5173：开发模式前端服务（如果适用）
   - 8000：API服务（如果适用）
   如果看到以下类似输出，表示端口被占用：
   ```
   TCP    127.0.0.1:5000    0.0.0.0:0    LISTENING    1234
   ```
   关闭占用端口的进程：
   ```cmd
   taskkill /PID 进程ID /F
   ```

## 详细部署步骤

1. 解压项目文件
   - 将traffic-case-analyzer.tar.gz解压到任意目录
   - 打开命令提示符(cmd)，进入解压后的目录

2. 创建虚拟环境（在项目目录下执行）
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
   确认看到(venv)前缀表示虚拟环境激活成功

3. 安装依赖包（确保联网）
   ```cmd
   pip install -r requirements.txt
   ```
   安装完成后可断开网络，后续操作均支持离线

4. 数据文件确认
   - 检查data/cases.json存在
   - 确认文件包含500条交通事故案例数据：
     ```cmd
     # PowerShell方式统计案例数量
     (Get-Content .\data\cases.json -Encoding UTF8 | ConvertFrom-Json).Length
     
     # 预期输出：500
     ```
   - 文件编码为UTF-8格式（可在记事本中打开并另存为，确认编码为UTF-8）

5. 启动应用
   ```cmd
   python app.py
   ```
   正确启动会看到：
   ```
   * Running on http://127.0.0.1:5000
   * Debug mode: on
   ```

## 验证清单

1. 基础检查
   - [ ] 服务器成功启动在127.0.0.1:5000
   - [ ] 浏览器能访问http://127.0.0.1:5000
   - [ ] 页面正常加载，无JavaScript错误

2. 功能验证
   - [ ] 页面布局正确：
     * 顶部显示案例输入框
     * 底部左侧显示相关法条
     * 底部右侧显示相似案例
   - [ ] 输入测试案例并验证结果：
     ```
     案例：2023年8月在城市道路上，驾驶人酒后驾驶机动车，经检测血液酒精含量为80mg/100ml。
     预期结果：
     - 左侧显示相关法条（如《道路交通安全法》关于酒驾的规定）
     - 右侧显示类似酒驾案例
     ```
   - [ ] 确认相似案例从500个案例库中匹配
   - [ ] 所有功能在断开互联网的情况下正常工作

## 离线运行说明

本应用完全支持离线运行：
1. 内置500条完整的交通事故案例数据
2. 所有数据存储在本地data/cases.json
3. 不需要外部API或互联网连接
4. 所有依赖包都在requirements.txt中列出
5. 分析逻辑完全在本地执行

## 文件路径说明
- 数据文件：data/cases.json（500条案例）
- 静态文件：app/static/index.html
- 主程序：app.py
- 所有路径使用跨平台兼容的相对路径

## 常见问题排查

1. 启动失败
   - 确认Python 3.12已正确安装
   - 检查端口5000是否被占用
   - 确认虚拟环境已激活(venv)

2. 页面无法访问
   - 确认服务器已启动
   - 检查防火墙设置
   - 尝试使用不同浏览器

3. 数据相关
   - 确保cases.json存在且格式正确
   - 确认文件包含500条案例数据
   - 检查文件编码为UTF-8

4. 依赖问题
   - 重新运行pip install -r requirements.txt
   - 确保所有依赖包安装成功
   - 检查Python版本兼容性

如遇其他问题，请检查控制台错误信息并对照上述步骤排查。
