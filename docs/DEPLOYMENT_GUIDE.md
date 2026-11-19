# 校园网内的 PaperAgent 服务部署

## 1. 目标与技术选择

目标是在你的校园网电脑上运行一个服务，让处于可达网络的同学通过浏览器链接使用。链接为 `http://主机校园网IPv4:端口`。主机承担解析、检索和接口服务，通义/DeepSeek 请求由主机发出；同学的浏览器不直接接触模型密钥。

| 方式 | 优点 | 代价与限制 |
| --- | --- | --- |
| Python 虚拟环境直接运行 | 沿用本机已验证环境、便于断点调试、网络与宿主一致 | 换电脑要重新安装依赖，需要手动管理进程 |
| Docker Compose（本次提供） | 固定 Python/依赖，统一启动、健康检查和持久化目录 | Windows 需要 Docker Desktop/WSL2，增加内存开销，容器校园权限需独立验证 |
| 公网云服务器 | 有稳定公网入口，不依赖本机开机 | 增加运维费用，通常不具备学校出口与机构全文权限 |
| 多进程集群 | 可扩展吞吐量 | 需要重新设计跨进程会话锁、队列、数据库和向量服务，当前项目未实现 |

当前继续使用一个 Uvicorn 服务进程，内部用线程领取 SQLite 任务、用子进程解析 PDF。`thread_id` 隔离会话图状态，但没有账号系统；文档库和会话列表为共享空间。请按共享学习资料使用。

## 2. 新增文件的职责

- `Dockerfile`：Python 3.14 Linux 镜像，安装 CPU OCR/ONNX 所需系统库和 Python 依赖；非 root 用户运行，健康检查访问 `/health`。
- `compose.yaml`：构建镜像，注入 `.env`，发布端口，挂载命名卷；正常退出最长等待 5 分钟。
- `.dockerignore`：排除密钥、虚拟环境、个人 PDF、数据库、模型及实验产物，避免把本地数据装进镜像。
- `.env.example`：记录 API、模型、预算、路径、代理开关和宿主端口。
- `pdfprocess/paths.py`：默认路径及运行时向量目录读取；PDF 配置、检索 CLI 与模型缓存共同遵守环境设置。

当前机器没有检测到 Docker CLI/Desktop，因此本次完成的是配置与静态核对，尚未获得镜像构建、容器健康或同学跨设备访问的实测结果。Python 服务和自动化测试另行验证。

## 3. 首次启动

安装 Docker Desktop，启用 Linux containers（通常使用 WSL2），启动后执行 `docker version` 和 `docker compose version`。在项目根目录确认已有 `.env`，不要覆盖原配置。新项目才从 `.env.example` 复制，填写两个模型密钥。

如果原 Python 服务已经占用 8000，在 `.env` 加入：

```dotenv
PAPERAGENT_BIND_ADDRESS=0.0.0.0
PAPERAGENT_PORT=8001
```

`main.py` 的默认本机启动地址不受上述 Compose 发布端口控制；容器内部仍使用 8000，宿主端口可以是 8001。

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 paperagent
```

`config --quiet` 只验证配置，避免把展开后的密钥显示到终端。构建阶段需要能访问基础镜像和 Python 包源；首次重排需要获取固定模型权重。

验证本机：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Start-Process http://127.0.0.1:8001
```

上传一份短 PDF，确认任务由 queued/running 变 completed，再创建会话提问、核对引用。不要只用 `/health` 成功来证明模型密钥、全文权限和解析质量可用。

## 4. 分享链接与网络检查

1. 执行 `ipconfig`，选择校园网正在使用的有线或无线适配器 IPv4，避免选 Docker/WSL 虚拟网卡或回环地址。
2. 用同学电脑访问 `http://该IPv4:8001`；浏览器必须能到达这台主机，校园无线可能隔离客户端。
3. 如果本机可以访问、同学不能访问，先确认同一可达网络与端口映射，再检查 Windows 入站防火墙。
4. 如需增加防火墙规则，在管理员 PowerShell 按实际端口执行，并把访问范围限制为本地子网：

```powershell
New-NetFirewallRule -DisplayName 'PaperAgent campus 8001' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8001 -RemoteAddress LocalSubnet
```

这条规则只是操作示例，本次没有改动你的防火墙。若校园网跨子网，LocalSubnet 未必覆盖同学地址，需要按实际学校网络调整。主机休眠、断网或退出 Docker 后链接不可用，校园 IP 改变后需更新链接。

## 5. 数据目录、模型和备份

| 容器路径 | 内容 |
| --- | --- |
| `/app` | 程序与文档，普通用户无写权限 |
| `/data/service` | 原 PDF、业务 SQLite、Checkpointer SQLite |
| `/data/ingestion` | 页图、OCR/VLM JSON、Markdown、解析缓存与任务产物 |
| `/data/chroma` | Chroma 向量与元数据 |
| `/data/models/mmarco-minilm` | 固定版本重排权重 |

以上运行数据全部在 `paperagent-data` 命名卷。`docker compose stop`/`down` 保留卷，`docker compose down -v` 会删除运行数据。不要为“重启应用”使用 `-v`。

宿主的历史数据库没有自动导入卷。Windows 业务库记录了绝对原件路径，不能直接复制后就假设 Linux 可用；当前推荐在容器重新上传论文。已登记在原 Windows 服务的论文继续留在原路径，没有迁移或覆盖。

重排权重可以在容器首次问答时下载，也可以先复制已准备好的完整缓存：

```powershell
docker compose cp ./pdfprocess/data/models/mmarco-minilm/. paperagent:/data/models/mmarco-minilm
```

复制后检查写权限和文件完整性。OCR 权重来自固定 RapidOCR 安装包。备份时先等待任务/问答完成，再停止容器，对整个命名卷做一致性备份；只复制正在写入的单个 SQLite 主文件可能漏掉 WAL。

## 6. 校园论文权限

论文请求从容器发起，可能通过 Docker Desktop/WSL2 NAT 出网。即使宿主浏览器已通过验证码或机构识别，容器 HTTP 会话仍可能没有权限。

- 万方元数据搜索已经实测可用，全文请求在本机后端返回 401，机构身份检查未识别到机构。
- 知网通过网页索引发现官网详情页，直接 HTTP 请求曾要求安全验证，真实全文下载尚未成功验证。
- `PAPERAGENT_PAPERS_USE_PROXY=false` 可让校园论文 HTTP 客户端绕过环境代理，适合学校出口必须直连的情况。知网的 Bing 索引定位仍沿用网页搜索模块自身的环境代理设置。
- `.env` 中如有指向 `127.0.0.1` 的代理，容器的这个地址指向容器自己，不能直接当成 Windows 宿主代理。
- 仅有 CAJ 时会报错；不会把错误网页伪装成 PDF，也不会在失败后虚报入库。

验收应分开记录“检索成功”“PDF 获取成功”“解析入库成功”，并保留 HTTP 状态和任务 ID。当前可以手动从官网取得已授权的 PDF 后使用批量上传。

## 7. 如何量化部署效果

使用相同电脑、论文集、问题集、模型、候选数和压缩预算，分别测量直接 Python 与 Docker。先完成一次预热，再重复至少 3 轮；冷启动单独记录。

| 指标 | 手动采集方式 |
| --- | --- |
| 首次准备与冷启动耗时 | PowerShell `Measure-Command` 测构建和启动；记录到 `/health`、首次可用问答的时间 |
| 空闲/解析/问答的内存和 CPU | Docker `docker stats --no-stream`；本机任务管理器，同时记录 Docker Desktop/WSL 总开销 |
| 上传确认时间、排队时间、完整入库耗时 | 使用 `upload_service.benchmark`，对照任务阶段事件；同一 PDF 的冷缓存/热缓存分开 |
| 问答 P50/P95、失败率、模型用量 | 使用 `upload_service.benchmark_chat`，保存每轮 metrics；固定并发与问题顺序 |
| 检索质量 | 固定知识库与证据标注，计算 Recall@K/MRR；Docker 本身不应被当作检索算法收益 |
| 重启恢复 | 先完成会话和入库，执行 `docker compose restart`，核对原文件、任务、答案和追问历史 |
| 校园访问成功率 | 两台或多台同学设备逐项验证首页、上传、问答、PDF 引用，分别记录网络与服务失败 |

先从 1～3 名同学同时使用开始测，不对未经压测的并发量作承诺。共享空间没有用户级配额，同学的调用产生主机 API 账单。
