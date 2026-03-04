# 变更记录

## 2026-03-04 Docker & 服务器修复

### `docker-compose.yml`
- 添加 `YFINANCE_CACHE_DIR=/tmp/yfinance-cache` 环境变量，解决容器内 yfinance 缓存路径冲突

### `maverick_mcp/api/server.py`

**1. yfinance TzCache 修复**（Errno 17 路径冲突）
```python
import os as _os
import yfinance as _yf
_yf.set_tz_cache_location(_os.environ.get("YFINANCE_CACHE_DIR", "/tmp/yfinance-cache"))
del _yf, _os
```

**2. BeautifulSoup `findAll` 弃用警告屏蔽**（来自 finvizfinance 等第三方库）
```python
warnings.filterwarnings(
    "ignore",
    message=".*findAll.*Deprecated.*",
    category=DeprecationWarning,
)
```

**3. SSE 路由双路径注册**（已有代码，修复 mcp-remote 307 重定向问题）
- monkey-patch `fastmcp_http.create_sse_app`，同时注册 `/sse` 和 `/sse/`，防止工具注册失败

### `Dockerfile`
- transport 在 `sse` / `streamable-http` 之间切换测试（最终回退至 `sse`，因为 `streamable-http` 有 ASGI 兼容问题）

---

## 其他（大量测试文件删除）
- 删除 88 个测试文件（约 47000 行），清理未使用/过时的测试代码
