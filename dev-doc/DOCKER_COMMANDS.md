# Docker 常用命令

## Alembic 数据库迁移

```bash
# 容器运行中时（推荐）
docker compose exec backend /app/.venv/bin/alembic upgrade head

# 新建临时容器执行
docker compose run --rm backend /app/.venv/bin/alembic upgrade head
```

其他命令：
```bash
docker exec maverick-mcp-backend-1 /app/.venv/bin/alembic current   # 查看当前版本
docker exec maverick-mcp-backend-1 /app/.venv/bin/alembic history   # 查看历史
docker exec maverick-mcp-backend-1 /app/.venv/bin/alembic downgrade -1  # 回滚
```

> **macOS 提示**：若 `docker compose` 报 `stat .env: operation not permitted`，改用 `docker exec <容器名>` 替代。

## 其他常用

```bash
docker exec -it maverick-mcp-backend-1 /bin/bash  # 进入容器 shell
docker logs maverick-mcp-backend-1 -f              # 查看日志
docker ps --filter "name=maverick"                 # 查看容器状态
```
