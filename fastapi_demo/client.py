"""
FastAPI Client Demo — 调用 server.py 的所有端点

使用 httpx 库（支持同步和异步请求）
先启动 server: python server.py
再运行 client: python client.py

演示的 Python 语法:
- with 语句（上下文管理器）
- json 处理
- 条件表达式
- 字符串格式化
- 函数组织
"""

import httpx
import json

BASE_URL = "http://localhost:8000"


def pretty(title: str, data: dict | list):
    """格式化打印 API 响应"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_syntax_endpoints(client: httpx.Client):
    """测试所有语法演示端点"""

    # --- 1. 类型系统 ---
    resp = client.get("/syntax/types")
    pretty("1. 变量类型 & 类型注解", resp.json())

    # --- 2. 字符串操作 ---
    resp = client.get("/syntax/strings")
    pretty("2. 字符串操作", resp.json())

    # --- 3. 推导式 ---
    resp = client.get("/syntax/comprehensions")
    pretty("3. 列表推导式 & 生成器", resp.json())

    # --- 4. 函数 ---
    resp = client.get("/syntax/functions")
    pretty("4. 函数 & Lambda & 高阶函数", resp.json())

    # --- 5. 类 & 继承 ---
    resp = client.get("/syntax/classes")
    pretty("5. 类 & 继承 & 枚举", resp.json())

    # --- 7. 异常处理 ---
    resp = client.get("/syntax/exceptions", params={"a": 10, "b": 0})
    pretty("7. 异常处理 (10 / 0)", resp.json())

    resp = client.get("/syntax/exceptions", params={"a": 10, "b": 3})
    pretty("7. 异常处理 (10 / 3)", resp.json())

    # --- 8. 装饰器 ---
    resp = client.get("/syntax/decorators", params={"n": 500000})
    pretty("8. 装饰器 (计时器)", resp.json())

    # --- 9. 依赖注入 ---
    resp = client.get("/syntax/dependency", params={"token": "admin-token"})
    pretty("9. 依赖注入 (admin)", resp.json())

    resp = client.get("/syntax/dependency", params={"token": "bad-token"})
    pretty("9. 依赖注入 (invalid token)", {
        "status_code": resp.status_code,
        "detail": resp.json().get("detail"),
    })

    # --- 10. 异步 ---
    resp = client.get("/syntax/async")
    pretty("10. 异步 async/await", resp.json())


def test_crud(client: httpx.Client):
    """测试 CRUD 端点"""

    print("\n" + "#" * 60)
    print("  CRUD 操作演示")
    print("#" * 60)

    # POST — 创建用户
    users_to_create = [
        {"name": "Alice", "age": 28, "email": "alice@test.com", "role": "admin"},
        {"name": "Bob", "age": 17, "email": "bob@test.com", "role": "user"},
        {"name": "Charlie", "age": 35, "role": "user"},
    ]

    created_ids = []
    for user_data in users_to_create:
        resp = client.post("/users", json=user_data)
        user = resp.json()
        created_ids.append(user["id"])
        pretty(f"POST /users — 创建: {user_data['name']}", user)

    # GET — 查询所有用户
    resp = client.get("/users")
    pretty("GET /users — 所有用户", resp.json())

    # GET — 按角色过滤
    resp = client.get("/users", params={"role": "admin"})
    pretty("GET /users?role=admin — 过滤管理员", resp.json())

    # GET — 按年龄过滤
    resp = client.get("/users", params={"min_age": 18})
    pretty("GET /users?min_age=18 — 成年用户", resp.json())

    # GET — 获取单个用户
    resp = client.get(f"/users/{created_ids[0]}")
    pretty(f"GET /users/{created_ids[0]} — 获取 Alice", resp.json())

    # GET — 不存在的用户（404）
    resp = client.get("/users/999")
    pretty("GET /users/999 — 不存在的用户", {
        "status_code": resp.status_code,
        "detail": resp.json().get("detail"),
    })

    # PUT — 更新用户
    resp = client.put(f"/users/{created_ids[1]}", json={
        "name": "Bob Updated",
        "age": 18,
        "email": "bob.new@test.com",
        "role": "admin",
    })
    pretty(f"PUT /users/{created_ids[1]} — 更新 Bob", resp.json())

    # DELETE — 删除用户
    resp = client.delete(f"/users/{created_ids[2]}")
    pretty(f"DELETE /users/{created_ids[2]} — 删除 Charlie", resp.json())

    # GET — 验证删除后的列表
    resp = client.get("/users")
    pretty("GET /users — 删除后的用户列表", resp.json())

    # POST — Pydantic 校验失败（name 为空）
    resp = client.post("/users", json={"name": "", "age": 200})
    pretty("POST /users — 校验失败 (name空, age>150)", {
        "status_code": resp.status_code,
        "errors": resp.json().get("detail", []),
    })


def main():
    print("FastAPI Client Demo")
    print(f"目标服务器: {BASE_URL}")
    print("请确保 server.py 已启动\n")

    # with 语句：自动管理资源（请求结束后自动关闭连接）
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 先检查服务器是否可用
        try:
            client.get("/syntax/types")
        except httpx.ConnectError:
            print("错误: 无法连接到服务器！")
            print("请先运行: python server.py")
            return

        test_syntax_endpoints(client)
        test_crud(client)

    print("\n" + "=" * 60)
    print("  全部测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
