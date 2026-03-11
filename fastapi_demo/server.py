"""
FastAPI Server Demo — Python 核心语法速览

这个文件既是一个可运行的 FastAPI 服务，也是一份 Python 关键语法参考。
每个 API 端点演示一个 Python 核心概念。

启动方式: uvicorn server:app --reload --port 8000
API 文档: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

app = FastAPI(title="Python 语法学习 API", version="1.0.0")


# ============================================================
# 1. 变量类型 & 类型注解 (Type Hints)
# ============================================================
# Python 是动态类型语言，但支持类型注解提高可读性

@app.get("/syntax/types")
def demo_types():
    name: str = "Alice"               # 字符串
    age: int = 25                     # 整数
    height: float = 1.68              # 浮点数
    is_student: bool = True           # 布尔
    skills: list[str] = ["Python", "FastAPI"]  # 列表（有序、可变）
    scores: dict[str, int] = {"math": 95, "english": 88}  # 字典（键值对）
    unique_ids: set[int] = {1, 2, 3}  # 集合（无序、不重复）
    coordinates: tuple[float, float] = (39.9, 116.4)  # 元组（有序、不可变）
    nickname: Optional[str] = None    # 可选类型，值可以是 str 或 None

    return {
        "str": name,
        "int": age,
        "float": height,
        "bool": is_student,
        "list": skills,
        "dict": scores,
        "set": list(unique_ids),
        "tuple": list(coordinates),
        "optional": nickname,
    }


# ============================================================
# 2. 字符串操作 (String Operations)
# ============================================================

@app.get("/syntax/strings")
def demo_strings():
    s = "Hello, Python!"

    return {
        "original": s,
        "upper": s.upper(),              # 全部大写
        "lower": s.lower(),              # 全部小写
        "split": s.split(", "),          # 按分隔符拆分为列表
        "replace": s.replace("Python", "FastAPI"),  # 替换子串
        "strip": "  spaces  ".strip(),   # 去除两端空白
        "startswith": s.startswith("Hello"),  # 判断前缀
        "find": s.find("Python"),        # 查找子串位置，找不到返回 -1
        "f-string": f"{s} 字符串长度是 {len(s)}",  # f-string 格式化
        "slice": s[0:5],                 # 切片 [start:end)
        "reverse": s[::-1],              # 反转字符串
    }


# ============================================================
# 3. 列表推导式 & 生成器 (Comprehensions & Generators)
# ============================================================

@app.get("/syntax/comprehensions")
def demo_comprehensions():
    nums = list(range(1, 11))  # [1, 2, ..., 10]

    squares = [x ** 2 for x in nums]                    # 列表推导式
    evens = [x for x in nums if x % 2 == 0]             # 带条件的列表推导式
    pairs = {x: x ** 2 for x in nums[:5]}               # 字典推导式
    unique = {x % 3 for x in nums}                       # 集合推导式
    gen_sum = sum(x ** 2 for x in nums)                  # 生成器表达式（惰性求值，省内存）

    return {
        "range": nums,
        "squares": squares,
        "evens": evens,
        "dict_comp": pairs,
        "set_comp": list(unique),
        "generator_sum": gen_sum,
    }


# ============================================================
# 4. 函数 & Lambda & 高阶函数
# ============================================================

def greet(name: str, greeting: str = "Hello") -> str:
    """带默认参数的函数"""
    return f"{greeting}, {name}!"

def apply(func, value):
    """高阶函数：接收函数作为参数"""
    return func(value)

@app.get("/syntax/functions")
def demo_functions():
    double = lambda x: x * 2  # lambda 匿名函数

    nums = [3, 1, 4, 1, 5, 9]
    return {
        "greet_default": greet("Alice"),
        "greet_custom": greet("Bob", greeting="Hi"),
        "lambda_result": double(21),
        "apply_higher_order": apply(lambda x: x ** 3, 5),
        "map": list(map(str, nums)),            # map: 对每个元素应用函数
        "filter": list(filter(lambda x: x > 3, nums)),  # filter: 过滤元素
        "sorted": sorted(nums),                 # 升序排序（返回新列表）
        "sorted_desc": sorted(nums, reverse=True),  # 降序排序
    }


# ============================================================
# 5. 类 & 继承 & Pydantic Model
# ============================================================

# --- Enum 枚举 ---
class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


# --- Pydantic Model（FastAPI 的核心：自动验证 + 序列化）---
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, examples=["Alice"])
    age: int = Field(..., ge=0, le=150, examples=[25])
    email: Optional[str] = Field(None, examples=["alice@example.com"])
    role: Role = Field(default=Role.USER)


class UserResponse(BaseModel):
    id: int
    name: str
    age: int
    email: Optional[str]
    role: Role
    is_adult: bool


# --- 普通 Python 类 & 继承 ---
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return f"{self.name} makes a sound"

class Dog(Animal):
    """继承 Animal，重写 speak 方法"""
    def speak(self) -> str:
        return f"{self.name} says: Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says: Meow!"


@app.get("/syntax/classes")
def demo_classes():
    dog = Dog("Buddy")
    cat = Cat("Kitty")

    return {
        "dog": dog.speak(),
        "cat": cat.speak(),
        "isinstance_check": isinstance(dog, Animal),  # 类型检查
        "roles": [r.value for r in Role],              # 枚举遍历
    }


# ============================================================
# 6. CRUD 模拟（内存数据库）— POST / GET / PUT / DELETE
# ============================================================

fake_db: dict[int, UserResponse] = {}
next_id = 1


@app.post("/users", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate):
    """POST — 创建用户（演示 Pydantic 自动校验）"""
    global next_id
    new_user = UserResponse(
        id=next_id,
        name=user.name,
        age=user.age,
        email=user.email,
        role=user.role,
        is_adult=user.age >= 18,
    )
    fake_db[next_id] = new_user
    next_id += 1
    return new_user


@app.get("/users", response_model=list[UserResponse])
def list_users(
    role: Optional[Role] = Query(None, description="按角色过滤"),
    min_age: int = Query(0, ge=0, description="最小年龄"),
):
    """GET — 查询用户列表（演示 Query 参数）"""
    users = list(fake_db.values())
    if role:
        users = [u for u in users if u.role == role]
    users = [u for u in users if u.age >= min_age]
    return users


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int = Path(..., ge=1, description="用户ID")):
    """GET — 获取单个用户（演示 Path 参数 + 异常处理）"""
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return fake_db[user_id]


@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserCreate):
    """PUT — 更新用户"""
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    updated = UserResponse(
        id=user_id,
        name=user.name,
        age=user.age,
        email=user.email,
        role=user.role,
        is_adult=user.age >= 18,
    )
    fake_db[user_id] = updated
    return updated


@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    """DELETE — 删除用户"""
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    del fake_db[user_id]
    return {"message": f"User {user_id} deleted"}


# ============================================================
# 7. 异常处理 (try / except / finally)
# ============================================================

@app.get("/syntax/exceptions")
def demo_exceptions(
    a: int = Query(10, description="被除数"),
    b: int = Query(0, description="除数"),
):
    """演示 try/except/finally 和自定义异常"""
    results = {}
    try:
        results["division"] = a / b
    except ZeroDivisionError:
        results["division"] = "Error: 除数不能为0"
    except TypeError as e:
        results["division"] = f"Error: 类型错误 - {e}"
    finally:
        results["finally"] = "finally 块总是会执行"

    return results


# ============================================================
# 8. 装饰器 (Decorators)
# ============================================================

import time
from functools import wraps

def timer(func):
    """装饰器：计算函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return {"result": result, "elapsed_ms": round(elapsed * 1000, 2)}
    return wrapper

@timer
def slow_calculation(n: int) -> int:
    """模拟耗时计算"""
    return sum(i * i for i in range(n))

@app.get("/syntax/decorators")
def demo_decorators(n: int = Query(100000, description="计算范围")):
    return slow_calculation(n)


# ============================================================
# 9. 依赖注入 (Dependency Injection) — FastAPI 特色
# ============================================================

def get_current_user(token: str = Query("guest-token", description="模拟 token")):
    """依赖函数：模拟用户认证"""
    users = {
        "admin-token": {"username": "admin", "role": "admin"},
        "guest-token": {"username": "guest", "role": "guest"},
    }
    if token not in users:
        raise HTTPException(status_code=401, detail="Invalid token")
    return users[token]

@app.get("/syntax/dependency")
def demo_dependency(current_user: dict = Depends(get_current_user)):
    """演示 FastAPI 的 Depends 依赖注入"""
    return {
        "message": f"Welcome, {current_user['username']}!",
        "user": current_user,
    }


# ============================================================
# 10. 异步 (async/await) — FastAPI 原生支持
# ============================================================

import asyncio

@app.get("/syntax/async")
async def demo_async():
    """演示 async/await 异步编程"""

    async def fetch_data(name: str, delay: float) -> dict:
        await asyncio.sleep(delay)  # 模拟 IO 操作（不阻塞其他请求）
        return {"source": name, "delay": delay}

    start = time.perf_counter()
    # asyncio.gather 并发执行多个异步任务
    results = await asyncio.gather(
        fetch_data("database", 0.3),
        fetch_data("cache", 0.1),
        fetch_data("api", 0.2),
    )
    elapsed = time.perf_counter() - start

    return {
        "results": results,
        "total_time_ms": round(elapsed * 1000, 2),
        "note": "三个任务并发执行，总时间 ≈ 最长的那个（~300ms），而非串行的 600ms",
    }


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
