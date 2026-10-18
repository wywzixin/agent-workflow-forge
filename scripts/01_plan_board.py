"""01 主 Agent 生成公共任务板。

这个脚本故意不抽公共模块。课后阅读时，学生从上往下看一遍，就能看到：
读取业务目标 -> 主 Agent 生成任务建议 -> 代码收窄成可执行任务板 -> 保存结果。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agently import Agently
from dotenv import find_dotenv, load_dotenv

LESSON_DIR = Path(__file__).resolve().parents[1]
GOAL_PATH = LESSON_DIR / "materials" / "product_goal.txt"
RUN_DIR = LESSON_DIR / ".demo_runs" / "multi_agent"
BOARD_PATH = RUN_DIR / "01_task_board.json"
ALLOWED_AGENTS = {"codex_research", "codex_plan"}
MIN_TASK_CARDS = 2
MAX_TASK_CARDS = 6
TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


@dataclass(frozen=True)
class ProjectBrief:
    """主 Agent 收到的原始业务输入。"""

    text: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "text": self.text}


@dataclass(frozen=True)
class TaskCard:
    """公共任务板上的一张任务卡。"""

    task_id: str
    title: str
    assigned_agent: str
    goal: str
    context: str
    depends_on: tuple[str, ...] = ()
    status: str = "todo"
    result: str = ""
    session_id: str = ""
    attempts: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["depends_on"] = list(self.depends_on)
        return payload


def configure_model() -> None:
    """配置主 Agent 使用的模型请求器。"""

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("需要 DEEPSEEK_API_KEY（放在项目根目录 .env 或 shell 导出）。")

    Agently.set_settings(
        "OpenAICompatible",
        {
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "model": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
            "model_type": "chat",
            "auth": api_key,
            "request_options": {"temperature": 0},
        },
    )


def load_project_brief() -> ProjectBrief:
    return ProjectBrief(
        text=GOAL_PATH.read_text(encoding="utf-8").strip(),
        source=str(GOAL_PATH),
    )


def compact_text(text: str, limit: int = 1600) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...(已由主 Agent 截断)"


def build_board_request(project: ProjectBrief) -> Any:
    """让主 Agent 真实决定任务数量、任务内容、执行者和依赖关系。"""

    task_schema = {
        "tasks": [
            {
                "task_id": (str, "稳定 id，由主 Agent 自行命名；使用小写英文、数字和下划线。"),
                "title": (str, "任务标题。"),
                "assigned_agent": (str, "只能使用 codex_research、codex_plan 之一。"),
                "goal": (str, "这个子 Agent 要完成的局部目标，控制在 160 个中文字符以内。"),
                "context": (str, "只写允许给这个子任务看的公共背景，控制在 320 个中文字符以内。"),
                "depends_on": [
                    (str, "上游 task_id；没有依赖时返回空列表。")
                ],
            }
        ]
    }

    return (
        Agently.create_agent()
        .info({"业务目标": project.text})
        .input(
            "请作为主 Agent，根据业务目标自行拆解一个可执行任务板。"
            f"任务数量由你决定，但必须在 {MIN_TASK_CARDS}-{MAX_TASK_CARDS} 张之间。"
            "每张卡都要有清晰局部目标、可交给一个子 Agent 独立推进。"
            "请自行决定每张卡分配给 codex_research 还是 codex_plan，并设计 depends_on。"
            "依赖关系必须是 DAG，不能循环，不能引用不存在的 task_id，且至少有一张卡依赖上游结果。"
            "task_id 请使用 lower_snake_case，例如 research_knowns、plan_route、review_risks；不要使用中文或连字符。"
            "不要把任务拆成实时查询或本地文件操作；需要外部核验的事实写入目标或上下文里的待核验口径。"
        )
        .output(task_schema)
    )


def add_execution_boundary(text: str) -> str:
    boundary = (
        "执行边界:子 Agent 只基于当前输入和显式上游结果分析; "
        "需要外部核验的内容列为待核验,不要声称已经实时查证。"
    )
    return f"{text}\n{boundary}" if boundary not in text else text


def require_text(item: dict[str, object], key: str, task_label: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task_label}.{key} 必须是非空字符串")
    return value.strip()


def normalize_depends_on(item: dict[str, object], task_label: str) -> tuple[str, ...]:
    raw_deps = item.get("depends_on", [])
    if raw_deps is None:
        return ()
    if not isinstance(raw_deps, list):
        raise ValueError(f"{task_label}.depends_on 必须是列表")
    return tuple(str(dep).strip() for dep in raw_deps if str(dep).strip())


def ensure_board_is_dag(cards: list[TaskCard]) -> None:
    by_id = {card.task_id: card for card in cards}
    resolved: set[str] = set()
    pending = set(by_id)
    while pending:
        ready = [
            task_id
            for task_id in pending
            if all(dep in resolved for dep in by_id[task_id].depends_on)
        ]
        if not ready:
            raise ValueError("任务板依赖存在循环，或某些任务永远无法进入 ready 状态")
        for task_id in ready:
            resolved.add(task_id)
            pending.remove(task_id)


def validate_model_board(raw_board: object) -> list[TaskCard]:
    """只做真实应用里的护栏校验；不补预设任务，也不固定任务数量。"""

    raw_items = raw_board.get("tasks", []) if isinstance(raw_board, dict) else []
    if not isinstance(raw_items, list):
        raise ValueError("主 Agent 必须返回 tasks 列表")
    if not (MIN_TASK_CARDS <= len(raw_items) <= MAX_TASK_CARDS):
        raise ValueError(f"任务数量必须在 {MIN_TASK_CARDS}-{MAX_TASK_CARDS} 张之间，实际为 {len(raw_items)}")

    cards: list[TaskCard] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{index}] 必须是 dict")
        task_label = f"tasks[{index}]"
        task_id = require_text(item, "task_id", task_label)
        if not TASK_ID_PATTERN.match(task_id):
            raise ValueError(f"{task_label}.task_id 必须是 lower_snake_case，实际为 {task_id!r}")
        if task_id in seen_ids:
            raise ValueError(f"重复 task_id:{task_id}")
        seen_ids.add(task_id)

        assigned_agent = require_text(item, "assigned_agent", task_label)
        if assigned_agent not in ALLOWED_AGENTS:
            raise ValueError(f"{task_id}.assigned_agent 只能是 {sorted(ALLOWED_AGENTS)}，实际为 {assigned_agent}")

        depends_on = normalize_depends_on(item, task_label)
        cards.append(
            TaskCard(
                task_id=task_id,
                title=compact_text(require_text(item, "title", task_label), limit=120),
                assigned_agent=assigned_agent,
                goal=compact_text(require_text(item, "goal", task_label), limit=360),
                context=add_execution_boundary(compact_text(require_text(item, "context", task_label), limit=900)),
                depends_on=depends_on,
            )
        )

    known_ids = {card.task_id for card in cards}
    if not any(card.depends_on for card in cards):
        raise ValueError("任务板至少需要一条依赖边，否则无法体现多 Agent 协作")
    for card in cards:
        for dep in card.depends_on:
            if dep == card.task_id:
                raise ValueError(f"{card.task_id} 不能依赖自己")
            if dep not in known_ids:
                raise ValueError(f"{card.task_id} 依赖不存在的任务:{dep}")
    ensure_board_is_dag(cards)
    return cards


def render_board(cards: list[TaskCard]) -> str:
    lines = ["task_id      | status  | agent          | deps                      | session"]
    for card in cards:
        deps = ",".join(card.depends_on) if card.depends_on else "-"
        session = card.session_id[:8] if card.session_id else "-"
        lines.append(
            f"{card.task_id:12} | {card.status:7} | {card.assigned_agent:14} | "
            f"{deps:25} | {session}"
        )
    return "\n".join(lines)


def save_json(path: Path, payload: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    configure_model()

    project = load_project_brief()
    request = build_board_request(project)
    prompt_text = request.get_prompt_text()
    raw_board = await request.async_start()
    cards = validate_model_board(raw_board)

    print("===== 主 Agent prompt =====")
    print(prompt_text)
    print("\n===== 模型建议任务板 =====")
    print(json.dumps(raw_board, ensure_ascii=False, indent=2))
    print("\n===== 校验后的动态公共任务板 =====")
    print(render_board(cards))

    save_json(
        BOARD_PATH,
        {
            "project_brief": project.to_dict(),
            "raw_board": raw_board,
            "tasks": [card.to_dict() for card in cards],
        },
    )
    print(f"\nsaved: {BOARD_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
