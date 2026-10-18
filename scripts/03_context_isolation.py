"""03 把公共任务板裁剪成子 Agent 私有任务包。

这个脚本展示上下文隔离，不追求复用：
读取真实任务板 -> 构造一张 ready 卡片的私有任务包 -> 调用 Codex worker
-> 把公开结果写回任务板 -> 构造下游 ready 卡片的私有任务包，观察它只拿到显式上游结果。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from acp import PROTOCOL_VERSION, Client, spawn_agent_process, text_block
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    Implementation,
    PermissionOption,
    ReadTextFileResponse,
    RequestPermissionResponse,
    ToolCallUpdate,
)

LESSON_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = LESSON_DIR / ".demo_runs" / "multi_agent"
BOARD_PATH = RUN_DIR / "01_task_board.json"
CONTEXT_PATH = RUN_DIR / "03_context_isolation.json"

CODEX_BIN_DIR = Path("/Applications/Codex.app/Contents/Resources")
ACP_ENV = os.environ.copy()
if CODEX_BIN_DIR.is_dir():
    ACP_ENV["PATH"] = f"{CODEX_BIN_DIR}:{ACP_ENV.get('PATH', '')}"
NPX_COMMAND = os.getenv("ACP_NPX_COMMAND", "npx")


@dataclass(frozen=True)
class TaskCard:
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


@dataclass(frozen=True)
class WorkerTaskInput:
    task_id: str
    title: str
    assigned_agent: str
    goal: str
    context: str
    upstream_results: dict[str, str]
    acceptance: tuple[str, ...]
    forbidden: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "assigned_agent": self.assigned_agent,
            "goal": self.goal,
            "context": self.context,
            "upstream_results": self.upstream_results,
            "acceptance": list(self.acceptance),
            "forbidden": list(self.forbidden),
        }


@dataclass(frozen=True)
class AgentSpec:
    name: str
    command: str
    args: tuple[str, ...]

    def launcher_available(self) -> bool:
        if Path(self.command).is_file():
            return True
        return shutil.which(self.command, path=ACP_ENV.get("PATH")) is not None


@dataclass(frozen=True)
class WorkerResult:
    agent_name: str
    agent_info: str
    session_id: str
    stop_reason: str
    update_kinds: tuple[str, ...]
    text: str
    permission_decisions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


AGENTS = {
    "codex_research": AgentSpec("codex_research", NPX_COMMAND, ("-y", "@agentclientprotocol/codex-acp@1.1.0")),
    "codex_plan": AgentSpec("codex_plan", NPX_COMMAND, ("-y", "@agentclientprotocol/codex-acp@1.1.0")),
}


def card_from_dict(raw: dict[str, object]) -> TaskCard:
    deps = raw.get("depends_on", [])
    attempts = raw.get("attempts", 0)
    return TaskCard(
        task_id=str(raw["task_id"]),
        title=str(raw["title"]),
        assigned_agent=str(raw["assigned_agent"]),
        goal=str(raw["goal"]),
        context=str(raw["context"]),
        depends_on=tuple(str(item) for item in deps) if isinstance(deps, list) else (),
        status=str(raw.get("status", "todo")),
        result=str(raw.get("result", "")),
        session_id=str(raw.get("session_id", "")),
        attempts=attempts if isinstance(attempts, int) else 0,
    )


def load_cards() -> list[TaskCard]:
    if not BOARD_PATH.exists():
        raise RuntimeError(
            f"没有找到真实任务板:{BOARD_PATH}\n"
            "请先运行 scripts/01_plan_board.py，让主 Agent 真实拆解任务。"
        )
    payload = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    items = payload.get("tasks", []) if isinstance(payload, dict) else []
    cards = [card_from_dict(item) for item in items if isinstance(item, dict)]
    if not cards:
        raise RuntimeError("真实任务板里没有可用 TaskCard")
    return cards


def save_json(path: Path, payload: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def compact_text(text: str, limit: int = 1200) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...(已截断，完整结果保存在运行产物中)"


def next_ready_card(cards: list[TaskCard]) -> TaskCard | None:
    done_ids = {card.task_id for card in cards if card.status == "done"}
    ready = [
        card
        for card in cards
        if card.status == "todo" and set(card.depends_on) <= done_ids
    ]
    return ready[0] if ready else None


def downstream_ready_card(cards: list[TaskCard]) -> TaskCard | None:
    """找一张已经可以执行、且确实带有上游结果的下游卡片。"""

    done_ids = {card.task_id for card in cards if card.status == "done"}
    for card in cards:
        if card.status == "todo" and card.depends_on and set(card.depends_on) <= done_ids:
            return card
    return None


def build_worker_input(card: TaskCard, done_cards: dict[str, TaskCard]) -> WorkerTaskInput:
    missing = [task_id for task_id in card.depends_on if task_id not in done_cards]
    if missing:
        raise RuntimeError(f"{card.task_id} 依赖未完成:{', '.join(missing)}")

    return WorkerTaskInput(
        task_id=card.task_id,
        title=card.title,
        assigned_agent=card.assigned_agent,
        goal=card.goal,
        context=card.context,
        upstream_results={
            task_id: compact_text(done_cards[task_id].result)
            for task_id in card.depends_on
        },
        acceptance=(
            "只基于任务包和显式上游结果回答。",
            "把需要外部核验的信息列为缺口。",
            "输出可交给主 Agent 回收的摘要。",
        ),
        forbidden=(
            "不要声称读取了完整主会话。",
            "不要读取或修改本地文件。",
            "不要运行命令。",
        ),
    )


def format_worker_prompt(worker_input: WorkerTaskInput) -> str:
    upstream = json.dumps(worker_input.upstream_results, ensure_ascii=False, indent=2)
    return (
        f"任务卡片:{worker_input.task_id} - {worker_input.title}\n\n"
        f"目标:\n{worker_input.goal}\n\n"
        f"背景:\n{worker_input.context}\n\n"
        f"上游结果:\n{upstream}\n\n"
        "输出要求:\n"
        "1. 本任务结论\n"
        "2. 信息缺口或风险\n"
        "3. 可交给下游的摘要\n\n"
        "禁止事项:\n"
        + "\n".join(f"- {item}" for item in worker_input.forbidden)
    )


def resolve_agent(name: str) -> AgentSpec:
    spec = AGENTS[name]
    if not spec.launcher_available():
        raise RuntimeError(f"{name} 的启动器不可见:{spec.command}")
    return spec


class CourseAcpClient:
    def __init__(self) -> None:
        self.text_chunks: list[str] = []
        self.update_kinds: list[str] = []
        self.permission_decisions: list[str] = []

    def begin_turn(self) -> None:
        self.text_chunks = []
        self.update_kinds = []
        self.permission_decisions = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        update_kind = str(getattr(update, "session_update", ""))
        self.update_kinds.append(update_kind)
        if update_kind != "agent_message_chunk":
            return
        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        if getattr(content, "type", None) == "text" and isinstance(text, str):
            self.text_chunks.append(text)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        reject = next((item for item in options if item.kind.startswith("reject")), None)
        if reject is None:
            self.permission_decisions.append("cancelled")
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        self.permission_decisions.append(reject.option_id)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=reject.option_id)
        )

    async def write_text_file(self, **kwargs: Any) -> None:
        raise RuntimeError("课程示例没有声明文件写入能力")

    async def read_text_file(self, **kwargs: Any) -> ReadTextFileResponse:
        raise RuntimeError("课程示例没有声明文件读取能力")

    @property
    def text(self) -> str:
        return "".join(self.text_chunks).strip()


class AcpWorkerSession:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.client = CourseAcpClient()
        self.agent_info = "(未初始化)"
        self.session_id = ""
        self._process_cm: Any | None = None
        self._connection: Any | None = None

    async def __aenter__(self) -> "AcpWorkerSession":
        self._process_cm = spawn_agent_process(
            cast(Client, self.client),
            self.spec.command,
            *self.spec.args,
            cwd=LESSON_DIR,
            env=ACP_ENV,
        )
        connection, _process = await self._process_cm.__aenter__()
        self._connection = connection
        init = await connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="course-master-agent", title="课程主 Agent 调度器", version="0.2.0"),
        )
        self.agent_info = init.agent_info.name if init.agent_info is not None else "(未提供 agentInfo)"
        session = await connection.new_session(cwd=str(LESSON_DIR), mcp_servers=[])
        self.session_id = session.session_id
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._process_cm is not None:
            await self._process_cm.__aexit__(exc_type, exc, tb)

    async def prompt(self, prompt_text: str) -> WorkerResult:
        if self._connection is None or not self.session_id:
            raise RuntimeError("ACP session 尚未初始化")
        self.client.begin_turn()
        response = await self._connection.prompt(
            prompt=[text_block(prompt_text)],
            session_id=self.session_id,
        )
        return WorkerResult(
            agent_name=self.spec.name,
            agent_info=self.agent_info,
            session_id=self.session_id,
            stop_reason=str(response.stop_reason),
            update_kinds=tuple(self.client.update_kinds),
            text=self.client.text,
            permission_decisions=tuple(self.client.permission_decisions),
        )


async def main() -> None:
    cards = load_cards()
    first_input: WorkerTaskInput | None = None
    first_result: WorkerResult | None = None
    downstream_input: WorkerTaskInput | None = None

    # 动态任务板里可能有多个根任务。这里按主 Agent 给出的顺序推进 ready 卡片，
    # 直到有一张下游卡片的依赖已满足，方便观察 upstream_results 如何进入私有任务包。
    while downstream_input is None:
        downstream_card = downstream_ready_card(cards)
        if downstream_card is not None:
            current_done = {card.task_id: card for card in cards if card.status == "done"}
            downstream_input = build_worker_input(downstream_card, current_done)
            break

        ready_card = next_ready_card(cards)
        if ready_card is None:
            raise RuntimeError("没有 ready 任务卡片，任务板依赖可能不完整")

        worker_input = build_worker_input(ready_card, {card.task_id: card for card in cards if card.status == "done"})
        if first_input is None:
            first_input = worker_input
            print("===== 第一张 ready 卡片的私有任务包 =====")
            print(format_worker_prompt(worker_input)[:1800])

        async with AcpWorkerSession(resolve_agent(ready_card.assigned_agent)) as session:
            worker_result = await session.prompt(format_worker_prompt(worker_input))

        if first_result is None:
            first_result = worker_result

        done_card = replace(ready_card, status="done", result=worker_result.text, session_id=worker_result.session_id)
        cards = [done_card if card.task_id == done_card.task_id else card for card in cards]

        print(f"\n===== {done_card.task_id} 完成后的公共任务板 =====")
        print(render_board(cards))

    if first_input is None or first_result is None or downstream_input is None:
        raise RuntimeError("没有执行任何上游卡片，无法演示上下文隔离")

    print("\n===== 下游 ready 卡片的私有任务包 =====")
    print(format_worker_prompt(downstream_input)[:2200])

    save_json(
        CONTEXT_PATH,
        {
            "first_worker_input": first_input.to_dict(),
            "first_worker_result": first_result.to_dict(),
            "downstream_worker_input": downstream_input.to_dict(),
            "board_after_first_task": [card.to_dict() for card in cards],
        },
    )
    print(f"\nsaved: {CONTEXT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
