"""04 用 TriggerFlow state 托管主 Agent 调度 Loop。

这是课后最完整的 examples 脚本。为了可读性，它不依赖 common.py：
1. 读取杭州三日游团建目标
2. 主 Agent 动态生成任务板
3. TriggerFlow state 保存全局任务板
4. ACP 把私有任务包发给两个 Codex worker profile
5. worker_delta 流式输出子 Agent 文字增量
6. 主 Agent 回收结果并综合最终报告
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

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
from agently import Agently, TriggerFlow, TriggerFlowRuntimeData
from dotenv import find_dotenv, load_dotenv

LESSON_DIR = Path(__file__).resolve().parents[1]
GOAL_PATH = LESSON_DIR / "materials" / "product_goal.txt"
RUN_DIR = LESSON_DIR / ".demo_runs" / "multi_agent"
FLOW_RESULT_PATH = RUN_DIR / "04_triggerflow_result.json"
FULL_RUN_PATH = RUN_DIR / "05_hangzhou_team_building_result.json"
MAX_ATTEMPTS = 2
ALLOWED_AGENTS = {"codex_research", "codex_plan"}
MIN_TASK_CARDS = 2
MAX_TASK_CARDS = 6
TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

StreamCallback = Callable[[dict[str, object]], Awaitable[None]]

CODEX_BIN_DIR = Path("/Applications/Codex.app/Contents/Resources")
ACP_ENV = os.environ.copy()
if CODEX_BIN_DIR.is_dir():
    ACP_ENV["PATH"] = f"{CODEX_BIN_DIR}:{ACP_ENV.get('PATH', '')}"
NPX_COMMAND = os.getenv("ACP_NPX_COMMAND", "npx")


@dataclass(frozen=True)
class ProjectBrief:
    text: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "text": self.text}


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


@dataclass(frozen=True)
class BoardDecision:
    action: str
    task_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    command: str
    args: tuple[str, ...]

    def launcher_available(self) -> bool:
        if Path(self.command).is_file():
            return True
        return shutil.which(self.command, path=ACP_ENV.get("PATH")) is not None


AGENTS = {
    "codex_research": AgentSpec("codex_research", NPX_COMMAND, ("-y", "@agentclientprotocol/codex-acp@1.1.0")),
    "codex_plan": AgentSpec("codex_plan", NPX_COMMAND, ("-y", "@agentclientprotocol/codex-acp@1.1.0")),
}


def configure_model() -> None:
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


def save_json(path: Path, payload: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_text(text: str, limit: int = 1600) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...(已由主 Agent 截断)"


def create_board_request(project: ProjectBrief) -> Any:
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


async def build_task_board(project: ProjectBrief) -> tuple[list[TaskCard], object]:
    configure_model()
    request = create_board_request(project)
    raw_board = await request.async_start()
    return validate_model_board(raw_board), raw_board


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


def cards_to_state(cards: list[TaskCard]) -> dict[str, object]:
    return {
        "order": [card.task_id for card in cards],
        "cards": {card.task_id: card.to_dict() for card in cards},
    }


def cards_from_state(value: object) -> list[TaskCard]:
    if not isinstance(value, dict):
        raise ValueError("TriggerFlow state['board'] 必须是 dict")
    raw_cards = value.get("cards", {})
    order = value.get("order", [])
    if not isinstance(raw_cards, dict):
        raise ValueError("TriggerFlow state['board']['cards'] 必须是 dict")
    if not isinstance(order, list):
        order = list(raw_cards.keys())

    cards: list[TaskCard] = []
    for task_id in order:
        raw_card = raw_cards.get(str(task_id))
        if isinstance(raw_card, dict):
            cards.append(card_from_dict(raw_card))
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


def next_ready_card(cards: list[TaskCard]) -> TaskCard | None:
    done_ids = {card.task_id for card in cards if card.status == "done"}
    ready = [
        card
        for card in cards
        if card.status == "todo" and set(card.depends_on) <= done_ids
    ]
    return ready[0] if ready else None


def decide_board_action(cards: list[TaskCard]) -> BoardDecision:
    failed = [card.task_id for card in cards if card.status == "failed"]
    if failed:
        return BoardDecision("blocked", reason=f"存在失败任务:{', '.join(failed)}")

    target = next_ready_card(cards)
    if target is not None:
        action = "retry" if target.attempts > 0 else "dispatch"
        return BoardDecision(action, task_id=target.task_id, reason="依赖已满足")

    unfinished = [card.task_id for card in cards if card.status != "done"]
    if not unfinished:
        return BoardDecision("finish", reason="所有任务已完成")
    return BoardDecision("blocked", reason=f"等待依赖或运行中任务:{', '.join(unfinished)}")


def update_card(cards: list[TaskCard], updated: TaskCard) -> list[TaskCard]:
    return [updated if card.task_id == updated.task_id else card for card in cards]


def done_cards(cards: list[TaskCard]) -> dict[str, TaskCard]:
    return {card.task_id: card for card in cards if card.status == "done"}


def mark_retry_or_failed(card: TaskCard, error: BaseException) -> TaskCard:
    error_text = compact_text(str(error), limit=800)
    if card.attempts < MAX_ATTEMPTS:
        return replace(card, status="todo", result=f"上次尝试失败，等待重试:{error_text}")
    return replace(card, status="failed", result=f"已达到最大尝试次数:{error_text}")


def build_worker_input(card: TaskCard, upstream_cards: dict[str, TaskCard]) -> WorkerTaskInput:
    missing = [task_id for task_id in card.depends_on if task_id not in upstream_cards]
    if missing:
        raise RuntimeError(f"{card.task_id} 缺少上游结果:{', '.join(missing)}")

    return WorkerTaskInput(
        task_id=card.task_id,
        title=card.title,
        assigned_agent=card.assigned_agent,
        goal=card.goal,
        context=card.context,
        upstream_results={
            task_id: compact_text(upstream_cards[task_id].result)
            for task_id in card.depends_on
        },
        acceptance=(
            "只回答当前任务卡片，不扩展到完整主会话。",
            "列出可交给主 Agent 的结论、信息缺口和下游摘要。",
            "总长度控制在 500 个中文字符以内。",
        ),
        forbidden=(
            "不要假设自己知道完整主会话。",
            "不要读取或修改本地文件。",
            "不要运行命令，只返回文字结论。",
            "不要联网或实时查证；需要外部核验的事实列入信息缺口。",
        ),
    )


def format_worker_prompt(worker_input: WorkerTaskInput) -> str:
    upstream = "\n\n".join(
        f"[{task_id}]\n{result}" for task_id, result in worker_input.upstream_results.items()
    )
    acceptance = "\n".join(f"- {item}" for item in worker_input.acceptance)
    forbidden = "\n".join(f"- {item}" for item in worker_input.forbidden)
    payload = json.dumps(worker_input.to_dict(), ensure_ascii=False, indent=2)
    prompt = (
        "你是主 Agent 通过 ACP 调用的子 Agent。你只能看到下面这个私有任务包，"
        "看不到完整主会话，也不能读取本地文件或运行命令。\n\n"
        f"私有任务包:\n{payload}\n\n"
        f"验收口径:\n{acceptance}\n\n"
        f"执行边界:\n{forbidden}\n\n"
        "请用中文输出以下三段:\n"
        "1. 本任务结论\n"
        "2. 信息缺口或风险\n"
        "3. 可交给下游的摘要\n"
        "总长度控制在 500 个中文字符以内。"
    )
    if upstream:
        prompt += f"\n\n上游产出摘要:\n{upstream}"
    return prompt


def resolve_agent(name: str) -> AgentSpec:
    spec = AGENTS[name]
    if not spec.launcher_available():
        raise RuntimeError(f"{name} 的启动器不可见:{spec.command}")
    return spec


class CourseAcpClient:
    """ACP Client 只处理协议事件，并把子 Agent 文本增量转成 worker_delta。"""

    def __init__(self, on_stream: StreamCallback | None = None) -> None:
        self.on_stream = on_stream
        self.text_chunks: list[str] = []
        self.stream_emit_size = 0
        self.update_kinds: list[str] = []
        self.permission_decisions: list[str] = []

    def begin_turn(self) -> None:
        self.text_chunks = []
        self.stream_emit_size = 0
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
            await self.emit_text_delta(session_id, force="\n" in text)

    async def emit_text_delta(self, session_id: str, *, force: bool = False) -> None:
        if self.on_stream is None:
            return
        current = self.text
        if not force and len(current) - self.stream_emit_size < 120:
            return
        delta = current[self.stream_emit_size :]
        if not delta:
            return
        self.stream_emit_size = len(current)
        await self.on_stream({"event": "worker_delta", "session_id": session_id, "text": delta})

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
    def __init__(self, spec: AgentSpec, on_stream: StreamCallback | None = None) -> None:
        self.spec = spec
        self.client = CourseAcpClient(on_stream=on_stream)
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
        await self.client.emit_text_delta(self.session_id, force=True)
        return WorkerResult(
            agent_name=self.spec.name,
            agent_info=self.agent_info,
            session_id=self.session_id,
            stop_reason=str(response.stop_reason),
            update_kinds=tuple(self.client.update_kinds),
            text=self.client.text,
            permission_decisions=tuple(self.client.permission_decisions),
        )


async def call_worker_agent(
    worker_input: WorkerTaskInput,
    on_stream: StreamCallback | None = None,
) -> WorkerResult:
    prompt = format_worker_prompt(worker_input)
    async with AcpWorkerSession(resolve_agent(worker_input.assigned_agent), on_stream=on_stream) as session:
        return await session.prompt(prompt)


def board_for_model(cards: list[TaskCard]) -> list[dict[str, object]]:
    return [
        {
            "task_id": card.task_id,
            "title": card.title,
            "assigned_agent": card.assigned_agent,
            "status": card.status,
            "session_id": card.session_id,
            "result": compact_text(card.result, limit=2400),
        }
        for card in cards
    ]


async def synthesize_final_report(project: ProjectBrief, cards: list[TaskCard]) -> dict[str, object]:
    configure_model()
    report_schema = {
        "executive_summary": (str, "面向组织者的一段摘要。"),
        "confirmed_facts": [(str, "已确认事实。")],
        "information_gaps": [(str, "仍需补齐的信息。")],
        "draft_plan": [(str, "可执行方案建议，按步骤或天数组织。")],
        "risks": [(str, "预算、交通、体力、天气、安全等风险。")],
        "next_questions": [(str, "交给组织者确认的问题。")],
    }
    result = await (
        Agently.create_agent()
        .info({"原始业务目标": project.text, "任务板结果": board_for_model(cards)})
        .input("基于任务板结果，综合一份可交给组织者确认的最终摘要。")
        .output(report_schema)
        .async_start()
    )
    if not isinstance(result, dict):
        raise RuntimeError("final_report 必须是 dict")
    return result


def render_report(report: dict[str, object]) -> str:
    def string_list(key: str) -> list[str]:
        value = report.get(key, [])
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)] if value else []

    lines = [f"综合摘要:\n{report.get('executive_summary', '')}"]
    for title, key in [
        ("已确认事实", "confirmed_facts"),
        ("仍需补齐", "information_gaps"),
        ("方案草案", "draft_plan"),
        ("风险提示", "risks"),
        ("下一步问题", "next_questions"),
    ]:
        lines.append(f"\n{title}:")
        for item in string_list(key):
            lines.append(f"- {item}")
    return "\n".join(lines)


master_flow = TriggerFlow(name="master-agent-board-loop")


def read_board_state(data: TriggerFlowRuntimeData) -> list[TaskCard]:
    return cards_from_state(data.get_state("board", {}))


async def write_board_state(data: TriggerFlowRuntimeData, cards: list[TaskCard]) -> None:
    await data.async_set_state("board", cards_to_state(cards), emit=False)


async def append_state_item(data: TriggerFlowRuntimeData, key: str, item: dict[str, object]) -> None:
    values = data.get_state(key, []) or []
    if not isinstance(values, list):
        values = []
    values.append(item)
    await data.async_set_state(key, values, emit=False)


@master_flow.chunk("plan_board")
async def plan_board(data: TriggerFlowRuntimeData) -> str:
    """主 Agent 建板：只负责把业务目标拆成公共任务板，并写入 TriggerFlow state。"""

    project = ProjectBrief(text=str(data.input), source="execution.input")
    cards, raw_board = await build_task_board(project)

    await data.async_set_state("project_brief", project.to_dict(), emit=False)
    await data.async_set_state("raw_board", raw_board, emit=False)
    await data.async_set_state("worker_inputs", [], emit=False)
    await data.async_set_state("worker_results", [], emit=False)
    await write_board_state(data, cards)
    await data.async_put_into_stream({"event": "board_created", "task_count": len(cards)})
    return "board_created"


@master_flow.chunk("dispatch_tasks")
async def dispatch_tasks(data: TriggerFlowRuntimeData) -> str:
    """主 Agent 调度：观察任务板，按依赖逐张派发给 ACP 子 Agent。"""

    cards = read_board_state(data)

    while True:
        decision = decide_board_action(cards)
        await data.async_set_state("last_decision", asdict(decision), emit=False)

        if decision.action == "finish":
            break
        if decision.action == "blocked":
            raise RuntimeError(f"主 Agent 阻塞:{decision.reason}")

        target = next(card for card in cards if card.task_id == decision.task_id)
        running = replace(target, status="running", attempts=target.attempts + 1)
        cards = update_card(cards, running)
        await write_board_state(data, cards)
        await data.async_put_into_stream(
            {
                "event": "task_started",
                "action": decision.action,
                "task_id": running.task_id,
                "agent": running.assigned_agent,
                "attempts": running.attempts,
            }
        )

        worker_input = build_worker_input(running, done_cards(cards))
        await append_state_item(data, "worker_inputs", worker_input.to_dict())

        async def forward_worker_stream(item: dict[str, object]) -> None:
            payload = dict(item)
            payload["task_id"] = running.task_id
            payload["agent"] = running.assigned_agent
            await data.async_put_into_stream(payload)

        try:
            worker_result = await call_worker_agent(worker_input, on_stream=forward_worker_stream)
        except Exception as error:
            retried = mark_retry_or_failed(running, error)
            cards = update_card(cards, retried)
            await write_board_state(data, cards)
            await data.async_put_into_stream(
                {
                    "event": "task_retry" if retried.status == "todo" else "task_failed",
                    "task_id": retried.task_id,
                    "attempts": retried.attempts,
                    "reason": retried.result,
                }
            )
            continue

        finished = replace(running, status="done", result=worker_result.text, session_id=worker_result.session_id)
        cards = update_card(cards, finished)
        await write_board_state(data, cards)
        await append_state_item(data, "worker_results", worker_result.to_dict())
        await data.async_put_into_stream(
            {
                "event": "task_done",
                "task_id": finished.task_id,
                "agent": finished.assigned_agent,
                "session_id": finished.session_id,
                "stop_reason": worker_result.stop_reason,
            }
        )

    await data.async_put_into_stream({"event": "dispatch_done"})
    return "dispatch_done"


@master_flow.chunk("synthesize_report")
async def synthesize_report(data: TriggerFlowRuntimeData) -> dict[str, object]:
    """主 Agent 收尾：读取最终任务板，再发起一次模型请求综合交付物。"""

    cards = read_board_state(data)
    project_payload = data.get_state("project_brief", {}) or {}
    if not isinstance(project_payload, dict):
        raise RuntimeError("project_brief state 必须是 dict")
    project = ProjectBrief(
        text=str(project_payload["text"]),
        source=str(project_payload["source"]),
    )
    final_report = await synthesize_final_report(project, cards)
    await data.async_set_state("final_report", final_report, emit=False)
    await data.async_put_into_stream({"event": "final_report_ready"})
    return final_report


# TriggerFlow 编排只描述阶段关系：
# 1. execution 启动时先进入 plan_board；
# 2. plan_board 完成后触发 dispatch_tasks；
# 3. dispatch_tasks 完成后触发 synthesize_report。
master_flow.to(plan_board)
master_flow.when(plan_board).to(dispatch_tasks)
master_flow.when(dispatch_tasks).to(synthesize_report)


def print_stream_event(event: object) -> None:
    if not isinstance(event, dict):
        print(event, flush=True)
        return
    if event.get("event") == "worker_delta":
        print(f"[{event.get('task_id')}:{event.get('agent')}] {event.get('text', '')}", end="", flush=True)
        return
    print(f"\n{event}", flush=True)


async def run_master_flow(project_text: str) -> tuple[dict[str, Any], list[object]]:
    execution = master_flow.create_execution(auto_close=False)
    start_task = asyncio.create_task(execution.async_start(project_text))

    while execution.result.get_meta().get("status") == "created":
        if start_task.done():
            await start_task
            break
        await asyncio.sleep(0.05)

    async def close_after_start() -> dict[str, Any] | None:
        await start_task
        return await execution.async_close()

    close_task = asyncio.create_task(close_after_start())
    stream_events: list[object] = []
    async for event in execution.get_async_runtime_stream(timeout=None):
        stream_events.append(event)
        print_stream_event(event)

    state = await close_task
    if state is None:
        raise RuntimeError("TriggerFlow 没有返回 close snapshot")
    return state, stream_events


def save_run_result(path: Path, project: ProjectBrief, cards: list[TaskCard], final_report: dict[str, object], stream_events: list[object]) -> None:
    save_json(
        path,
        {
            "project_brief": project.to_dict(),
            "tasks": [card.to_dict() for card in cards],
            "final_report": final_report,
            "stream_events": stream_events,
        },
    )


async def main() -> None:
    project = load_project_brief()
    state, stream_events = await run_master_flow(project.text)

    cards = cards_from_state(state["board"])
    final_report = state["final_report"]
    if not isinstance(final_report, dict):
        raise RuntimeError("final_report state 必须是 dict")

    print("\n--- board ---")
    print(render_board(cards))
    print("\n--- final report ---")
    print(render_report(final_report))

    save_run_result(FLOW_RESULT_PATH, project, cards, final_report, stream_events)
    save_run_result(FULL_RUN_PATH, project, cards, final_report, stream_events)
    print(f"\nsaved: {FLOW_RESULT_PATH}")
    print(f"saved: {FULL_RUN_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
