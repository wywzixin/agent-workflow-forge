"""02 验证两个官方 Codex ACP worker 可以被调用。

这个脚本只讲 ACP 连接本身：
启动官方 Codex ACP 连接器 -> initialize -> new_session -> prompt -> 同 session 再 prompt。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass
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
SMOKE_PATH = RUN_DIR / "02_worker_smoke.json"

CODEX_BIN_DIR = Path("/Applications/Codex.app/Contents/Resources")
ACP_ENV = os.environ.copy()
if CODEX_BIN_DIR.is_dir():
    # VS Code/Jupyter 的 PATH 常常不含 Codex.app 内置 CLI 目录。
    ACP_ENV["PATH"] = f"{CODEX_BIN_DIR}:{ACP_ENV.get('PATH', '')}"
NPX_COMMAND = os.getenv("ACP_NPX_COMMAND", "npx")


@dataclass(frozen=True)
class AgentSpec:
    """一个外部子 Agent profile：这里用两个名字代表两个执行者。"""

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
    "codex_research": AgentSpec(
        name="codex_research",
        command=NPX_COMMAND,
        args=("-y", "@agentclientprotocol/codex-acp@1.1.0"),
    ),
    "codex_plan": AgentSpec(
        name="codex_plan",
        command=NPX_COMMAND,
        args=("-y", "@agentclientprotocol/codex-acp@1.1.0"),
    ),
}


def resolve_agent(name: str) -> AgentSpec:
    spec = AGENTS[name]
    if not spec.launcher_available():
        raise RuntimeError(f"{name} 的启动器不可见:{spec.command}")
    return spec


class CourseAcpClient:
    """主 Agent 侧最小 ACP Client：只收公开文本，拒绝读写和命令权限。"""

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
    """一个官方 Codex ACP session。prompt() 可以在同一个 session 里多次调用。"""

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
            client_info=Implementation(
                name="course-master-agent",
                title="课程主 Agent 调度器",
                version="0.2.0",
            ),
        )
        self.agent_info = init.agent_info.name if init.agent_info is not None else "(未提供 agentInfo)"

        session = await connection.new_session(cwd=str(LESSON_DIR), mcp_servers=[])
        self.session_id = session.session_id
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._process_cm is not None:
            await self._process_cm.__aexit__(exc_type, exc, tb)

    async def prompt(self, text: str) -> WorkerResult:
        if self._connection is None or not self.session_id:
            raise RuntimeError("ACP session 尚未初始化")

        self.client.begin_turn()
        response = await self._connection.prompt(
            prompt=[text_block(text)],
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


def save_json(path: Path, payload: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    async with AcpWorkerSession(resolve_agent("codex_research")) as research:
        research_result = await research.prompt(
            "只回复 CODEX_RESEARCH_OK。不要读取文件，不要运行命令，不要修改文件。"
        )

    async with AcpWorkerSession(resolve_agent("codex_plan")) as planner:
        plan_first = await planner.prompt(
            "只回复 CODEX_PLAN_TURN_1。不要读取文件，不要运行命令，不要修改文件。"
        )
        plan_second = await planner.prompt(
            "只回复 CODEX_PLAN_TURN_2。不要读取文件，不要运行命令，不要修改文件。"
        )

    print("research session :", research_result.session_id)
    print("research text    :", research_result.text)
    print("plan same session:", plan_first.session_id == plan_second.session_id)
    print("plan session     :", plan_first.session_id)
    print("plan first       :", plan_first.text)
    print("plan second      :", plan_second.text)

    save_json(
        SMOKE_PATH,
        {
            "codex_research": research_result.to_dict(),
            "codex_plan_first": plan_first.to_dict(),
            "codex_plan_second": plan_second.to_dict(),
            "codex_plan_same_session": plan_first.session_id == plan_second.session_id,
        },
    )
    print(f"\nsaved: {SMOKE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
