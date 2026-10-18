"""05 复盘杭州三日游团建完整案例结果。

04 已经负责真实运行主 Agent Loop。05 不再 import 04，也不再隐藏执行逻辑；
它只读取 04 保存的完整运行产物，检查任务板是否真的完成，并打印最终报告。
"""

from __future__ import annotations

import json
from pathlib import Path

LESSON_DIR = Path(__file__).resolve().parents[1]
FULL_RUN_PATH = LESSON_DIR / ".demo_runs" / "multi_agent" / "05_hangzhou_team_building_result.json"


def load_json(path: Path) -> object:
    if not path.exists():
        raise RuntimeError(
            f"没有找到完整案例结果:{path}\n"
            "请先运行 scripts/04_triggerflow_master_loop.py。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


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


def main() -> None:
    payload = load_json(FULL_RUN_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError("完整案例结果必须是 dict")

    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise RuntimeError("tasks 必须是 list")

    final_report = payload.get("final_report", {})
    if not isinstance(final_report, dict):
        raise RuntimeError("final_report 必须是 dict")

    stream_events = payload.get("stream_events", [])
    worker_delta_count = 0
    if isinstance(stream_events, list):
        worker_delta_count = sum(
            1
            for event in stream_events
            if isinstance(event, dict) and event.get("event") == "worker_delta"
        )

    print("===== task board check =====")
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            continue
        print(
            raw_task.get("task_id"),
            raw_task.get("assigned_agent"),
            raw_task.get("status"),
            str(raw_task.get("session_id", ""))[:8],
        )

    unfinished = [
        str(task.get("task_id"))
        for task in tasks
        if isinstance(task, dict) and task.get("status") != "done"
    ]
    if unfinished:
        raise RuntimeError(f"仍有任务未完成:{', '.join(unfinished)}")

    print("\n===== stream check =====")
    print("worker_delta_count:", worker_delta_count)
    if worker_delta_count == 0:
        raise RuntimeError("没有观察到 worker_delta，说明长任务缺少流式进展输出。")

    print("\n===== final report =====")
    print(render_report(final_report))
    print(f"\nchecked: {FULL_RUN_PATH}")


if __name__ == "__main__":
    main()
