from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi.testclient import TestClient

from aloepri.demo.app import DemoMessage, DemoRequest, DemoResponse, TokenTrace, create_demo_app


class FakeGateway:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def generate(self, request: DemoRequest) -> DemoResponse:
        return DemoResponse(
            request_id="demo-request",
            model_id="demo-model",
            key_id="demo-key",
            prompt=request.prompt,
            plain_input_ids=[1, 2],
            private_input_ids=[9, 8],
            private_input_text="cipher input",
            private_output_ids=[7],
            private_output_text="cipher output",
            recovered_output_ids=[3],
            answer="恢复后的回答",
            input_trace=[
                TokenTrace(
                    position=0,
                    plain_id=1,
                    private_id=9,
                    plain_piece="A",
                    private_piece="Z",
                )
            ],
            input_tokens=2,
            output_tokens=1,
            ttft_ms=12.5,
            tpot_ms=4.0,
            client_roundtrip_ms=18.0,
        )

    def stream(self, request: DemoRequest) -> Iterator[dict[str, object]]:
        yield {
            "type": "start",
            "model_id": "demo-model",
            "key_id": "demo-key",
            "prompt": request.prompt,
            "plain_input_ids": [1, 2],
            "private_input_ids": [9, 8],
            "private_input_text": "cipher input",
            "input_trace": [],
            "input_tokens": 2,
        }
        yield {
            "type": "token",
            "request_id": "demo-request",
            "sequence_no": 0,
            "private_output_id": 7,
            "recovered_output_id": 3,
            "private_output_text": "cipher output",
            "answer": "恢",
            "elapsed_ms": 12.5,
            "ttft_ms": 12.5,
            "tpot_ms": 0.0,
            "output_tokens": 1,
        }
        yield {
            "type": "done",
            "request_id": "demo-request",
            "model_id": "demo-model",
            "key_id": "demo-key",
            "output_tokens": 1,
            "ttft_ms": 12.5,
            "tpot_ms": 0.0,
            "client_roundtrip_ms": 18.0,
            "answer": "恢",
        }


def test_demo_page_and_observable_flow() -> None:
    gateway = FakeGateway()
    with TestClient(create_demo_app(gateway)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "有什么可以帮你" in page.text
        assert "显示隐私过程" in page.text
        assert 'id="saved-chat-list"' in page.text
        assert "回答完成后自动保存在此浏览器" in page.text
        app_script = client.get("/assets/app.js")
        assert app_script.status_code == 200
        assert "aloepri-saved-conversations-v1" in app_script.text
        assert "restoreConversation" in app_script.text
        assert "history: conversationHistory.slice(-MAX_CONTEXT_MESSAGES)" in app_script.text
        privacy = client.get("/privacy")
        assert privacy.status_code == 200
        assert "在线推理：真实数据流架构图" in privacy.text
        assert "亲手输入一句话" in privacy.text
        assert 'id="lab-form"' in privacy.text
        assert 'class="app-shell"' in privacy.text
        assert "TEE 国密原理" in privacy.text
        assert "查看模型服务器实际收到的请求载荷" in privacy.text
        tee_privacy = client.get("/privacy/tee")
        assert tee_privacy.status_code == 200
        assert "TEE 国密原理" in tee_privacy.text
        assert 'id="tee-lab-form"' in tee_privacy.text
        assert "加密后的提示词" in tee_privacy.text
        assert 'id="tee-loop-sample"' in tee_privacy.text
        assert "privacy-route-overview.png" in tee_privacy.text
        assert "privacy-route-tee.png" in tee_privacy.text
        assert "privacy-route-security.png" in tee_privacy.text
        assert 'id="stage-head-value"' in tee_privacy.text
        assert "下面只保留讲解所需的五个结果" in tee_privacy.text
        route_asset = client.get("/assets/privacy-route-tee.png")
        assert route_asset.status_code == 200
        assert route_asset.headers["content-type"] == "image/png"
        tee_script = client.get("/assets/privacy-tee.js")
        assert tee_script.status_code == 200
        assert "event.security_mode !== \"tee_gm\"" in tee_script.text
        assert "renderLoopSample(event)" in tee_script.text

        response = client.post(
            "/api/generate",
            json={"prompt": "真实问题", "max_new_tokens": 16, "temperature": 0.0},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["prompt"] == "真实问题"
        assert payload["private_input_ids"] == [9, 8]
        assert payload["private_output_ids"] == [7]
        assert payload["answer"] == "恢复后的回答"

        with client.stream(
            "POST",
            "/api/generate/stream",
            json={"prompt": "真实问题", "max_new_tokens": 16, "temperature": 0.0},
        ) as stream:
            assert stream.status_code == 200
            events = [
                json.loads(line.removeprefix("data: "))
                for line in stream.iter_lines()
                if line.startswith("data: ")
            ]
        assert [event["type"] for event in events] == ["start", "token", "done"]
        assert events[1]["private_output_id"] == 7
        assert events[1]["answer"] == "恢"
    assert gateway.closed


def test_demo_request_accepts_bounded_chat_history() -> None:
    request = DemoRequest(
        prompt="继续解释",
        history=[
            DemoMessage(role="user", content="上一轮问题"),
            DemoMessage(role="assistant", content="上一轮回答"),
        ],
    )

    assert [message.role for message in request.history] == ["user", "assistant"]


def test_public_demo_requires_login_and_supports_polling_jobs() -> None:
    gateway = FakeGateway()
    app = create_demo_app(
        gateway,
        access_code="shared-code-1234",
        session_secret="test-session-secret",
    )
    with TestClient(app, base_url="https://demo.example") as client:
        page = client.get("/", follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"] == "/login"
        assert client.post("/api/generate", json={"prompt": "blocked"}).status_code == 401

        rejected = client.post(
            "/login",
            content="access_code=wrong-code-000",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert rejected.status_code == 303
        assert rejected.headers["location"] == "/login?error=invalid"

        accepted = client.post(
            "/login",
            content="access_code=shared-code-1234",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert accepted.status_code == 303
        assert "HttpOnly" in accepted.headers["set-cookie"]
        assert "Secure" in accepted.headers["set-cookie"]
        assert client.get("/").status_code == 200

        created = client.post("/api/generate/jobs", json={"prompt": "public question"})
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        result: dict[str, object] = {}
        for _ in range(100):
            response = client.get(f"/api/generate/jobs/{job_id}?after=0")
            assert response.status_code == 200
            result = response.json()
            if result["done"]:
                break
            time.sleep(0.01)
        assert result["done"] is True
        events = result["events"]
        assert isinstance(events, list)
        assert [event["type"] for event in events] == ["start", "token", "done"]
    assert gateway.closed
