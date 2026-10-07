import io

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

import apps.api.routes.agent as agent_routes
from apps.api.dependencies import Settings
from apps.api.main import app
from tests.unit.fake_anthropic import FakeAnthropicClient, FakeMessage, FakeTextBlock, FakeToolUseBlock


@pytest.fixture(autouse=True)
def _reset_module_state():
    """The routes module holds registries/runs at module scope — reset between tests."""
    agent_routes._dataset_registry.__init__()
    agent_routes._experiment_registry.__init__()
    agent_routes._runs.clear()
    yield
    agent_routes._runs.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _churn_csv_bytes() -> bytes:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(size=n), "churned": (rng.normal(size=n) > 0).astype(int)})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()


class TestUploadDataset:
    def test_upload_returns_dataset_metadata(self, client):
        response = client.post(
            "/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_rows"] == 200
        assert set(body["columns"]) == {"x1", "churned"}
        assert body["dataset_version_id"]


class TestStartRun:
    def test_unknown_dataset_version_id_returns_404(self, client):
        response = client.post(
            "/api/runs", json={"dataset_version_id": "does_not_exist", "objective": "predict churn"}
        )
        assert response.status_code == 404

    def test_gemini_provider_without_model_configured_returns_400(self, client, monkeypatch):
        # Settings reads .env directly (not just os.environ), so override it
        # via get_settings rather than monkeypatching env vars.
        no_gemini_model = Settings(gemini_chat_model=None, gemini_api_key="k", litellm_gemini_base_url="http://x")
        monkeypatch.setattr(agent_routes, "get_settings", lambda: no_gemini_model)

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]

        response = client.post(
            "/api/runs",
            json={"dataset_version_id": dataset_version_id, "objective": "predict churn", "provider": "gemini"},
        )
        assert response.status_code == 400

    def test_gemini_provider_without_credentials_returns_400(self, client, monkeypatch):
        no_gemini_creds = Settings(gemini_chat_model="some-model", gemini_api_key=None, litellm_gemini_base_url=None)
        monkeypatch.setattr(agent_routes, "get_settings", lambda: no_gemini_creds)

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]

        response = client.post(
            "/api/runs",
            json={"dataset_version_id": dataset_version_id, "objective": "predict churn", "provider": "gemini"},
        )
        assert response.status_code == 400

    def test_start_run_with_fake_client_reaches_done(self, client, monkeypatch):
        fake = FakeAnthropicClient(
            [FakeMessage(content=[FakeTextBlock(text="No leakage found, ready.")], stop_reason="end_turn")]
        )
        monkeypatch.setattr(agent_routes, "_build_client", lambda provider, settings: fake)

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]

        response = client.post(
            "/api/runs", json={"dataset_version_id": dataset_version_id, "objective": "predict churn"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "done"
        assert body["final_text"] == "No leakage found, ready."
        assert body["run_id"]


class TestAskUserAndResume:
    def test_full_pause_and_resume_flow(self, client, monkeypatch):
        fake = FakeAnthropicClient(
            [
                FakeMessage(
                    content=[
                        FakeToolUseBlock(
                            id="ask_1", name="ask_user", input={"question": "Which column is the target?", "reason": "ambiguous"}
                        )
                    ],
                    stop_reason="tool_use",
                )
            ]
        )
        monkeypatch.setattr(agent_routes, "_build_client", lambda provider, settings: fake)

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]

        start = client.post("/api/runs", json={"dataset_version_id": dataset_version_id, "objective": "predict churn"})
        body = start.json()
        assert body["status"] == "awaiting_user_input"
        assert body["pending_question"]["question"] == "Which column is the target?"
        run_id = body["run_id"]

        # append a follow-up response for after the answer is supplied
        fake.messages._responses.append(
            FakeMessage(content=[FakeTextBlock(text="Using churned as target.")], stop_reason="end_turn")
        )

        answer = client.post(f"/api/runs/{run_id}/answer", json={"answer": "churned"})
        assert answer.status_code == 200
        answer_body = answer.json()
        assert answer_body["status"] == "done"
        assert answer_body["final_text"] == "Using churned as target."

    def test_answer_on_unknown_run_id_returns_404(self, client):
        response = client.post("/api/runs/does_not_exist/answer", json={"answer": "x"})
        assert response.status_code == 404

    def test_answer_when_not_awaiting_input_returns_400(self, client, monkeypatch):
        fake = FakeAnthropicClient(
            [FakeMessage(content=[FakeTextBlock(text="done")], stop_reason="end_turn")]
        )
        monkeypatch.setattr(agent_routes, "_build_client", lambda provider, settings: fake)

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]
        start = client.post("/api/runs", json={"dataset_version_id": dataset_version_id, "objective": "predict churn"})
        run_id = start.json()["run_id"]

        response = client.post(f"/api/runs/{run_id}/answer", json={"answer": "x"})
        assert response.status_code == 400


class TestProviderErrorMapping:
    def test_provider_auth_error_becomes_502_not_500(self, client, monkeypatch):
        import anthropic
        import httpx2

        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx2.Response(401, request=request, json={"error": "bad key"})
        auth_error = anthropic.AuthenticationError(message="bad key", response=response, body=None)

        class _RaisingClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise auth_error

        monkeypatch.setattr(agent_routes, "_build_client", lambda provider, settings: _RaisingClient())

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]

        response = client.post(
            "/api/runs", json={"dataset_version_id": dataset_version_id, "objective": "predict churn"}
        )
        assert response.status_code == 502

    def test_gemini_provider_error_becomes_502_not_500(self, client, monkeypatch):
        from google.genai import errors as genai_errors

        gemini_error = genai_errors.ClientError(code=401, response_json={"error": {"message": "bad key"}})

        class _RaisingClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise gemini_error

        monkeypatch.setattr(agent_routes, "_build_client", lambda provider, settings: _RaisingClient())

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]

        response = client.post(
            "/api/runs",
            json={"dataset_version_id": dataset_version_id, "objective": "predict churn", "provider": "gemini", "model": "m"},
        )
        assert response.status_code == 502
        assert "bad key" in response.json()["detail"]


class TestGetRun:
    def test_get_run_returns_last_known_status(self, client, monkeypatch):
        fake = FakeAnthropicClient(
            [FakeMessage(content=[FakeTextBlock(text="done")], stop_reason="end_turn")]
        )
        monkeypatch.setattr(agent_routes, "_build_client", lambda provider, settings: fake)

        upload = client.post("/api/datasets", files={"file": ("churn.csv", _churn_csv_bytes(), "text/csv")})
        dataset_version_id = upload.json()["dataset_version_id"]
        start = client.post("/api/runs", json={"dataset_version_id": dataset_version_id, "objective": "predict churn"})
        run_id = start.json()["run_id"]

        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "done"

    def test_get_unknown_run_id_returns_404(self, client):
        response = client.get("/api/runs/does_not_exist")
        assert response.status_code == 404
