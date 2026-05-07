"""Round-trip tests for job_registry: create/get/list/cleanup against a temp DB."""

from __future__ import annotations

import time


def test_create_and_get_job(job_registry_temp):
    jr = job_registry_temp
    job_id = jr.create_job(
        user_id="alice",
        filename="test.pdf",
        pdf_path="/tmp/fake.pdf",
        config_json="{}",
    )
    assert job_id

    job = jr.get_job(job_id)
    assert job is not None
    assert job["user_id"] == "alice"
    assert job["filename"] == "test.pdf"
    assert job["status"] == "queued"


def test_update_job_changes_status(job_registry_temp):
    jr = job_registry_temp
    job_id = jr.create_job(
        user_id="bob",
        filename="x.pdf",
        pdf_path="/tmp/x.pdf",
        config_json="{}",
    )
    jr.update_job(job_id, status="running", started_at=int(time.time()))
    job = jr.get_job(job_id)
    assert job["status"] == "running"
    assert job["started_at"] > 0


def test_get_user_jobs_returns_users_jobs_only(job_registry_temp):
    jr = job_registry_temp
    a = jr.create_job(user_id="alice", filename="a.pdf", pdf_path="/tmp/a", config_json="{}")
    b = jr.create_job(user_id="bob", filename="b.pdf", pdf_path="/tmp/b", config_json="{}")

    alice_jobs = jr.get_user_jobs("alice")
    bob_jobs = jr.get_user_jobs("bob")

    assert {j["job_id"] for j in alice_jobs} == {a}
    assert {j["job_id"] for j in bob_jobs} == {b}


def test_next_queued_job_picks_oldest_first(job_registry_temp):
    jr = job_registry_temp
    first = jr.create_job(user_id="u", filename="1.pdf", pdf_path="/tmp/1", config_json="{}")
    time.sleep(0.01)
    second = jr.create_job(user_id="u", filename="2.pdf", pdf_path="/tmp/2", config_json="{}")

    next_job = jr.next_queued_job()
    assert next_job is not None
    assert next_job["job_id"] == first

    # Mark first as running so the queue advances
    jr.update_job(first, status="running")
    next_job2 = jr.next_queued_job()
    assert next_job2 is not None
    assert next_job2["job_id"] == second


def test_count_running_and_queued(job_registry_temp):
    jr = job_registry_temp
    a = jr.create_job(user_id="u", filename="a", pdf_path="/tmp/a", config_json="{}")
    jr.create_job(user_id="u", filename="b", pdf_path="/tmp/b", config_json="{}")
    jr.update_job(a, status="running")

    assert jr.count_running_jobs() == 1
    assert jr.count_queued_jobs() == 1


def test_requeue_orphaned_running(job_registry_temp):
    """On API restart, jobs marked 'running' from a prior process are
    re-queued so they don't sit forever."""
    jr = job_registry_temp
    job_id = jr.create_job(user_id="u", filename="a", pdf_path="/tmp/a", config_json="{}")
    jr.update_job(job_id, status="running")

    moved = jr.requeue_orphaned_running()
    assert moved == 1
    assert jr.get_job(job_id)["status"] == "queued"


def test_save_and_load_results(job_registry_temp):
    jr = job_registry_temp
    job_id = jr.create_job(user_id="u", filename="a", pdf_path="/tmp/a", config_json="{}")

    payload = {"fence_pages": [{"page_idx": 0, "fence_text": "1"}], "non_fence_pages": []}
    jr.save_results(job_id, payload)
    loaded = jr.load_results(job_id)
    assert loaded == payload


def test_delete_job_removes_row(job_registry_temp):
    jr = job_registry_temp
    job_id = jr.create_job(user_id="u", filename="a", pdf_path="/tmp/a", config_json="{}")
    jr.delete_job(job_id)
    assert jr.get_job(job_id) is None
