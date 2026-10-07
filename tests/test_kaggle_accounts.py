"""Two Kaggle accounts (owner, 7 Oct 2026): a job goes to the second when the first is out of GPU
quota, and every later call on a kernel uses the credentials of the account that owns it."""

import json

from reva.kaggle import Accounts, Kaggle, Push, account_env, from_env


def test_account_env_drops_the_other_accounts_login(tmp_path):
    base = {"PATH": "/bin", "KAGGLE_API_TOKEN": "KGAT_first", "KAGGLE_USERNAME": "first", "KAGGLE_KEY": "k1"}
    env = account_env("second", "0123abcd", tmp_path / "h", base)
    assert "KAGGLE_API_TOKEN" not in env  # the CLI tries the token first, so a stray one would win
    assert env["KAGGLE_USERNAME"] == "second" and env["KAGGLE_KEY"] == "0123abcd" and env["PATH"] == "/bin"
    assert env["HOME"] == str(tmp_path / "h")  # no ~/.kaggle/access_token from another account


def test_account_env_takes_the_new_token_format(tmp_path):
    env = account_env("second", " KGAT_abc\n", tmp_path / "h", {"KAGGLE_USERNAME": "first", "KAGGLE_KEY": "k1"})
    assert env["KAGGLE_API_TOKEN"] == "KGAT_abc"
    assert "KAGGLE_USERNAME" not in env and "KAGGLE_KEY" not in env


def test_from_env_keeps_priority_order(tmp_path):
    environ = {"KAGGLE_USERNAME": "First", "KAGGLE_KEY": "k1", "KAGGLE_USERNAME_NEW": "second", "KAGGLE_KEY_NEW": "k2"}
    assert from_env(tmp_path, environ).users == ["first", "second"]
    assert from_env(tmp_path, {"KAGGLE_USERNAME": "first", "KAGGLE_KEY": "k1"}).users == ["first"]
    assert isinstance(from_env(tmp_path, {}), Kaggle)  # no secrets: the CLI's own login


class Recorder(Kaggle):
    def __init__(self, name, calls):
        super().__init__(runner=lambda cmd: (0, ""))
        self.name, self.calls = name, calls

    def status(self, slug):
        self.calls.append((self.name, slug))
        return "running", ""

    def push(self, kernel_dir, timeout_s=None, accelerator=None):
        self.calls.append((self.name, "push"))
        return Push(1, "u")


def test_calls_go_to_the_kernel_owner(tmp_path):
    calls = []
    acc = Accounts({"first": Recorder("a", calls), "second": Recorder("b", calls)})
    acc.status("second/reva-x")
    acc.status("First/reva-y")
    (tmp_path / "kernel-metadata.json").write_text(json.dumps({"id": "second/reva-z"}))
    acc.push(tmp_path)
    assert calls == [("b", "second/reva-x"), ("a", "First/reva-y"), ("b", "push")]
