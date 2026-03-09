"""ComfyUI API client for remote workflow execution.

Replaces the BFL API client from BFL_FLUXdemos with local/LAN ComfyUI calls.
Connects from Mac to ComfyUI running on Windows (3080) via LAN.

Usage:
    client = ComfyUIClient("http://192.168.x.x:8188")
    result = client.run_workflow(workflow, {"image": "path/to/frame.png"})
"""

import json
import time
import uuid
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Any

try:
    import websocket  # websocket-client
    HAS_WS = True
except ImportError:
    HAS_WS = False


class ComfyUIClient:
    """Client for ComfyUI's HTTP/WebSocket API."""

    def __init__(self, host: str = "http://127.0.0.1:8188"):
        self.host = host.rstrip("/")
        self.client_id = str(uuid.uuid4())

    # ── Core API ──────────────────────────────────────────────

    def queue_prompt(self, workflow: dict) -> str:
        """Submit workflow to /prompt queue. Returns prompt_id."""
        payload = json.dumps({
            "prompt": workflow,
            "client_id": self.client_id,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"ComfyUI rejected prompt ({e.code}): {body}") from e
        return data["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt_id."""
        url = f"{self.host}/history/{prompt_id}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download generated image from ComfyUI server."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        })
        url = f"{self.host}/view?{params}"
        with urllib.request.urlopen(url) as resp:
            return resp.read()

    def upload_image(self, filepath: str, subfolder: str = "", overwrite: bool = True) -> dict:
        """Upload image to ComfyUI input directory."""
        path = Path(filepath)
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + path.read_bytes() + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
            f"{subfolder}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"{'true' if overwrite else 'false'}\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def get_queue(self) -> dict:
        """Get current queue status."""
        with urllib.request.urlopen(f"{self.host}/queue") as resp:
            return json.loads(resp.read())

    def interrupt(self) -> None:
        """Interrupt current generation."""
        req = urllib.request.Request(f"{self.host}/interrupt", method="POST")
        urllib.request.urlopen(req)

    # ── Workflow Execution ────────────────────────────────────

    def run_workflow(
        self,
        workflow: dict,
        images: dict[str, str] | None = None,
        output_dir: str | Path | None = None,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
        on_progress: Any = None,
    ) -> list[Path]:
        """Execute workflow end-to-end. Upload images, queue, wait, download results.

        Args:
            workflow: ComfyUI workflow API format (node dict)
            images: {node_id_field: local_path} to upload and inject
            output_dir: where to save results (default: ./outputs/)
            poll_interval: seconds between status checks
            timeout: max wait in seconds
            on_progress: callback(node_id, current, total) for progress

        Returns:
            List of saved output file paths
        """
        output_dir = Path(output_dir or "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Upload input images
        if images:
            for key, path in images.items():
                result = self.upload_image(path)
                # key format: "node_id.field" e.g. "1.image"
                node_id, field = key.split(".", 1)
                workflow[node_id]["inputs"][field] = result["name"]

        prompt_id = self.queue_prompt(workflow)

        # Wait for completion
        if HAS_WS:
            return self._wait_ws(prompt_id, output_dir, timeout, on_progress)
        return self._wait_poll(prompt_id, output_dir, poll_interval, timeout)

    def _wait_poll(
        self, prompt_id: str, output_dir: Path, interval: float, timeout: float
    ) -> list[Path]:
        """Poll /history until prompt completes."""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return self._download_outputs(history[prompt_id], output_dir)
            time.sleep(interval)
        raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout}s")

    def _wait_ws(
        self, prompt_id: str, output_dir: Path, timeout: float, on_progress: Any
    ) -> list[Path]:
        """Wait via WebSocket for real-time progress."""
        ws_url = self.host.replace("http", "ws") + f"/ws?clientId={self.client_id}"
        ws = websocket.create_connection(ws_url, timeout=timeout)
        try:
            while True:
                msg = json.loads(ws.recv())
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "progress" and on_progress:
                    on_progress(data.get("node"), data.get("value"), data.get("max"))

                if msg_type == "executing" and data.get("node") is None:
                    if data.get("prompt_id") == prompt_id:
                        break

                if msg_type == "execution_error":
                    raise RuntimeError(f"Execution error: {data}")
        finally:
            ws.close()

        history = self.get_history(prompt_id)
        return self._download_outputs(history[prompt_id], output_dir)

    def _download_outputs(self, history_entry: dict, output_dir: Path) -> list[Path]:
        """Download all output images from a completed prompt."""
        saved = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_output in outputs.items():
            for img_info in node_output.get("images", []):
                data = self.get_image(
                    img_info["filename"],
                    img_info.get("subfolder", ""),
                    img_info.get("type", "output"),
                )
                out_path = output_dir / img_info["filename"]
                out_path.write_bytes(data)
                saved.append(out_path)
        return saved


# ── Workflow Template Helpers ─────────────────────────────────

def load_workflow(path: str | Path) -> dict:
    """Load workflow JSON file."""
    return json.loads(Path(path).read_text())


def set_node_input(workflow: dict, node_id: str, field: str, value: Any) -> None:
    """Set a specific input value in a workflow node."""
    workflow[node_id]["inputs"][field] = value


def parametrize_workflow(
    workflow: dict,
    params: dict[str, Any],
) -> dict:
    """Apply parameter overrides to workflow nodes.

    params format: {"node_id.field": value}
    e.g. {"3.seed": 42, "6.text": "my prompt", "11.denoise": 0.7}
    """
    import copy
    wf = copy.deepcopy(workflow)
    for key, value in params.items():
        node_id, field = key.split(".", 1)
        wf[node_id]["inputs"][field] = value
    return wf
