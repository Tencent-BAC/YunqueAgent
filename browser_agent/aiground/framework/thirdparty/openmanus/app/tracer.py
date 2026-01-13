# -*- coding: utf-8 -*-
"""
tracer tracer for mcp_server

- Writes JSONL by default (one JSON object per line).
- Can persist data:image/*;base64,... blobs as image files on disk and replace them with relative paths
  in the trace, avoiding huge base64 payloads inside JSONL.
"""

import base64
import json
import os
import re
import uuid
from typing import Optional, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field


class TraceMessage(BaseModel):
    req: Optional[dict] = Field(description="request message", default=None)
    rsp: Optional[ChatCompletion] = Field(description="response message", default=None)


_DATA_IMAGE_RE = re.compile(r"^data:(image/[^;]+);base64,(.*)$", re.DOTALL)


class Tracer(object):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        file_path: Optional[str] = None,
        images_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir/session_id: legacy mode, output to {data_dir}/{session_id}.jsonl
            file_path: explicit output file path (highest priority)
            images_dir: directory to persist images (defaults to sibling `images/` next to the trace file)
        """
        self._data_dir = data_dir or ""
        self._session_id = session_id or ""
        self._file_path = file_path
        self._images_dir = images_dir
        self._fout = None
        self._img_idx = 0

        # prepare dirs
        if self._file_path:
            os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        elif self._data_dir:
            os.makedirs(self._data_dir, exist_ok=True)

    def init(self, mode: str = "a"):
        if self._file_path:
            path = self._file_path
        else:
            path = os.path.join(self._data_dir, f"{self._session_id}.jsonl")

        if not self._images_dir:
            self._images_dir = os.path.join(os.path.dirname(path) or ".", "images")
        os.makedirs(self._images_dir, exist_ok=True)

        self._fout = open(path, mode, encoding="utf-8")
        return self

    def exit(self):
        if self._fout is not None:
            self._fout.close()
            self._fout = None

    def trace(self, message: Union[TraceMessage, dict]):
        if self._fout is None:
            raise RuntimeError("Tracer.init() must be called before trace().")
        if isinstance(message, TraceMessage):
            message = message.model_dump()

        sanitized = self._sanitize_images_in_obj(message)
        content = json.dumps(sanitized, ensure_ascii=False)
        self._fout.write(content + "\n")
        self._fout.flush()

    def _save_data_image(self, mime: str, b64: str) -> str:
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
        }.get(mime, "img")

        self._img_idx += 1
        name = f"llm_{self._img_idx:04d}_{uuid.uuid4().hex[:8]}.{ext}"
        abs_path = os.path.join(self._images_dir, name)
        with open(abs_path, "wb") as f:
            f.write(base64.b64decode(b64))

        # Return a relative path (relative to the trace file directory).
        trace_dir = os.path.dirname(self._file_path) if self._file_path else self._data_dir
        trace_dir = trace_dir or "."
        return os.path.relpath(abs_path, start=trace_dir)

    def _sanitize_images_in_obj(self, obj):
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                if isinstance(v, str):
                    m = _DATA_IMAGE_RE.match(v)
                    if m:
                        mime, b64 = m.group(1), m.group(2)
                        try:
                            new[k] = self._save_data_image(mime, b64)
                            continue
                        except Exception:
                            # Best-effort: if writing fails, keep the original field and avoid breaking the main flow.
                            new[k] = v
                            continue
                new[k] = self._sanitize_images_in_obj(v)
            return new
        if isinstance(obj, list):
            return [self._sanitize_images_in_obj(x) for x in obj]
        return obj
