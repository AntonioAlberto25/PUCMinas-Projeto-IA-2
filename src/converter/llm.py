from __future__ import annotations

import json
import re
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


class LLMError(Exception):
    """Base exception for LLM integration errors."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM credentials are invalid or unauthorized."""


class LLMRequestError(LLMError):
    """Raised when the LLM endpoint fails for non-auth reasons."""


class LLMClient(Protocol):
    def convert_chunk(
        self,
        *,
        source_language: str,
        target_language: str,
        chunk_id: str,
        chunk_content: str,
        accumulated_context: str,
        enterprise_context: str,
        style_instructions: dict[str, object],
    ) -> str:
        ...


class OpenAICompatibleLLMClient:
    def __init__(self, api_url: str, api_key: str, model: str, timeout_seconds: int = 60) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def convert_chunk(
        self,
        *,
        source_language: str,
        target_language: str,
        chunk_id: str,
        chunk_content: str,
        accumulated_context: str,
        enterprise_context: str,
        style_instructions: dict[str, object],
    ) -> str:
        system_prompt = (
            "You are a senior software engineer specialized in source-to-source translation. "
            "Convert code faithfully to the target language while preserving behavior. "
            "Return only raw converted code. "
            "Do not include markdown fences, prose, explanations, headers, or notes."
        )

        user_prompt = {
            "task": "convert_chunk",
            "source_language": source_language,
            "target_language": target_language,
            "chunk_id": chunk_id,
            "style_instructions": style_instructions,
            "accumulated_context": accumulated_context,
            "enterprise_context": enterprise_context,
            "chunk": chunk_content,
            "output_requirements": [
                "Only output converted code for this chunk",
                "Do not output explanations or comments unless they are part of target code semantics",
                "Keep behavior equivalent",
                "Follow architecture and coding conventions from enterprise_context when available",
                "Respect target language idioms and naming",
            ],
        }

        if self._is_google_native_endpoint(self.api_url):
            response = self._request_google_generate_content(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            content = self._extract_google_content(response)
        else:
            response = self._request_openai_compatible(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            content = self._extract_openai_content(response)

        result = content.strip()
        if not result:
            raise LLMRequestError("LLM returned empty content.")
        return result

    @staticmethod
    def _is_google_native_endpoint(api_url: str) -> bool:
        return "generativelanguage.googleapis.com" in api_url and "generateContent" in api_url

    def _request_openai_compatible(self, *, system_prompt: str, user_prompt: dict[str, object]) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
            "temperature": 0.1,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        body = self._raise_for_status_and_json(response)
        if not isinstance(body, dict):
            raise LLMRequestError("LLM response format is invalid or missing content.")
        return body

    def _request_google_generate_content(
        self,
        *,
        system_prompt: str,
        user_prompt: dict[str, object],
    ) -> dict[str, object]:
        prompt_text = json.dumps(user_prompt, ensure_ascii=True)
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "generationConfig": {"temperature": 0.1},
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._attach_google_api_key(self.api_url, self.api_key),
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        body = self._raise_for_status_and_json(response)
        if not isinstance(body, dict):
            raise LLMRequestError("LLM response format is invalid or missing content.")
        return body

    @staticmethod
    def _attach_google_api_key(api_url: str, api_key: str) -> str:
        parsed = urlsplit(api_url)
        query_pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query_pairs.setdefault("key", api_key)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query_pairs), parsed.fragment))

    def _raise_for_status_and_json(self, response: requests.Response) -> object:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = response.status_code
            response_text = response.text[:800]
            if status_code in {401, 403}:
                raise LLMAuthenticationError(
                    f"LLM authentication failed (status {status_code}). "
                    "Check LLM_API_KEY, LLM_API_URL and model access. "
                    f"Provider response: {response_text}"
                ) from exc
            raise LLMRequestError(
                f"LLM request failed (status {status_code}). Provider response: {response_text}"
            ) from exc
        except requests.RequestException as exc:
            raise LLMRequestError(f"Network error while calling LLM endpoint: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise LLMRequestError("LLM response format is invalid or missing content.") from exc

    @staticmethod
    def _extract_openai_content(body: dict[str, object]) -> str:
        try:
            choices = body["choices"]
            first_choice = choices[0]
            message = first_choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("LLM response format is invalid or missing content.") from exc

        return str(content)

    @staticmethod
    def _extract_google_content(body: dict[str, object]) -> str:
        try:
            candidates = body["candidates"]
            first_candidate = candidates[0]
            content = first_candidate["content"]
            parts = content["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("LLM response format is invalid or missing content.") from exc

        text_parts = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])

        merged = "\n".join(text_parts).strip()
        if not merged:
            raise LLMRequestError("LLM response format is invalid or missing content.")
        return merged


class MockLLMClient:
    def convert_chunk(
        self,
        *,
        source_language: str,
        target_language: str,
        chunk_id: str,
        chunk_content: str,
        accumulated_context: str,
        enterprise_context: str,
        style_instructions: dict[str, object],
    ) -> str:
        converted = mock_convert_code(
            source_language=source_language,
            target_language=target_language,
            code=chunk_content,
        )
        return (
            f"// MOCK CONVERSION {chunk_id}: {source_language} -> {target_language}\n"
            f"// style={style_instructions}\n"
            f"// NOTE: This is a deterministic mock conversion for local testing.\n"
            + converted
        )


def mock_convert_code(*, source_language: str, target_language: str, code: str) -> str:
    source = source_language.lower().strip()
    target = target_language.lower().strip()

    if source == "python" and target in {"javascript", "typescript"}:
        return python_to_braces_style(code, function_keyword="function")

    if source == "python" and target == "java":
        body = python_to_braces_style(code, function_keyword="public static Object")
        return "public class ConvertedSnippet {\n" + indent_block(body, "    ") + "\n}"

    if source == "javascript" and target == "python":
        return javascript_to_python_style(code)

    # Fallback that still changes output so local mock behavior is observable.
    return code.replace("\t", "    ") + "\n# mock: no dedicated rule for this language pair"


def python_to_braces_style(code: str, function_keyword: str) -> str:
    lines = code.splitlines()
    result: list[str] = []
    indent_stack = [0]

    for raw in lines:
        if not raw.strip():
            result.append("")
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        while indent < indent_stack[-1]:
            indent_stack.pop()
            result.append(" " * indent_stack[-1] + "}")

        if stripped.startswith("def ") and stripped.endswith(":"):
            match = re.match(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\):", stripped)
            if match:
                name = match.group(1)
                args = match.group(2)
                if function_keyword == "public static Object":
                    signature = f"public static Object {name}({format_java_args(args)}) {{"
                else:
                    signature = f"function {name}({args}) {{"
                result.append(" " * indent + signature)
                indent_stack.append(indent + 4)
                continue

        if stripped.startswith("for ") and stripped.endswith(":"):
            match = re.match(r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.+):", stripped)
            if match:
                var = match.group(1)
                iterable = py_expr_to_js(match.group(2))
                result.append(" " * indent + f"for (const {var} of {iterable}) {{")
                indent_stack.append(indent + 4)
                continue

        if stripped.startswith("if ") and stripped.endswith(":"):
            cond = py_expr_to_js(stripped[3:-1].strip())
            result.append(" " * indent + f"if ({cond}) {{")
            indent_stack.append(indent + 4)
            continue

        if stripped.startswith("elif ") and stripped.endswith(":"):
            cond = py_expr_to_js(stripped[5:-1].strip())
            # Close previous block at same indentation before else-if.
            if indent_stack and indent_stack[-1] == indent + 4:
                indent_stack.pop()
                result.append(" " * indent + "} else if (" + cond + ") {")
                indent_stack.append(indent + 4)
            else:
                result.append(" " * indent + f"else if ({cond}) {{")
                indent_stack.append(indent + 4)
            continue

        if stripped == "else:":
            if indent_stack and indent_stack[-1] == indent + 4:
                indent_stack.pop()
                result.append(" " * indent + "} else {")
                indent_stack.append(indent + 4)
            else:
                result.append(" " * indent + "else {")
                indent_stack.append(indent + 4)
            continue

        if stripped.startswith("return "):
            ret = py_expr_to_js(stripped[7:].strip())
            result.append(" " * indent + f"return {ret};")
            continue

        if stripped.startswith("#"):
            result.append(" " * indent + "//" + stripped[1:])
            continue

        transformed = py_expr_to_js(stripped)
        if not transformed.endswith(";") and not transformed.endswith("{") and not transformed.endswith("}"):
            transformed += ";"
        result.append(" " * indent + transformed)

    while len(indent_stack) > 1:
        indent_stack.pop()
        result.append(" " * indent_stack[-1] + "}")

    return "\n".join(result)


def javascript_to_python_style(code: str) -> str:
    result: list[str] = []
    indent = 0
    lines = code.splitlines()

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            result.append("")
            continue

        if stripped == "}":
            indent = max(0, indent - 4)
            continue

        if stripped.endswith("{"):
            header = stripped[:-1].strip()
            header = header.replace("function ", "def ")
            if header.startswith("if (") and header.endswith(")"):
                header = "if " + header[4:-1]
            elif header.startswith("else if (") and header.endswith(")"):
                header = "elif " + header[9:-1]
            elif header == "else":
                header = "else"
            result.append(" " * indent + header + ":")
            indent += 4
            continue

        line = stripped.rstrip(";")
        line = line.replace("true", "True").replace("false", "False").replace("null", "None")
        result.append(" " * indent + line)

    return "\n".join(result)


def py_expr_to_js(expr: str) -> str:
    return (
        expr.replace(" and ", " && ")
        .replace(" or ", " || ")
        .replace(" not ", " !")
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )


def format_java_args(args: str) -> str:
    if not args.strip():
        return ""
    pieces = [piece.strip() for piece in args.split(",") if piece.strip()]
    return ", ".join(f"Object {piece}" for piece in pieces)


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else "" for line in text.splitlines())
