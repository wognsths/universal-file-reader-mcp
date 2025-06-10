# Universal File Reader MCP

Universal File Reader는 PDF, CSV, 이미지 파일에서 정보를 추출하는 MCP 서버입니다. 서버는 파일 형식에 맞는 프로세서를 자동으로 선택하며 필요 시 OCR을 사용합니다.

## 설치

```bash
pip install -e .
```

## 서버 실행

```bash
universal-file-reader
```

서버는 `read_file`, `get_supported_formats`, `validate_file` 세 가지 도구를 제공합니다. 자세한 입력 형식은 `src/document_reader/mcp_server.py`를 참고하세요.

## 환경 변수

- `GOOGLE_API_KEY` – Gemini 기반 OCR 사용을 위한 API 키

## 테스트 실행

의존성을 설치한 뒤 다음 명령으로 테스트를 실행합니다.

```bash
pip install -r requirements.txt
pytest
```

## 개발

```bash
pip install -e .[development]
ruff check
```
