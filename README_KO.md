# Universal File Reader MCP

<p align="right">
  <a href="README.md">English</a>
</p>

Universal File Reader는 PDF, CSV, 이미지 파일에서 정보를 추출하는 MCP 서버입니다. 서버는 파일 형식에 맞는 프로세서를 자동으로 선택하며 필요 시 OCR을 사용합니다.

## 설치

```bash
pip install -e .
```

## 서버 실행

```bash
universal-file-reader
```

### SSH로 원격 실행하기

다른 서버에서 실행하려면 SSH 명령을 사용할 수 있습니다.

```bash
ssh user@remote-host universal-file-reader
```

Google ADK에서는 위 명령을 `ToolSubprocess`에 지정해 원격 서버와
통신할 수 있습니다.

서버는 `read_file`, `get_supported_formats`, `validate_file` 세 가지 도구를 제공합니다. 자세한 입력 형식은 `src/document_reader/mcp_server.py`를 참고하세요.

## API 서버 실행

REST API를 사용하려면 다음 명령으로 서버를 실행할 수 있습니다.

```bash
universal-file-reader-api
```

도커 컴포즈 사용 시:

```bash
docker-compose up
```

API는 <http://localhost:8000> 에서 접근할 수 있습니다.

### MCP 메시지 형식

`POST /mcp` 엔드포인트는 다음과 같은 JSON을 받습니다.

```json
{
  "tool": "read_file",
  "arguments": {
    "file_path": "/path/to/file.pdf",
    "output_format": "markdown"
  }
}
```

## 환경 변수

- `MODEL_NAME` – 사용 모델 이름 (예: `gpt-4o`, `gemini-2.0-flash`)
- `MODEL_API_KEY` – 선택한 모델의 API 키
- `MAX_PAGE_PER_PROCESS` – PDF OCR 시 한 번에 처리할 최대 페이지 수
- `OCR_TIMEOUT_SECONDS` – PDF 페이지당 타임아웃(기본 30초)
- `TIMEOUT_SECONDS` – 전역 처리 타임아웃
- `EXTRACT_IMAGES` – `true`로 설정하면 PDF의 이미지도 OCR 처리하여 결과에 포함합니다.

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
