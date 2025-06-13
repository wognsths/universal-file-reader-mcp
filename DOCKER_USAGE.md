# Docker로 MCP 서버 실행하기

## 장점
- Windows에서 발생하는 signal, NumPy 호환성 문제 해결
- 일관된 리눅스 환경에서 실행
- 깔끔한 stdio 통신

## 사전 요구사항
- Docker Desktop (Windows) 또는 Docker Engine (Linux/Mac)
- Google API 키 (OCR 기능 사용 시)

## 빌드 및 실행

### 1. Docker 이미지 빌드
```bash
docker build -t universal-file-reader-mcp .
```

### 2. Docker로 직접 실행
```bash
# Windows PowerShell
docker run -it `
  -v ${PWD}/test_files:/app/test_files `
  -v ${PWD}/output:/app/output `
  -e GOOGLE_API_KEY=$env:GOOGLE_API_KEY `
  universal-file-reader-mcp

# Linux/Mac
docker run -it \
  -v $(pwd)/test_files:/app/test_files \
  -v $(pwd)/output:/app/output \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  universal-file-reader-mcp
```

### 3. Docker Compose로 실행
```bash
# .env 파일 생성 (OCR 사용 시)
echo "GOOGLE_API_KEY=your-api-key-here" > .env

# 실행
docker-compose up --build
```

## 파일 처리 예시

### 디렉토리 구조
```
project/
├── test_files/      # 처리할 파일들
│   ├── document.pdf
│   ├── data.csv
│   └── image.png
└── output/          # 처리 결과
```

### MCP 클라이언트에서 사용
```python
# 파일 경로는 컨테이너 내부 경로 사용
result = await session.call_tool("read_file", {
    "file_path": "/app/test_files/document.pdf",
    "output_format": "markdown"
})
```

## 문제 해결

### 권한 문제
```bash
# Linux에서 output 디렉토리 권한 설정
chmod 777 output/
```

### 로그 확인
```bash
docker logs <container-id>
```

## 성능 최적화
- 대용량 파일은 호스트에서 직접 실행하는 것이 더 빠를 수 있음
- Docker Desktop의 리소스 할당을 늘려서 성능 개선 가능 