FROM python:3.11-slim

# 시스템 패키지 및 Node.js 설치
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY pyproject.toml .

# MCP 서버 설치
RUN pip install -e .

# MCP 서버 실행
CMD ["npx", "@modelcontextprotocol/inspector", "universal-file-reader"]