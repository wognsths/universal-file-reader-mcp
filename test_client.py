#!/usr/bin/env python3
"""
MCP 서버 테스트를 위한 간단한 클라이언트
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """MCP 서버 테스트"""
    
    server_params = StdioServerParameters(
        command="universal-file-reader",
        args=[]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 서버 초기화
            await session.initialize()
            
            print("=== MCP 서버 연결 성공! ===\n")
            
            # 1. 사용 가능한 도구 목록 확인
            print("1. 사용 가능한 도구들:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"   - {tool.name}: {tool.description}")
            print()
            
            # 2. 지원되는 파일 형식 확인
            print("2. 지원되는 파일 형식 확인:")
            try:
                result = await session.call_tool("get_supported_formats", {})
                print(result.content[0].text)
                print()
            except Exception as e:
                print(f"   오류: {e}\n")
            
            # 3. 파일 읽기 테스트 (샘플 파일이 있는 경우)
            test_files = [  # 텍스트 파일
                "test.pdf",   # PDF 파일 (존재한다면)
                "test.csv",   # CSV 파일 (존재한다면)
            ]
            
            print("3. 파일 읽기 테스트:")
            for file_path in test_files:
                try:
                    print(f"   테스트 파일: {file_path}")
                    result = await session.call_tool("read_file", {
                        "file_path": file_path,
                        "output_format": "markdown"
                    })
                    
                    content = result.content[0].text
                    if len(content) > 500:
                        content = content[:500] + "..."
                    
                    print(f"   결과 (첫 500자): {content}\n")
                    
                except Exception as e:
                    print(f"   {file_path} 처리 실패: {e}\n")
            
            # 4. 파일 검증 테스트
            print("4. 파일 검증 테스트:")
            for file_path in ["README.md"]:
                try:
                    print(f"   검증 파일: {file_path}")
                    result = await session.call_tool("validate_file", {
                        "file_path": file_path
                    })
                    print(f"   결과: {result.content[0].text}\n")
                except Exception as e:
                    print(f"   {file_path} 검증 실패: {e}\n")

if __name__ == "__main__":
    print("MCP 서버 테스트 클라이언트 시작...\n")
    asyncio.run(test_mcp_server())
    print("테스트 완료!") 