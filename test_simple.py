#!/usr/bin/env python3
"""간단한 비동기 테스트"""

import asyncio
import sys
import os

# NumPy 경고 억제
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'

sys.path.insert(0, 'src')

from document_reader.mcp_server import UniversalFileReaderMCP

async def test_direct():
    print("직접 테스트 시작...")
    
    mcp = UniversalFileReaderMCP()
    await mcp.initialize()
    
    # CSV 파일 테스트
    print("\nCSV 파일 처리 중...")
    result = await mcp.read_file("test.pdf", output_format="markdown")
    print(result)
    
    print(f"성공: {result.get('success')}")
    if result.get('success'):
        print(f"내용 길이: {len(result.get('content', ''))}")
    else:
        print(f"오류: {result.get('error')}")
        print(f"오류 타입: {result.get('error_type')}")

if __name__ == "__main__":
    asyncio.run(test_direct()) 