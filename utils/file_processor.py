"""
파일 처리 유틸리티
"""
import os
import tempfile
from typing import Optional, Tuple
from config.settings import get_config
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import streamlit as st

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ORIGINAL_CONTAINER_NAME = os.getenv("AZURE_STORAGE_ORIGINAL_CONTAINER_NAME")
AZURE_STORAGE_ENHANCED_CONTAINER_NAME = os.getenv("AZURE_STORAGE_ENHANCED_CONTAINER_NAME")

class FileProcessor:
    """파일 처리 클래스"""

    def __init__(self):
        self.config = get_config("document")
        self.supported_formats = self.config["supported_formats"]
        self.max_file_size = self.config["max_file_size"]

    def validate_file(self, file) -> Tuple[bool, Optional[str]]:
        """파일 유효성 검사"""
        if file is None:
            return False, "파일이 선택되지 않았습니다."

        # 파일 크기 검사
        if file.size > self.max_file_size:
            return False, f"파일 크기가 너무 큽니다. (최대: {self.max_file_size // (1024*1024)}MB)"

        # 파일 형식 검사
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in self.supported_formats:
            return False, f"지원하지 않는 파일 형식입니다. (지원 형식: {', '.join(self.supported_formats)})"

        return True, None

    def save_temp_file(self, file) -> str:
        """임시 파일로 저장"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getvalue())
            return temp_file.name

    def get_file_info(self, file) -> dict:
        """파일 정보 반환"""
        return {
            "name": file.name,
            "size": file.size,
            "type": file.type,
            "extension": os.path.splitext(file.name)[1].lower()
        }

    def upload_file(self, file, upload_type: str) -> str:
        """파일 업로드"""

        if file is not None:
            
            try:
                blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

                azure_storage_container = AZURE_STORAGE_ORIGINAL_CONTAINER_NAME if upload_type == "original" else AZURE_STORAGE_ENHANCED_CONTAINER_NAME

                container_client = blob_service_client.get_container_client(container=azure_storage_container) 

                blob_client = container_client.get_blob_client(file.name)

                blob_client.upload_blob(file, overwrite=True)

                # 업로드된 파일의 URL 생성
                blob_url = blob_client.url
                
                st.success(f"{file.name} 파일이 {azure_storage_container} 컨테이너에 업로드되었습니다.")
                
                return blob_url
            
            except Exception as e:
                st.error(f"파일 업로드 실패: {str(e)}")
                return None
        
        return None