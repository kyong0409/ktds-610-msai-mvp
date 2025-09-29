"""
파일 처리 유틸리티
"""
import os
import tempfile
from typing import Optional, Tuple
from config.settings import get_config

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

    def extract_text(self, file) -> str:
        """파일에서 텍스트 추출"""
        file_ext = os.path.splitext(file.name)[1].lower()

        try:
            if file_ext == ".txt":
                return str(file.read(), "utf-8")
            elif file_ext == ".pdf":
                return self._extract_from_pdf(file)
            elif file_ext == ".docx":
                return self._extract_from_docx(file)
            elif file_ext == ".pptx":
                return self._extract_from_pptx(file)
            else:
                return f"[{file.name}] 파일 내용 추출 시뮬레이션"
        except Exception as e:
            return f"파일 읽기 오류: {str(e)}"

    def _extract_from_pdf(self, file) -> str:
        """PDF 파일에서 텍스트 추출"""
        # TODO: PyPDF2 또는 pdfplumber 사용
        # import PyPDF2
        # reader = PyPDF2.PdfReader(file)
        # text = ""
        # for page in reader.pages:
        #     text += page.extract_text()
        # return text

        return f"[PDF 시뮬레이션] {file.name} 파일의 내용입니다."

    def _extract_from_docx(self, file) -> str:
        """DOCX 파일에서 텍스트 추출"""
        # TODO: python-docx 사용
        # import docx
        # doc = docx.Document(file)
        # text = ""
        # for paragraph in doc.paragraphs:
        #     text += paragraph.text + "\n"
        # return text

        return f"[DOCX 시뮬레이션] {file.name} 파일의 내용입니다."

    def _extract_from_pptx(self, file) -> str:
        """PPTX 파일에서 텍스트 추출"""
        # TODO: python-pptx 사용
        # import pptx
        # presentation = pptx.Presentation(file)
        # text = ""
        # for slide in presentation.slides:
        #     for shape in slide.shapes:
        #         if hasattr(shape, "text"):
        #             text += shape.text + "\n"
        # return text

        return f"[PPTX 시뮬레이션] {file.name} 파일의 내용입니다."

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