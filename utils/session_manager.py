"""
Streamlit 세션 상태 관리 유틸리티
"""
import streamlit as st
from typing import Any, Dict, List

class SessionManager:
    """세션 상태 관리 클래스"""

    @staticmethod
    def initialize():
        """세션 상태 초기화"""
        default_states = {
            'chat_history': [],
            'board_posts': [],
            'vector_db': [],
            'current_analysis': {},
            'user_settings': {
                'theme': 'light',
                'language': 'ko'
            }
        }

        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """세션 상태 값 조회"""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        """세션 상태 값 설정"""
        st.session_state[key] = value

    @staticmethod
    def append(key: str, value: Any):
        """리스트 타입 세션 상태에 값 추가"""
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key].append(value)

    @staticmethod
    def clear(key: str):
        """특정 세션 상태 초기화"""
        if key in st.session_state:
            if isinstance(st.session_state[key], list):
                st.session_state[key] = []
            elif isinstance(st.session_state[key], dict):
                st.session_state[key] = {}
            else:
                st.session_state[key] = None

    @staticmethod
    def clear_all():
        """모든 세션 상태 초기화"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        SessionManager.initialize()

    @staticmethod
    def get_stats() -> Dict:
        """세션 상태 통계"""
        return {
            'chat_messages': len(st.session_state.get('chat_history', [])),
            'board_posts': len(st.session_state.get('board_posts', [])),
            'vector_documents': len(st.session_state.get('vector_db', [])),
            'has_current_analysis': bool(st.session_state.get('current_analysis')),
            'session_keys': list(st.session_state.keys())
        }