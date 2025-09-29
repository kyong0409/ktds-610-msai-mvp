"""
게시판 서비스 로직
"""
import streamlit as st
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
from io import StringIO

class BoardService:
    """게시판 서비스 클래스"""

    def __init__(self):
        pass

    def get_all_posts(self) -> List[Dict]:
        """모든 게시글 조회"""
        return st.session_state.get('board_posts', [])

    def apply_filters(self, posts: List[Dict], filter_options: Dict) -> List[Dict]:
        """필터 옵션 적용"""
        if not posts:
            return []

        filtered_posts = posts.copy()

        # 품질 점수 필터
        quality_min = filter_options.get('quality_min', 0)
        if quality_min > 0:
            filtered_posts = [
                post for post in filtered_posts
                if post.get('quality_score', 0) >= quality_min
            ]

        # 검색어 필터
        search_term = filter_options.get('search', '').lower().strip()
        if search_term:
            filtered_posts = [
                post for post in filtered_posts
                if (search_term in post.get('title', '').lower() or
                    search_term in post.get('content', '').lower() or
                    search_term in post.get('author', '').lower())
            ]

        # 날짜 범위 필터
        date_range = filter_options.get('date_range')
        if date_range and date_range != "전체":
            cutoff_date = self._get_cutoff_date(date_range)
            filtered_posts = [
                post for post in filtered_posts
                if post.get('timestamp', datetime.now()) >= cutoff_date
            ]

        # 작성자 필터
        author_filter = filter_options.get('author')
        if author_filter and author_filter != "전체":
            filtered_posts = [
                post for post in filtered_posts
                if post.get('author') == author_filter
            ]

        # 정렬 적용
        sort_option = filter_options.get('sort', '최신순')
        filtered_posts = self._apply_sort(filtered_posts, sort_option)

        return filtered_posts

    def _get_cutoff_date(self, date_range: str) -> datetime:
        """날짜 범위에 따른 기준 날짜 계산"""
        now = datetime.now()
        if date_range == "오늘":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_range == "1주일":
            return now - timedelta(days=7)
        elif date_range == "1개월":
            return now - timedelta(days=30)
        elif date_range == "3개월":
            return now - timedelta(days=90)
        else:
            return datetime.min

    def _apply_sort(self, posts: List[Dict], sort_option: str) -> List[Dict]:
        """정렬 적용"""
        if sort_option == "최신순":
            return sorted(posts, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
        elif sort_option == "조회수순":
            return sorted(posts, key=lambda x: x.get('views', 0), reverse=True)
        elif sort_option == "품질점수순":
            return sorted(posts, key=lambda x: x.get('quality_score', 0), reverse=True)
        elif sort_option == "제목순":
            return sorted(posts, key=lambda x: x.get('title', ''))
        else:
            return posts

    def increment_view_count(self, post_index: int):
        """조회수 증가"""
        board_posts = st.session_state.get('board_posts', [])
        if 0 <= post_index < len(board_posts):
            board_posts[post_index]['views'] = board_posts[post_index].get('views', 0) + 1

    def delete_post(self, post_index: int) -> bool:
        """게시글 삭제"""
        board_posts = st.session_state.get('board_posts', [])
        if 0 <= post_index < len(board_posts):
            del board_posts[post_index]
            return True
        return False

    def add_to_vector_db(self, post: Dict) -> Dict:
        """게시글을 VectorDB에 추가"""
        if 'vector_db' not in st.session_state:
            st.session_state.vector_db = []

        # 중복 확인
        existing = any(
            doc.get('content') == post.get('content')
            for doc in st.session_state.vector_db
        )

        if existing:
            return {
                "success": False,
                "message": "이미 VectorDB에 저장된 내용입니다."
            }

        # VectorDB에 추가
        vector_entry = {
            "content": post.get('content', ''),
            "filename": post.get('title', 'Board Post'),
            "metadata": {
                "source": "board",
                "author": post.get('author', 'Unknown'),
                "original_timestamp": post.get('timestamp', datetime.now()),
                "views": post.get('views', 0)
            },
            "quality_score": post.get('quality_score', 0),
            "timestamp": datetime.now()
        }

        st.session_state.vector_db.append(vector_entry)

        return {
            "success": True,
            "message": "VectorDB에 추가되었습니다!",
            "count": len(st.session_state.vector_db)
        }

    def get_board_stats(self) -> Dict:
        """게시판 통계"""
        board_posts = st.session_state.get('board_posts', [])

        if not board_posts:
            return {
                'total_posts': 0,
                'total_views': 0,
                'avg_quality': 0,
                'recent_posts': 0
            }

        total_views = sum(post.get('views', 0) for post in board_posts)
        total_quality = sum(post.get('quality_score', 0) for post in board_posts)
        avg_quality = total_quality / len(board_posts) if board_posts else 0

        # 최근 7일 게시글
        week_ago = datetime.now() - timedelta(days=7)
        recent_posts = sum(
            1 for post in board_posts
            if post.get('timestamp', datetime.now()) >= week_ago
        )

        return {
            'total_posts': len(board_posts),
            'total_views': total_views,
            'avg_quality': avg_quality,
            'recent_posts': recent_posts
        }

    def delete_zero_view_posts(self) -> Dict:
        """조회수 0인 게시글 삭제"""
        board_posts = st.session_state.get('board_posts', [])
        original_count = len(board_posts)

        # 조회수가 0이 아닌 게시글만 남김
        st.session_state.board_posts = [
            post for post in board_posts
            if post.get('views', 0) > 0
        ]

        deleted_count = original_count - len(st.session_state.board_posts)

        return {
            'count': deleted_count,
            'remaining': len(st.session_state.board_posts)
        }

    def recalculate_quality_scores(self) -> Dict:
        """품질 점수 재계산"""
        board_posts = st.session_state.get('board_posts', [])
        updated_count = 0

        for post in board_posts:
            # 간단한 품질 점수 재계산 로직
            content_length = len(post.get('content', ''))
            has_structure = any(keyword in post.get('content', '') for keyword in ['#', '##', '목차', '결론'])
            view_bonus = min(post.get('views', 0) * 2, 20)  # 조회수 보너스 (최대 20점)

            # 기본 점수 계산
            base_score = 60
            if content_length > 1000:
                base_score += 15
            if content_length > 2000:
                base_score += 10
            if has_structure:
                base_score += 10

            new_score = min(base_score + view_bonus, 100)

            # 점수가 변경된 경우만 업데이트
            if post.get('quality_score', 0) != new_score:
                post['quality_score'] = new_score
                updated_count += 1

        return {
            'updated': updated_count,
            'total': len(board_posts)
        }

    def export_posts(self, format_type: str) -> str:
        """게시글 내보내기"""
        board_posts = st.session_state.get('board_posts', [])

        if format_type == "텍스트":
            return self._export_as_text(board_posts)
        elif format_type == "csv":
            return self._export_as_csv(board_posts)
        elif format_type == "json":
            return self._export_as_json(board_posts)
        else:
            return "지원하지 않는 형식입니다."

    def _export_as_text(self, posts: List[Dict]) -> str:
        """텍스트 형식으로 내보내기"""
        text_content = f"# 지식 게시판 내보내기\n"
        text_content += f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text_content += f"총 게시글: {len(posts)}개\n\n"
        text_content += "=" * 50 + "\n\n"

        for i, post in enumerate(posts, 1):
            text_content += f"## {i}. {post.get('title', 'Unknown')}\n\n"
            text_content += f"**작성자:** {post.get('author', 'Unknown')}\n"
            text_content += f"**등록일:** {post.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}\n"
            text_content += f"**조회수:** {post.get('views', 0)}회\n"
            text_content += f"**품질점수:** {post.get('quality_score', 0)}점\n\n"
            text_content += f"**내용:**\n{post.get('content', '')}\n\n"
            text_content += "-" * 50 + "\n\n"

        return text_content

    def _export_as_csv(self, posts: List[Dict]) -> str:
        """CSV 형식으로 내보내기"""
        output = StringIO()
        writer = csv.writer(output)

        # 헤더 작성
        writer.writerow(['제목', '작성자', '등록일', '조회수', '품질점수', '내용'])

        # 데이터 작성
        for post in posts:
            writer.writerow([
                post.get('title', ''),
                post.get('author', ''),
                post.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                post.get('views', 0),
                post.get('quality_score', 0),
                post.get('content', '')[:500] + "..." if len(post.get('content', '')) > 500 else post.get('content', '')
            ])

        return output.getvalue()

    def _export_as_json(self, posts: List[Dict]) -> str:
        """JSON 형식으로 내보내기"""
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_posts": len(posts),
            "posts": []
        }

        for post in posts:
            export_post = {
                "title": post.get('title', ''),
                "author": post.get('author', ''),
                "timestamp": post.get('timestamp', datetime.now()).isoformat(),
                "views": post.get('views', 0),
                "quality_score": post.get('quality_score', 0),
                "content": post.get('content', '')
            }
            export_data["posts"].append(export_post)

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def advanced_search(self, query: str, search_in: List[str]) -> List[Dict]:
        """고급 검색"""
        board_posts = st.session_state.get('board_posts', [])
        results = []

        query_lower = query.lower()

        for post in board_posts:
            match_score = 0

            if "제목" in search_in and query_lower in post.get('title', '').lower():
                match_score += 3

            if "내용" in search_in and query_lower in post.get('content', '').lower():
                match_score += 2

            if "작성자" in search_in and query_lower in post.get('author', '').lower():
                match_score += 1

            if match_score > 0:
                post_with_score = post.copy()
                post_with_score['match_score'] = match_score
                results.append(post_with_score)

        # 일치 점수순으로 정렬
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results

    def get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def get_unique_authors(self) -> List[str]:
        """게시글 작성자 목록"""
        board_posts = st.session_state.get('board_posts', [])
        authors = set(post.get('author', 'Unknown') for post in board_posts)
        return sorted(list(authors))