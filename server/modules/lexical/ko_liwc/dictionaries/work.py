"""Korean work-related word dictionary.

Based on LIWC Work/Achievement category.
Words related to employment, work, and professional achievement.
"""

from typing import Set

# Work-related words (업무/직장 관련)
WORK: Set[str] = {
    # Employment
    "일", "일하다", "업무",
    "직장", "회사", "기업",
    "직업", "직위", "직책",
    "고용", "취업", "취직",
    "사원", "직원", "동료",

    # Job roles
    "상사", "부하", "팀원",
    "팀장", "부장", "과장", "대리",
    "매니저", "리더", "책임자",

    # Work activities
    "근무", "근무하다", "출근", "퇴근",
    "야근", "업무처리",
    "협업", "협력", "협조",
    "회의", "미팅", "보고",

    # Projects/tasks
    "프로젝트", "과제", "업무",
    "태스크", "목표", "마감",
    "일정", "스케줄", "계획",

    # Achievement
    "성과", "실적", "결과",
    "달성", "달성하다",
    "성취", "성취하다",
    "완료", "완성", "마무리",

    # Career
    "경력", "경험", "이력",
    "승진", "진급",
    "능력", "역량", "스킬",
    "전문", "전문성", "전문가",

    # Business
    "비즈니스", "사업",
    "거래", "계약", "협상",
    "클라이언트", "고객",

    # Industry terms
    "산업", "분야", "섹터",
    "시장", "경쟁",

    # Education/training (work-related)
    "교육", "훈련", "연수",
    "자격증", "인증",
}
