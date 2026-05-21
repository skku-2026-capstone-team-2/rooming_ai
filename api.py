"""
Rooming AI 서버
===============
실행: uvicorn api:app --port 8001

POST /recommend
  요청: 자연어 쿼리 + 선호 조건 + 자주 가는 장소
  응답: 추천 매물 property_id + GPT 설명
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from search import search_stream, explain_article

app = FastAPI(title="Rooming AI 서버", version="1.0.0")


# ── 요청 스키마 ──────────────────────────────────────────────────

class UserPlace(BaseModel):
    name: str           = Field(..., description="장소명")
    lat:  Optional[float] = Field(None, description="위도")
    lng:  Optional[float] = Field(None, description="경도")


class RecommendRequest(BaseModel):
    query:       str            = Field(..., description="자연어 검색 쿼리")
    preferences: list[str]      = Field([], description="선호 조건 레이블 목록")
    target_places: list[UserPlace] = Field([], description="자주 가는 장소 목록")
    top_n:       int            = Field(3, ge=1, le=5, description="추천 매물 수")


# ── 응답 스키마 ──────────────────────────────────────────────────

class PropertyResult(BaseModel):
    property_id: int       = Field(..., description="매물 PK")
    infra_ids:   list[int] = Field([], description="매물 주변 인프라 PK 목록")
    explanation: str       = Field(..., description="GPT 추천 설명")


class RecommendResponse(BaseModel):
    success: bool
    message: str
    results: list[PropertyResult]


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse, tags=["AI 추천"])
def recommend(req: RecommendRequest):
    frequent_dicts = [{"name": p.name, "lat": p.lat, "lng": p.lng} for p in req.target_places]
    frequent_names = [p.name for p in req.target_places]

    results_raw: list[dict] = []
    error_msg: Optional[str] = None

    for step, status, message, data in search_stream(
        req.query, req.top_n, frequent_dicts, req.preferences
    ):
        if status == "error":
            error_msg = message
            break
        if step == 7 and status == "done" and data:
            results_raw = data

    if error_msg:
        return RecommendResponse(success=False, message=error_msg, results=[])

    with ThreadPoolExecutor(max_workers=len(results_raw) or 1) as ex:
        explanations = list(ex.map(
            lambda a: explain_article(a, req.query, frequent_names),
            results_raw,
        ))

    results = [
        PropertyResult(property_id=a["id"], infra_ids=a.get("infra_ids", []), explanation=exp)
        for a, exp in zip(results_raw, explanations)
    ]

    return RecommendResponse(
        success=True,
        message=f"TOP {len(results)} 추천 완료",
        results=results,
    )
