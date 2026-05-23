"""
Rooming AI 검색 파이프라인 (DB 전용)
======================================
DB에 데이터가 이미 적재된 상태를 전제로 동작합니다.
JSON 폴백 없음 — DB 연결 실패 시 예외를 그대로 올립니다.

.env 필수 항목:
  OPENAI_API_KEY=sk-...
  TMAP_API_KEY=...
  POSTGRES_DB_URL=postgresql://...
"""

import os
import json
import re
import math
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TMAP_API_KEY = os.environ.get("TMAP_API_KEY", "")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _get_conn():
    from db import get_connection
    return get_connection()


# ──────────────────────────────────────────────────────────────
# 선호 조건 매핑
# ──────────────────────────────────────────────────────────────

PREF_HARD = {
    '월세 30만원 이하':     ('max_rent',  30),
    '월세 40만원 이하':     ('max_rent',  40),
    '월세 50만원 이하':     ('max_rent',  50),
    '월세 60만원 이하':     ('max_rent',  60),
    '월세 70만원 이하':     ('max_rent',  70),
    '보증금 500만원 이하':   ('max_price', 500),
    '보증금 1,000만원 이하': ('max_price', 1000),
    '보증금 2,000만원 이하': ('max_price', 2000),
    '보증금 3,000만원 이하': ('max_price', 3000),
    '보증금 5,000만원 이하': ('max_price', 5000),
}

PREF_INFRA = {
    '편의점 가까움':    'convenience_store',
    '마트/슈퍼 가까움': 'mart',
    '카페 가까움':      'cafe',
    '헬스장 가까움':    'gym',
    '병원/약국 가까움': 'hospital',
    '세탁소 가까움':    'laundry',
}

# Tmap 검색은 한국어 키워드 필요 — DB 영어 코드 → 한국어 변환
INFRA_EN_TO_KR: dict[str, str] = {
    "gym":               "헬스장",
    "convenience_store": "편의점",
    "cafe":              "카페",
    "hospital":          "병원",
    "pharmacy":          "약국",
    "laundry":           "세탁소",
    "mart":              "마트",
}


def apply_prefs(parsed: dict, prefs: list[str]) -> dict:
    """선호 조건을 GPT 파싱 결과에 병합합니다."""
    if not prefs:
        return parsed

    hard       = parsed.setdefault('hard', {}) or {}
    infra      = list(parsed.get('infra') or [])
    soft_parts = [parsed.get('soft') or '']

    for pref in prefs:
        if pref in PREF_HARD:
            key, val = PREF_HARD[pref]
            existing = hard.get(key)
            if existing is None:
                hard[key] = val
            elif key.startswith('max'):
                hard[key] = min(existing, val)
            else:
                hard[key] = max(existing, val)
        elif pref in PREF_INFRA:
            kw = PREF_INFRA[pref]
            if kw not in infra:
                infra.append(kw)
        else:
            soft_parts.append(pref)

    parsed['hard'] = hard
    parsed['infra'] = infra
    parsed['soft']  = ' '.join(filter(None, soft_parts))
    return parsed


# ──────────────────────────────────────────────────────────────
# Step 1. GPT-4o-mini 자연어 파싱
# ──────────────────────────────────────────────────────────────

PARSE_SYSTEM = """
당신은 부동산 검색 조건 파서입니다.
사용자의 자연어 입력을 아래 JSON 스키마로 변환하세요.

{
  "location": "동 이름 (예: 명륜3가). 없으면 null",
  "infra": ["gym", "convenience_store", ...],
  "hard": {
    "trade_type": "월세|전세|단기임대 중 하나. 명시 없으면 null",
    "max_price": <보증금 최대, 만원 단위 정수. 없으면 null>,
    "min_price": <보증금 최소, 만원 단위 정수. 없으면 null>,
    "max_rent":  <월세 최대, 만원 단위 정수. 없으면 null>,
    "min_rent":  <월세 최소, 만원 단위 정수. 없으면 null>,
    "min_area_m2": <최소 면적(m²). 없으면 null>,
    "max_area_m2": <최대 면적(m²). 없으면 null>
  },
  "soft": "원본 쿼리에서 location·hard·infra에 해당하는 표현만 제거한 나머지 원문. 요약하거나 바꾸지 말 것. 제거할 게 없으면 빈 문자열."
}

infra 코드표 (반드시 아래 영어 코드만 사용):
  헬스장/피트니스 → gym
  편의점          → convenience_store
  카페/커피숍     → cafe
  병원/의원       → hospital
  약국            → pharmacy
  세탁소/코인워시 → laundry
  마트/슈퍼       → mart

변환 규칙:
- 금액은 반드시 만원 단위 정수로 변환하세요. 절대 문자열로 출력하지 마세요.
- "1억 8000만원" → 18000,  "2억" → 20000,  "5000만원" → 5000
- "30만원 정도" → max_rent: 35  (±5만원 여유)
- "500만원 이하" → max_price: 500
- soft: location·hard·infra에 해당하는 표현을 원문에서 삭제한 나머지만 남길 것
  · infra 키워드는 그 키워드에 붙은 조사·어미·연결어까지 함께 제거
    예) "편의점이 가깝고" → "가깝고"까지 제거
    예) "헬스장 근처에" → 전부 제거
  · location·hard도 마찬가지로 연결 표현까지 제거
    예) "명륜3가 근처" → 전부 제거,  "보증금 500 이하" → 전부 제거
- 반드시 JSON만 출력하고 설명은 쓰지 마세요.
"""


def parse_query(text: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


# ──────────────────────────────────────────────────────────────
# Step 2. SQL 필터링
# ──────────────────────────────────────────────────────────────

def _to_manwon(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).replace(",", "").replace(" ", "")
    eok_match = re.search(r'(\d+)억', s)
    man_match  = re.search(r'억(\d+)|^(\d+)$', s)
    eok = int(eok_match.group(1)) * 10000 if eok_match else 0
    man = int((man_match.group(1) or man_match.group(2))) if man_match else 0
    return eok + man or None


def _price_filter(rows: list[dict], hard: dict) -> list[dict]:
    pf = {
        "max_price": _to_manwon(hard.get("max_price")),
        "min_price": _to_manwon(hard.get("min_price")),
        "max_rent":  _to_manwon(hard.get("max_rent")),
        "min_rent":  _to_manwon(hard.get("min_rent")),
    }
    def passes(row):
        p = row.get("deposit") or 0
        r = row.get("monthly_rent") or 0
        if pf["max_price"] is not None and p > pf["max_price"]: return False
        if pf["min_price"] is not None and p < pf["min_price"]: return False
        if pf["max_rent"]  is not None and r > pf["max_rent"]:  return False
        if pf["min_rent"]  is not None and r < pf["min_rent"]:  return False
        return True
    return [r for r in rows if passes(r)]


def filter_articles(parsed: dict) -> list[dict]:
    from psycopg2.extras import RealDictCursor

    hard     = parsed.get("hard", {}) or {}
    location = parsed.get("location")

    conditions, params = [], []
    if location:
        conditions.append("address ILIKE %s")
        params.append(f"%{location}%")
    if hard.get("trade_type"):
        conditions.append("trade_type = %s")
        params.append(hard["trade_type"])
    for col, key in [("aream2", "min_area_m2"), ("aream2", "max_area_m2")]:
        val = hard.get(key)
        if val is not None:
            op = ">=" if key.startswith("min") else "<="
            conditions.append(f"{col} {op} %s")
            params.append(val)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT property_id AS id, title, address, room_type,
               trade_type, deposit, monthly_rent, aream2 AS area_m2, floor_info,
               maintenance_fee, has3dmodel,
               description, tags, latitude AS lat, longitude AS lng
        FROM property {where} LIMIT 500;
    """
    conn = _get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return _price_filter(rows, hard)[:100]


# ──────────────────────────────────────────────────────────────
# Step 3. 인프라 근접 점수 (PostGIS)
# ──────────────────────────────────────────────────────────────

INFRA_MAX_WALK_MIN    = 10   # 도보 10분 초과 시 점수 0
FREQUENT_MAX_WALK_MIN = 30   # 도보 30분 초과 시 점수 0


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _apply_infra_score(article: dict, keyword: str, infra_name: str,
                       infra_id: int, walking_min: float) -> None:
    if walking_min <= INFRA_MAX_WALK_MIN:
        article["infra_score"] += 1.0 - (walking_min / INFRA_MAX_WALK_MIN)
    article["infra_found"].append(f"{keyword}({infra_name}, {walking_min:.1f}분)")
    if infra_id not in article["infra_ids"]:
        article["infra_ids"].append(infra_id)


def _fetch_and_save_infra(article: dict, keyword_en: str) -> tuple[int, str, float] | None:
    """infra_accessibility 미존재 매물에 대해 Tmap POI + 도보시간 조회 후 DB 저장."""
    from tmap import search_places, parse_place, get_walking_time
    from psycopg2.extras import RealDictCursor
    from datetime import datetime, timezone

    alat, alng = article.get("lat"), article.get("lng")
    if not alat or not alng:
        return None

    keyword_kr = INFRA_EN_TO_KR.get(keyword_en, keyword_en)
    docs = search_places(TMAP_API_KEY, keyword_kr, alat, alng, radius=1000)
    if not docs:
        return None

    nearest = min(docs, key=lambda d: d.get("_dist_m", 99999))
    place   = parse_place(nearest)

    try:
        plat = float(place["위도"])
        plng = float(place["경도"])
    except (TypeError, ValueError):
        return None

    walking_min = get_walking_time(TMAP_API_KEY, alat, alng, plat, plng)
    if walking_min is None:
        return None

    now  = datetime.now(timezone.utc)
    conn = _get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        INSERT INTO infrastructures (name, category, location, road_address, created_at, updated_at)
        VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s)
        ON CONFLICT (name, category) DO UPDATE SET updated_at = EXCLUDED.updated_at
        RETURNING id;
    """, (place["이름"], keyword_en, plng, plat, place["도로명주소"], now, now))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return None
    infra_id = row["id"]

    cur.execute("""
        INSERT INTO infra_accessibility (property_id, infrastructure_id, walking_time)
        VALUES (%s, %s, %s)
        ON CONFLICT (property_id, infrastructure_id) DO NOTHING;
    """, (article["id"], infra_id, walking_min))
    conn.commit()
    cur.close()
    conn.close()

    return infra_id, place["이름"], walking_min


def score_infra(articles: list[dict], infra_keywords: list[str]) -> list[dict]:
    for a in articles:
        a["infra_score"] = 0.0
        a["infra_found"] = []
        a["infra_ids"]   = []

    if not infra_keywords or not articles:
        return articles

    from psycopg2.extras import RealDictCursor
    conn = _get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    article_ids = [a["id"] for a in articles]
    id_map      = {a["id"]: a for a in articles}

    for keyword in infra_keywords:
        cur.execute("""
            SELECT DISTINCT ON (ia.property_id)
                ia.property_id  AS article_id,
                ia.walking_time,
                i.id            AS infra_id,
                i.name          AS infra_name
            FROM infra_accessibility ia
            JOIN infrastructures i ON i.id = ia.infrastructure_id
            WHERE ia.property_id = ANY(%s)
              AND i.category = %s
            ORDER BY ia.property_id, ia.walking_time;
        """, (article_ids, keyword))

        found_ids = set()
        for row in cur.fetchall():
            aid = row["article_id"]
            found_ids.add(aid)
            _apply_infra_score(id_map[aid], keyword, row["infra_name"],
                               row["infra_id"], float(row["walking_time"]))

        for aid in article_ids:
            if aid in found_ids:
                continue
            result = _fetch_and_save_infra(id_map[aid], keyword)
            if result:
                infra_id, infra_name, walking_min = result
                _apply_infra_score(id_map[aid], keyword, infra_name, infra_id, walking_min)

    cur.close()
    conn.close()

    n = len(infra_keywords)
    for a in articles:
        a["infra_score"] /= n
    return articles


# ──────────────────────────────────────────────────────────────
# Step 4. 임베딩 유사도 (pgvector)
# ──────────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in resp.data], dtype=np.float32)


def _strip_keywords(text: str, keywords: list[str]) -> str:
    for kw in keywords:
        text = text.replace(kw, "")
    return " ".join(text.split())


def score_embedding(articles: list[dict], soft_text: str, infra_keywords: list[str] | None = None) -> list[dict]:
    for a in articles:
        a["embed_score"] = 0.0

    if not soft_text or not articles:
        return articles

    clean_query = _strip_keywords(soft_text, infra_keywords or [])
    if not clean_query:
        return articles

    from psycopg2.extras import RealDictCursor
    q_vec = _embed([clean_query])[0]
    q_str = "[" + ",".join(f"{x:.8f}" for x in q_vec.tolist()) + "]"

    article_ids = [a["id"] for a in articles]
    id_map      = {a["id"]: a for a in articles}

    conn = _get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT property_id,
               1 - (embedding <=> %s::vector) AS cosine_sim
        FROM property
        WHERE property_id = ANY(%s)
          AND embedding IS NOT NULL;
    """, (q_str, article_ids))

    for row in cur.fetchall():
        aid = row["property_id"]
        if aid in id_map:
            id_map[aid]["embed_score"] = (float(row["cosine_sim"]) + 1.0) / 2.0

    cur.close()
    conn.close()
    return articles


# ──────────────────────────────────────────────────────────────
# Step 5. 랭킹
# ──────────────────────────────────────────────────────────────

INFRA_MAX_PTS    = 40
FREQUENT_MAX_PTS = 20
EMBED_MAX_PTS    = 40


def rank_and_pick(articles: list[dict], top_n: int = 3) -> list[dict]:
    for a in articles:
        infra_pts    = a.get("infra_score",    0.0) * INFRA_MAX_PTS
        frequent_pts = a.get("frequent_score", 0.0) * FREQUENT_MAX_PTS
        embed_pts    = a.get("embed_score",    0.0) * EMBED_MAX_PTS
        a["infra_pts"]    = round(infra_pts, 2)
        a["frequent_pts"] = round(frequent_pts, 2)
        a["embed_pts"]    = round(embed_pts, 2)
        a["total_score"]  = round(infra_pts + frequent_pts + embed_pts, 2)
    return sorted(articles, key=lambda x: x["total_score"], reverse=True)[:top_n]


# ──────────────────────────────────────────────────────────────
# Step 5. 자주 가는 장소 점수 (routes 테이블 + Tmap fallback)
# ──────────────────────────────────────────────────────────────

def _apply_frequent_score(article: dict, place_name: str, duration_min: float) -> None:
    if duration_min <= FREQUENT_MAX_WALK_MIN:
        article["frequent_score"] += 1.0 - (duration_min / FREQUENT_MAX_WALK_MIN)
    article["frequent_found"].append(f"{place_name}({duration_min:.1f}분)")


def _fetch_and_save_route(article: dict, user_place: dict) -> float | None:
    """routes 미존재 쌍에 대해 Tmap 도보시간 계산 후 DB 저장."""
    from tmap import get_walking_time

    alat, alng = article.get("lat"), article.get("lng")
    plat, plng = user_place.get("lat"), user_place.get("lng")
    if not all([alat, alng, plat, plng]):
        return None

    dur = get_walking_time(TMAP_API_KEY, alat, alng, plat, plng)
    if dur is None:
        return None

    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO routes (property_id, userplace_id, duration_minutes, transport_mode, updated_at)
        VALUES (%s, %s, %s, 'pedestrian', NOW())
        ON CONFLICT (property_id, userplace_id) DO NOTHING;
    """, (article["id"], user_place["id"], dur))
    conn.commit()
    cur.close()
    conn.close()
    return dur


def score_frequent(articles: list[dict], seeker_id: int) -> tuple[list[dict], list[str]]:
    """
    seeker_id로 DB에서 user_places 조회 → routes 테이블 기반 점수 계산.
    Returns: (articles, frequent_names)
    """
    for a in articles:
        a["frequent_score"] = 0.0
        a["frequent_found"] = []

    if not seeker_id or not articles:
        return articles, []

    from db import get_user_places_for_seeker
    from psycopg2.extras import RealDictCursor

    user_places = get_user_places_for_seeker(seeker_id)
    if not user_places:
        return articles, []

    article_ids   = [a["id"] for a in articles]
    userplace_ids = [up["id"] for up in user_places]
    id_map        = {a["id"]: a for a in articles}
    up_map        = {up["id"]: up for up in user_places}

    conn = _get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT property_id, userplace_id, MIN(duration_minutes) AS duration_minutes
        FROM routes
        WHERE property_id = ANY(%s)
          AND userplace_id = ANY(%s)
        GROUP BY property_id, userplace_id;
    """, (article_ids, userplace_ids))

    found_pairs = set()
    for row in cur.fetchall():
        aid  = row["property_id"]
        upid = row["userplace_id"]
        found_pairs.add((aid, upid))
        _apply_frequent_score(id_map[aid], up_map[upid]["name"], float(row["duration_minutes"]))

    cur.close()
    conn.close()

    for aid in article_ids:
        for upid in userplace_ids:
            if (aid, upid) in found_pairs:
                continue
            dur = _fetch_and_save_route(id_map[aid], up_map[upid])
            if dur is not None:
                _apply_frequent_score(id_map[aid], up_map[upid]["name"], dur)

    n = len(user_places)
    for a in articles:
        a["frequent_score"] /= n

    return articles, [up["name"] for up in user_places]


# ──────────────────────────────────────────────────────────────
# 매물 줄글 설명 (GPT)
# ──────────────────────────────────────────────────────────────

def explain_article(article: dict, query: str = "", frequent_places: list[str] | None = None) -> str:
    lines = [
        f"매물명: {article.get('title', '')}",
        f"위치: {article.get('address', '')}",
        f"거래: {article.get('trade_type', '')} | 보증금 {article.get('deposit', '')}만원 | 월세 {article.get('monthly_rent', '')}만원",
        f"면적: {article.get('area_m2', '')}m² | 층: {article.get('floor_info', '')}",
    ]
    if article.get("description"):
        lines.append(f"설명: {article['description']}")
    if article.get("tags"):
        lines.append(f"태그: {article['tags']}")
    if article.get("infra_found"):
        lines.append(f"주변 인프라: {', '.join(article['infra_found'])}")
    if article.get("frequent_found"):
        lines.append(f"자주 가는 장소 접근성: {', '.join(article['frequent_found'])}")
    lines += [
        f"인프라 점수: {article.get('infra_pts', 0)}/{INFRA_MAX_PTS}점",
        f"자주 가는 장소 점수: {article.get('frequent_pts', 0)}/{FREQUENT_MAX_PTS}점",
        f"임베딩 유사도 점수: {article.get('embed_pts', 0)}/{EMBED_MAX_PTS}점",
        f"총점: {article.get('total_score', 0)}/100점",
    ]

    user_msg = f"사용자 검색 조건: {query}\n\n매물 정보:\n" + "\n".join(lines)
    if frequent_places:
        user_msg += f"\n\n사용자가 자주 방문하는 장소: {', '.join(frequent_places)}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "당신은 친근하고 전문적인 부동산 컨설턴트입니다. "
                "매물 정보를 바탕으로 이 매물의 특징, 생활 편의성, 장단점을 "
                "자연스러운 한국어 줄글로 3~5문단 설명해주세요. "
                "주변 인프라나 자주 가는 장소의 접근성을 언급할 때는 "
                "구체적인 소요 시간(분)은 절대 쓰지 말고, "
                "'도보 거리', '가까운', '멀지 않은', '편리하게 이용할 수 있는' 등의 표현을 사용하세요."
            )},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# 메인 검색 파이프라인
# ──────────────────────────────────────────────────────────────

def search_stream(query: str, top_n: int = 3, seeker_id: int | None = None,
                  selected_prefs: list[str] | None = None):
    """Yields: (step, status, message, data)"""

    yield (1, "running", "자연어 파싱 중...", None)
    try:
        parsed = parse_query(query)
        parsed = apply_prefs(parsed, selected_prefs or [])
    except Exception as e:
        yield (1, "error", f"파싱 실패: {e}", None)
        return
    yield (1, "done", "파싱 완료", parsed)

    yield (2, "running", "매물 필터링 중...", None)
    candidates = filter_articles(parsed)
    if not candidates:
        yield (2, "error", "조건에 맞는 매물이 없습니다", None)
        return
    yield (2, "done", f"{len(candidates)}건 필터링됨", {"count": len(candidates)})

    infra = parsed.get("infra", [])
    yield (4, "running", "인프라 근접 점수 계산 중...", None)
    candidates = score_infra(candidates, infra)
    yield (4, "done", "완료", None)

    frequent_names: list[str] = []
    if seeker_id:
        yield (5, "running", "자주 가는 장소 점수 계산 중...", None)
        candidates, frequent_names = score_frequent(candidates, seeker_id)
        yield (5, "done", "완료", {"frequent_names": frequent_names})

    soft_text = parsed.get("soft", "")
    print(f"  [임베딩 쿼리] {soft_text!r}")
    yield (6, "running", "임베딩 유사도 계산 중...", None)
    candidates = score_embedding(candidates, soft_text, infra)
    yield (6, "done", "완료", None)

    yield (7, "running", "최종 랭킹 중...", None)
    results = rank_and_pick(candidates, top_n)
    yield (7, "done", f"TOP {len(results)} 선정", results)
