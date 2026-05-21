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
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from tmap import search_places, parse_place

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

INFRA_RADIUS_M = 500


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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
            SELECT DISTINCT ON (a.property_id)
                a.property_id AS article_id,
                i.id          AS infra_id,
                i.name        AS nearest_name,
                ST_Distance(
                    ST_SetSRID(ST_MakePoint(a.longitude, a.latitude), 4326)::geography,
                    i.location
                ) AS min_dist_m
            FROM property a
            JOIN infrastructures i
                ON i.category = %s
               AND ST_DWithin(
                   ST_SetSRID(ST_MakePoint(a.longitude, a.latitude), 4326)::geography,
                   i.location, %s
               )
            WHERE a.property_id = ANY(%s)
            ORDER BY a.property_id, ST_Distance(
                ST_SetSRID(ST_MakePoint(a.longitude, a.latitude), 4326)::geography,
                i.location
            );
        """, (keyword, INFRA_RADIUS_M, article_ids))

        for row in cur.fetchall():
            aid  = row["article_id"]
            dist = float(row["min_dist_m"])
            id_map[aid]["infra_score"] += 1.0 - (dist / INFRA_RADIUS_M)
            id_map[aid]["infra_found"].append(f"{keyword}({row['nearest_name']}, {int(dist)}m)")
            id_map[aid]["infra_ids"].append(row["infra_id"])

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
# Step 5. 자주 가는 장소 점수 (Tmap 지오코딩 + Haversine)
# ──────────────────────────────────────────────────────────────

FREQUENT_RADIUS_M = 3000


def score_frequent(articles: list[dict], frequent_places: list[dict]) -> list[dict]:
    """
    frequent_places 항목: {"name": str, "lat": float|None, "lng": float|None}
    lat/lng 있으면 Tmap 지오코딩 생략.
    """
    for a in articles:
        a["frequent_score"] = 0.0
        a["frequent_found"] = []

    if not frequent_places or not articles:
        return articles

    lats = [a["lat"] for a in articles if a.get("lat")]
    lngs = [a["lng"] for a in articles if a.get("lng")]
    if not lats:
        return articles
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    place_coords: dict[str, tuple[float, float]] = {}
    for place in frequent_places:
        name = place["name"]
        if place.get("lat") and place.get("lng"):
            place_coords[name] = (place["lat"], place["lng"])
            print(f"  [자주 가는 장소] '{name}' 좌표 직접 사용")
        else:
            docs = search_places(TMAP_API_KEY, name, center_lat, center_lng, radius=10000)
            print(f"  [자주 가는 장소] '{name}' Tmap 결과: {len(docs)}개")
            if docs:
                p = parse_place(docs[0])
                try:
                    place_coords[name] = (float(p["위도"]), float(p["경도"]))
                except (TypeError, ValueError) as e:
                    print(f"  [자주 가는 장소] 좌표 파싱 실패: {e}")

    if not place_coords:
        print("  [자주 가는 장소] 좌표 확보 실패 → 점수 0")
        return articles

    n = len(place_coords)
    for a in articles:
        if not a.get("lat") or not a.get("lng"):
            continue
        total = 0.0
        for name, (plat, plng) in place_coords.items():
            dist = _haversine(a["lat"], a["lng"], plat, plng)
            if dist <= FREQUENT_RADIUS_M:
                score = 1.0 - (dist / FREQUENT_RADIUS_M)
                a["frequent_found"].append(f"{name}({int(dist)}m)")
            else:
                score = 0.0
            total += score
        a["frequent_score"] = total / n

    return articles


# ──────────────────────────────────────────────────────────────
# 인프라 수집 (DB에 없으면 Tmap → DB 저장)
# ──────────────────────────────────────────────────────────────

def _save_infra_to_db(places: list[dict], keyword_en: str) -> None:
    try:
        from db import insert_infrastructures
        insert_infrastructures(places, "", keyword_en)
    except Exception as e:
        print(f"    → 인프라 DB 저장 실패: {e}")


def _ensure_infra(infra_keywords: list[str], candidates: list[dict]) -> None:
    """infra_keywords는 영어 코드 (gym, convenience_store, ...) 기준."""
    if not infra_keywords or not candidates:
        return

    lats = [c["lat"] for c in candidates if c.get("lat")]
    lngs = [c["lng"] for c in candidates if c.get("lng")]
    if not lats:
        return
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    conn = _get_conn()
    cur  = conn.cursor()
    missing = []
    for keyword_en in infra_keywords:
        cur.execute("""
            SELECT 1 FROM infrastructures
            WHERE category = %s
              AND ST_DWithin(
                  location,
                  ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                  2000
              )
            LIMIT 1;
        """, (keyword_en, center_lng, center_lat))
        if cur.fetchone():
            print(f"    '{keyword_en}' DB 확인")
        else:
            missing.append(keyword_en)
    cur.close()
    conn.close()

    if not missing:
        return

    def _fetch_and_save(keyword_en: str) -> None:
        keyword_kr = INFRA_EN_TO_KR.get(keyword_en, keyword_en)
        docs   = search_places(TMAP_API_KEY, keyword_kr, center_lat, center_lng, radius=2000)
        places = [parse_place(d) for d in docs]
        if places:
            _save_infra_to_db(places, keyword_en)
            print(f"    '{keyword_en}' → {len(places)}건 수집 완료")
        else:
            print(f"    '{keyword_en}' 반경 2km 내 결과 없음")

    with ThreadPoolExecutor(max_workers=len(missing)) as executor:
        list(executor.map(_fetch_and_save, missing))


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
                "자연스러운 한국어 줄글로 3~5문단 설명해주세요."
            )},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# 메인 검색 파이프라인
# ──────────────────────────────────────────────────────────────

def search_stream(query: str, top_n: int = 3, frequent_places=None,
                  selected_prefs: list[str] | None = None):
    """
    frequent_places: list[str] 또는 list[dict {"name", "lat"?, "lng"?}]
    Yields: (step, status, message, data)
    """
    raw = frequent_places or []
    frequent_places = [
        p if isinstance(p, dict) else {"name": p, "lat": None, "lng": None}
        for p in raw
    ]

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
    if infra:
        yield (3, "running", "인프라 확인 / Tmap 수집 중...", None)
        _ensure_infra(infra, candidates)
        yield (3, "done", "인프라 확인 완료", None)

    yield (4, "running", "인프라 근접 점수 계산 중...", None)
    candidates = score_infra(candidates, infra)
    yield (4, "done", "완료", None)

    if frequent_places:
        yield (5, "running", f"자주 가는 장소 점수 계산 중... ({len(frequent_places)}곳)", None)
        candidates = score_frequent(candidates, frequent_places)
        yield (5, "done", "완료", None)

    soft_text = parsed.get("soft", "")
    print(f"  [임베딩 쿼리] {soft_text!r}")
    yield (6, "running", "임베딩 유사도 계산 중...", None)
    candidates = score_embedding(candidates, soft_text, infra)
    yield (6, "done", "완료", None)

    yield (7, "running", "최종 랭킹 중...", None)
    results = rank_and_pick(candidates, top_n)
    yield (7, "done", f"TOP {len(results)} 선정", results)
