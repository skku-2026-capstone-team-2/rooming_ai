import os

from dotenv import load_dotenv

load_dotenv()

# ── 카테고리 영어 통일 매핑 ───────────────────────────────────────


INFRA_KR_TO_EN: dict[str, str] = {
    "헬스장": "GYM",
    "편의점": "CONVENIENT_STORE",
    "카페":   "CAFE",
    "병원":   "HOSPITAL",
    "약국":   "PHARMACY",
    "세탁소": "LAUNDRY",
    "마트":   "MART",
    "슈퍼":   "MART",
    "지하철역": "SUBWAY",
    "은행":   "BANK",
    "노래방": "KARAOKE",
    "PC방":   "PC_ROOM",
}


def get_connection():
    url = os.environ.get("POSTGRES_DB_URL")
    if url:
        import psycopg2
        return psycopg2.connect(url)

    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


def get_user_places_for_seeker(seeker_id: int) -> list[dict]:
    """매물 탐색자의 활성화된 자주 가는 장소 목록 반환. lat/lng 포함."""
    from psycopg2.extras import RealDictCursor
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT up.id, up.name, up.address, up.category,
               ST_Y(up.location::geometry) AS lat,
               ST_X(up.location::geometry) AS lng
        FROM target_places up
        JOIN registered_target_place sup ON sup.target_place_id = up.id
        WHERE sup.seeker_id = %s;
    """, (seeker_id,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def insert_infrastructures(places: list[dict], dong_name: str, keyword: str) -> int:
    """인프라 장소를 infrastructures 테이블에 삽입합니다. category는 영어로 저장."""
    if not places:
        return 0

    category_en = INFRA_KR_TO_EN.get(keyword, keyword)

    import psycopg2
    from psycopg2.extras import execute_values

    rows = []
    for p in places:
        try:
            lat = float(p.get("위도") or 0)
            lng = float(p.get("경도") or 0)
            location = f"SRID=4326;POINT({lng} {lat})" if lat and lng else None
        except (TypeError, ValueError):
            location = None

        rows.append((
            p.get("이름"),
            category_en,
            location,
            p.get("도로명주소"),
        ))

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    conn = get_connection()
    cur  = conn.cursor()
    execute_values(cur, """
        INSERT INTO infrastructures (name, category, location, address, created_at, updated_at)
        VALUES %s
        ON CONFLICT ON CONSTRAINT uk_infrastructures_location DO NOTHING;
    """, [(r[0], r[1], r[2], r[3], now, now) for r in rows])

    inserted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return inserted
