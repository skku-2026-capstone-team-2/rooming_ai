import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# ── 카테고리 영어 통일 매핑 ───────────────────────────────────────


INFRA_KR_TO_EN: dict[str, str] = {
    "헬스장": "gym",
    "편의점": "convenience_store",
    "카페":   "cafe",
    "병원":   "hospital",
    "약국":   "pharmacy",
    "세탁소": "laundry",
    "마트":   "mart",
    "슈퍼":   "mart",
}


def get_connection():
    url = os.environ.get("POSTGRES_DB_URL")
    if url:
        p = urlparse(url)
        import psycopg2
        return psycopg2.connect(
            host=p.hostname,
            port=p.port or 5432,
            dbname=p.path.lstrip("/"),
            user=p.username,
            password=p.password,
        )

    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


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

    conn = get_connection()
    cur  = conn.cursor()
    execute_values(cur, """
        INSERT INTO infrastructures (name, category, location, road_address)
        VALUES %s
        ON CONFLICT (name, category) DO NOTHING;
    """, rows)

    inserted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return inserted
