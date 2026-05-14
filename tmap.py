"""
Tmap POI 장소 검색
=========================
키워드(편의점, 헬스장 등)로 특정 위치 주변 장소를 검색합니다.

설치:
  pip install requests

사용법:
  python tmap.py
"""

import requests
import json
import csv
import os
import math
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 지역 목록 (위도, 경도)
# ============================================================

DONG_LIST = {
    "1": ("종로구 명륜1가", 37.5894, 126.9980),
    "2": ("종로구 명륜2가", 37.5889, 126.9969),
    "3": ("종로구 명륜3가", 37.5875, 126.9931),
    "4": ("종로구 명륜4가", 37.5864, 126.9951),
    "5": ("종로구 혜화동",  37.5920, 126.9989),
}

# ============================================================
# Tmap API
# ============================================================

TMAP_POI_URL     = "https://apis.openapi.sk.com/tmap/pois"
TMAP_REVERSE_URL = "https://apis.openapi.sk.com/tmap/geo/reversegeocoding"

_TMAP_MAX_RADIUS_M = 5000  # Tmap POI 검색 최대 반경(m)
_TMAP_PAGE_SIZE  = 20    # 페이지당 결과 수


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def coord2address(api_key: str, lat: float, lng: float) -> str:
    """위도·경도 → 도로명주소 (없으면 전체주소) 변환. 실패 시 빈 문자열 반환."""
    resp = requests.get(
        TMAP_REVERSE_URL,
        params={
            "version":     1,
            "lat":         lat,
            "lon":         lng,
            "coordType":   "WGS84GEO",
            "addressType": "A10",
            "appKey":      api_key,
        },
        headers={"Accept": "application/json"},
    )
    if resp.status_code != 200:
        return ""
    info = resp.json().get("addressInfo", {})
    return info.get("roadFullAddr") or info.get("fullAddress") or ""


def search_places(api_key: str, keyword: str, lat: float, lng: float, radius: int = 500) -> list:
    """키워드로 주변 장소 전체 검색 (페이지 자동 순회)."""
    effective_radius_m = min(radius, _TMAP_MAX_RADIUS_M)
    effective_radius_km = max(1, math.ceil(effective_radius_m / 1000))
    results = []
    page = 1

    while True:
        params = {
            "version":       1,
            "searchKeyword": keyword,
            "centerLon":     lng,
            "centerLat":     lat,
            "radius":        effective_radius_km,
            "count":         _TMAP_PAGE_SIZE,
            "page":          page,
            "resCoordType":  "WGS84GEO",
            "appKey":        api_key,
        }
        resp = requests.get(TMAP_POI_URL, params=params, headers={"Accept": "application/json"})

        if resp.status_code != 200:
            print(f"❌ 요청 실패: {resp.status_code}")
            print(f"   응답: {resp.text[:300]}")
            return []

        info = resp.json().get("searchPoiInfo", {})
        pois = info.get("pois", {}).get("poi", [])
        if isinstance(pois, dict):
            pois = [pois]
        elif not isinstance(pois, list):
            pois = []

        # 각 POI에 중심점까지 거리(m) 계산하여 추가
        for poi in pois:
            try:
                poi_lat = float(poi.get("noorLat") or 0)
                poi_lng = float(poi.get("noorLon") or 0)
            except (TypeError, ValueError):
                poi_lat = poi_lng = 0.0
            poi["_dist_m"] = int(_haversine(lat, lng, poi_lat, poi_lng)) if poi_lat and poi_lng else 0

        results.extend(pois)
        total = int(info.get("totalCount") or 0)
        print(f"  → 페이지 {page}: {len(pois)}개 수집 (총 {len(results)}개)")

        if len(pois) < _TMAP_PAGE_SIZE or len(results) >= total:
            break
        page += 1

    return results


def parse_place(poi: dict) -> dict:
    """Tmap POI 결과를 표준 포맷으로 변환 (kakao.parse_place와 동일한 키 구조)."""
    new_addrs = poi.get("newAddressList", {}).get("newAddress", [])
    if isinstance(new_addrs, dict):
        new_addrs = [new_addrs]
    road_addr = new_addrs[0].get("fullAddressRoad", "") if new_addrs else ""

    category = " > ".join(filter(None, [
        poi.get("upperBizName", ""),
        poi.get("middleBizName", ""),
        poi.get("lowerBizName", ""),
    ]))

    try:
        poi_lat = float(poi.get("noorLat") or 0) or None
        poi_lng = float(poi.get("noorLon") or 0) or None
    except (TypeError, ValueError):
        poi_lat = poi_lng = None

    return {
        "이름":       poi.get("name"),
        "카테고리":   category,
        "전화번호":   poi.get("telNo"),
        "주소":       road_addr,
        "지번주소":   road_addr,
        "도로명주소": road_addr,
        "거리(m)":    poi.get("_dist_m"),
        "위도":       str(poi_lat) if poi_lat else None,
        "경도":       str(poi_lng) if poi_lng else None,
        "상세URL":    None,
    }


def save_results(places, dong_name, keyword, save_json=True, save_csv=True):
    if not places:
        print("❌ 저장할 데이터가 없습니다.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{dong_name}_{keyword}_{ts}"

    if save_json:
        path = f"{prefix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(places, f, ensure_ascii=False, indent=2)
        print(f"  💾 {path} ({len(places)}건)")

    if save_csv:
        path = f"{prefix}.csv"
        fields = list(places[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(places)
        print(f"  💾 {path} ({len(places)}건)")


def print_summary(places):
    print()
    print("-" * 60)
    print(" 검색 결과 미리보기")
    print("-" * 60)
    for i, p in enumerate(places[:10], 1):
        print(f"\n  [{i}] {p['이름']}")
        print(f"      {p['카테고리']}")
        print(f"      {p['도로명주소'] or p['주소']}")
        if p['전화번호']:
            print(f"      📞 {p['전화번호']}")
        print(f"      📍 {p['거리(m)']}m")
    if len(places) > 10:
        print(f"\n  ... 외 {len(places) - 10}개")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print(" Tmap POI 장소 검색")
    print("=" * 60)

    api_key = os.environ.get("TMAP_API_KEY", "").strip()
    if api_key:
        print("\n✅ .env에서 API 키 로드됨")
    else:
        api_key = input("\nTmap App Key 입력: ").strip()
    if not api_key:
        print("❌ API 키를 입력하세요.")
        exit(1)

    print("\n[지역 선택]")
    for key, (name, lat, lng) in DONG_LIST.items():
        print(f"  {key}. {name}")
    print("  0. 직접 입력")

    while True:
        sel = input("번호 입력: ").strip()
        if sel in DONG_LIST:
            dong_name, lat, lng = DONG_LIST[sel]
            break
        if sel == "0":
            dong_name = input("  지역 이름: ").strip()
            lat = float(input("  위도 (예: 37.5875): ").strip())
            lng = float(input("  경도 (예: 126.9931): ").strip())
            break
        print("  올바른 번호를 입력하세요.")

    keyword = input("\n검색어 입력 (예: 편의점, 헬스장, 카페): ").strip()
    if not keyword:
        print("❌ 검색어를 입력하세요.")
        exit(1)

    radius_input = input("검색 반경 (미터, 기본값 500, 최대 5000): ").strip()
    radius = int(radius_input) if radius_input.isdigit() else 500

    print("\n[저장 형식 선택] (복수 선택 가능, 예: 1,3)")
    print("  1. JSON")
    print("  2. CSV")
    print("  3. DB (Supabase)")
    while True:
        sel = input("번호 입력 (기본값 1,2): ").strip() or "1,2"
        keys = [k.strip() for k in sel.split(",")]
        if all(k in ("1", "2", "3") for k in keys):
            save_json = "1" in keys
            save_csv  = "2" in keys
            save_db   = "3" in keys
            break
        print("  올바른 번호를 입력하세요.")

    print()
    print(f"  지역: {dong_name} ({lat}, {lng})")
    print(f"  검색어: {keyword}")
    print(f"  반경: {radius}m")
    fmt = " ".join(filter(None, ["JSON" if save_json else "", "CSV" if save_csv else "", "DB" if save_db else ""]))
    print(f"  저장: {fmt}")
    print()
    print("🔍 검색 중...")

    docs   = search_places(api_key, keyword, lat, lng, radius)
    places = [parse_place(d) for d in docs]

    print(f"\n✅ 총 {len(places)}개 수집")

    if places:
        print_summary(places)

        print()
        print("-" * 60)
        print(" 저장")
        print("-" * 60)
        save_results(places, dong_name, keyword, save_json, save_csv)

        if save_db:
            from db import setup_tables, insert_places
            setup_tables()
            inserted = insert_places(places, dong_name, keyword)
            print(f"  🗄️  DB 저장: {inserted}건 삽입")

    print()
    print("=" * 60)
    print(" 완료!")
    print("=" * 60)
