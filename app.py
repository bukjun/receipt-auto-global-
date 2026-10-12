import streamlit as st
import base64
import json
import os
import io
import copy
import hashlib
import html
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import cv2
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageDraw


# =========================================================
# 1. 기본 설정
# =========================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    st.stop()


client = OpenAI(
    api_key=API_KEY,
    timeout=35.0,
    max_retries=1,
)

# 1차 영수증 AI 분석 병렬 처리 수
FIRST_PASS_MAX_WORKERS = 4

# 1차 AI 전송용 이미지 최적화
# 원본 증빙 이미지는 그대로 보존하고, AI에 보내는 사본만 축소한다.
FIRST_PASS_AI_MAX_SIDE = 1800
FIRST_PASS_AI_JPEG_QUALITY = 88

# 매우 높은 신뢰도 + 핵심값/원문/crop이 정상일 때는
# complete_visible / needs_review 같은 보조 플래그만으로 재검증하지 않는다.
HIGH_CONFIDENCE_EVIDENCE_AUTO_APPROVE = 0.97


st.set_page_config(
    page_title="영수증 AI 자동정산",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. 계정과목
# =========================================================

CATEGORIES = [
    "유류비",
    "주차비",
    "통행료",
    "식대",
    "카페/음료",
    "교통비",
    "숙박비",
    "접대비",
    "기타",
]


# =========================================================
# 3. 색상
# =========================================================

BG = "#F5F5F7"
WHITE = "#FFFFFF"

TEXT = "#1D1D1F"
TEXT_SUB = "#6E6E73"

BLUE = "#2F9FD3"
BLUE_DARK = "#147CB5"

LINE = "#D2D2D7"

GREEN = "#248A3D"
GREEN_BG = "#EEF8F1"

RED = "#D92D20"
RED_BG = "#FFF2F1"

ORANGE = "#B54708"
ORANGE_BG = "#FFF7ED"


# =========================================================
# 4. CSS
# =========================================================

st.html(
    f"""
<style>

html, body, [class*="css"] {{
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Apple SD Gothic Neo",
        "Segoe UI",
        "Noto Sans KR",
        sans-serif;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

header[data-testid="stHeader"] {{
    background: transparent !important;
}}

.block-container {{
    max-width: 1320px !important;
    padding-top: 4rem !important;
    padding-bottom: 5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

.tw-hero {{
    width: 100%;
    text-align: center;
    padding: 32px 20px 65px 20px;
}}

.tw-hero-title {{
    color: {TEXT};
    font-size: clamp(42px, 5vw, 68px);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -0.055em;
    margin: 0 0 22px 0;
}}

.tw-hero-subtitle {{
    color: {TEXT};
    font-size: 22px;
    line-height: 1.5;
    font-weight: 650;
    margin: 0 0 13px 0;
}}

.tw-hero-description {{
    max-width: 780px;
    margin: 0 auto;
    color: {TEXT_SUB};
    font-size: 15px;
    line-height: 1.8;
}}

.tw-compact-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 18px;
    margin-bottom: 32px;
    border-bottom: 1px solid {LINE};
}}

.tw-compact-title {{
    font-size: 18px;
    font-weight: 750;
}}

.tw-compact-meta {{
    color: {TEXT_SUB};
    font-size: 13px;
}}

.tw-section {{
    margin-top: 44px;
    margin-bottom: 18px;
}}

.tw-section-title {{
    font-size: 25px;
    font-weight: 760;
    letter-spacing: -0.035em;
    margin-bottom: 5px;
}}

.tw-section-description {{
    color: {TEXT_SUB};
    font-size: 14px;
    line-height: 1.65;
}}

.tw-ready {{
    text-align: center;
    padding: 42px 20px 34px 20px;
}}

.tw-ready-label {{
    color: {GREEN};
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}}

.tw-ready-title {{
    font-size: 31px;
    font-weight: 760;
    letter-spacing: -0.04em;
}}

.tw-ready-description {{
    color: {TEXT_SUB};
    font-size: 14px;
    margin-top: 8px;
}}

section[data-testid="stFileUploaderDropzone"] {{
    background: {WHITE};
    border: 1px solid {LINE};
    border-radius: 18px;
    min-height: 110px;
    padding: 22px !important;
}}

.stButton > button {{
    border-radius: 999px !important;
    min-height: 43px;
    border: none !important;
    background: {BLUE};
    color: white;
    font-weight: 650;
}}

.stButton > button:hover {{
    background: {BLUE_DARK};
    color: white;
}}

.stButton > button:disabled {{
    background: #E5E5EA !important;
    color: #8E8E93 !important;
}}

.stDownloadButton > button {{
    border-radius: 999px !important;
    min-height: 43px;
    background: white;
    color: {BLUE};
    border: 1px solid {BLUE} !important;
    font-weight: 650;
}}

div[data-testid="stMetricValue"] {{
    font-size: 31px !important;
    font-weight: 750;
}}

div[data-testid="stDataFrame"] {{
    border-top: 1px solid {LINE};
    border-bottom: 1px solid {LINE};
}}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {{
    background: white !important;
    border-radius: 12px !important;
}}

details {{
    background: {WHITE};
    border-top: 1px solid {LINE};
    border-radius: 0 !important;
}}

div[data-testid="stForm"] {{
    background: #FAFAFA;
    border: 1px solid #E3E3E7;
    border-radius: 16px;
    padding: 16px;
}}

.tw-attention {{
    background: {RED_BG};
    border: 1px solid #F3C1BD;
    border-radius: 16px;
    padding: 17px 19px;
    margin: 18px 0 14px 0;
}}

.tw-attention-title {{
    color: {RED};
    font-size: 16px;
    font-weight: 750;
    margin-bottom: 7px;
}}

.tw-attention-body {{
    color: #7A271A;
    font-size: 14px;
    line-height: 1.65;
}}

.tw-safety {{
    background: {ORANGE_BG};
    border: 1px solid #FED7AA;
    border-radius: 14px;
    padding: 13px 16px;
    color: {ORANGE};
    font-size: 14px;
    line-height: 1.65;
    margin-top: 12px;
}}

.tw-status-ok {{
    background: {GREEN_BG};
    border: 1px solid #CDEAD5;
    border-radius: 14px;
    padding: 12px 16px;
    color: {GREEN};
    font-size: 14px;
    font-weight: 650;
    margin-top: 12px;
}}

.tw-status-neutral {{
    background: #FAFAFA;
    border: 1px solid #E6E6EB;
    border-radius: 14px;
    padding: 12px 16px;
    color: {TEXT_SUB};
    font-size: 14px;
    margin-top: 12px;
}}

.tw-source-box {{
    background: #F9F9FA;
    border: 1px solid #E5E5EA;
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 14px;
    font-size: 13px;
    color: {TEXT_SUB};
    line-height: 1.7;
}}

.tw-source-title {{
    color: {TEXT};
    font-weight: 700;
    margin-bottom: 5px;
}}

.tw-footer {{
    border-top: 1px solid {LINE};
    margin-top: 64px;
    padding-top: 22px;
    color: #929297;
    text-align: center;
    font-size: 11px;
}}

</style>
"""
)


# =========================================================
# 5. UI 함수
# =========================================================

def section(title, description=""):

    st.html(
        f"""
<div class="tw-section">
    <div class="tw-section-title">
        {html.escape(str(title))}
    </div>
    <div class="tw-section-description">
        {html.escape(str(description))}
    </div>
</div>
"""
    )


def show_attention(reason, action):

    st.html(
        f"""
<div class="tw-attention">
    <div class="tw-attention-title">
        이 영수증은 추가 확인이 필요합니다.
    </div>
    <div class="tw-attention-body">
        사유: {html.escape(str(reason))}<br>
        조치: {html.escape(str(action))}
    </div>
</div>
"""
    )


def show_safety_warning(text):

    st.html(
        f"""
<div class="tw-safety">
    <strong>원문 보호 확인</strong><br>
    {html.escape(str(text))}
</div>
"""
    )


def show_ok(text="증빙 이미지 확인 완료"):

    st.html(
        f"""
<div class="tw-status-ok">
    {html.escape(str(text))}
</div>
"""
    )


def show_neutral(text):

    st.html(
        f"""
<div class="tw-status-neutral">
    {html.escape(str(text))}
</div>
"""
    )


def show_source_data(row):

    merchant_source = str(
        row.get("source_merchant_text", "")
        or ""
    ).strip()

    amount_source = str(
        row.get("source_amount_text", "")
        or ""
    ).strip()

    date_source = str(
        row.get("source_date_text", "")
        or ""
    ).strip()

    items_source = row.get(
        "items_text",
        [],
    )

    if not isinstance(
        items_source,
        list,
    ):
        items_source = []

    items_text = (
        " / ".join(
            str(x)
            for x in items_source
            if str(x).strip()
        )
        or
        "인식된 품목 없음"
    )

    currency = normalize_currency_code(
        row.get("currency"),
        amount_source,
    )

    original_amount = safe_float(
        row.get("original_amount", 0)
    )

    exchange_rate = safe_float(
        row.get("exchange_rate", 0)
    )

    exchange_rate_date = str(
        row.get("exchange_rate_date", "")
        or ""
    ).strip()

    foreign_html = ""

    if currency != "KRW":
        foreign_html = (
            f"<br>원통화: {html.escape(currency)}"
            f"<br>외화금액: {original_amount:,.2f} {html.escape(currency)}"
            f"<br>적용환율: {exchange_rate:,.6f} KRW / 1 {html.escape(currency)}"
            f"<br>환율기준일: {html.escape(exchange_rate_date or '확인 필요')}"
            f"<br>원화환산: {safe_int(row.get('amount', 0)):,}원"
        )

    st.html(
        f"""
<div class="tw-source-box">
    <div class="tw-source-title">
        영수증에서 직접 읽은 원문
    </div>
    상호: {html.escape(merchant_source or "확인 불가")}<br>
    날짜: {html.escape(date_source or "확인 불가")}<br>
    금액: {html.escape(amount_source or "확인 불가")}<br>
    품목: {html.escape(items_text)}
    {foreign_html}
</div>
"""
    )


# =========================================================
# 6. SESSION STATE
# =========================================================

DEFAULT_SESSION = {
    "receipt_df": None,
    "uploaded_images": {},
    "deleted_row": None,
    "edit_receipt_id": None,
    "generated_files": None,
    "flash_message": None,
    "action_errors": {},
    "analysis_cache": {},
    "export_image_cache": {},
    "export_cache_signature": None,
    "fx_rate_cache": {},
}


for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 7. 기본 함수
# =========================================================

def invalidate_generated_files():

    st.session_state.generated_files = None
    st.session_state.export_image_cache = {}
    st.session_state.export_cache_signature = None


def safe_int(value):

    try:
        return int(float(value))
    except Exception:
        return 0


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return 0.0


def safe_bool(value):

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(
        value,
        (int, float),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def get_file_hash(data):

    return hashlib.sha256(
        data
    ).hexdigest()


def clean_json(text):

    text = str(text).strip()

    text = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
    )

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:

        text = text[
            start:
            end + 1
        ]

    return text.strip()


def clean_json_object(text):

    text = str(text).strip()

    text = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
    )

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        text = text[
            start:
            end + 1
        ]

    return text.strip()


def get_settlement_month(df):

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return "정산"

    return dates.max().strftime(
        "%Y-%m"
    )


def get_confidence_state(value):

    value = safe_float(
        value
    )

    if value >= 0.95:
        return "높음"

    if value >= 0.85:
        return "보통"

    return "낮음"


# =========================================================
# 8. 외국어 영수증 / 외화 환산
# =========================================================

SUPPORTED_FOREIGN_CURRENCIES = {
    "THB", "USD", "EUR", "JPY", "CNY", "GBP", "AUD", "CAD",
    "SGD", "HKD", "TWD", "VND", "MYR", "IDR", "PHP", "NZD",
}

CURRENCY_SYMBOL_MAP = {
    "₩": "KRW",
    "원": "KRW",
    "฿": "THB",
    "บาท": "THB",
    "THB": "THB",
    "$": "USD",
    "USD": "USD",
    "US$": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "JPY": "JPY",
    "円": "JPY",
    "CNY": "CNY",
    "RMB": "CNY",
    "CN¥": "CNY",
}


def normalize_currency_code(value, source_text=""):

    raw = str(value or "").strip().upper()

    aliases = {
        "WON": "KRW",
        "KRW": "KRW",
        "BAHT": "THB",
        "THAI BAHT": "THB",
        "THB": "THB",
        "DOLLAR": "USD",
        "US DOLLAR": "USD",
        "USD": "USD",
        "EURO": "EUR",
        "EUR": "EUR",
        "YEN": "JPY",
        "JPY": "JPY",
        "YUAN": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "POUND": "GBP",
        "GBP": "GBP",
    }

    if raw in aliases:
        return aliases[raw]

    combined = f"{value or ''} {source_text or ''}"
    combined_upper = combined.upper()

    # 태국어/기호처럼 명확한 표기를 우선한다.
    if "฿" in combined or "บาท" in combined or "THB" in combined_upper:
        return "THB"

    if "₩" in combined or "원" in combined or "KRW" in combined_upper:
        return "KRW"

    if "US$" in combined_upper or "USD" in combined_upper:
        return "USD"

    if "€" in combined or "EUR" in combined_upper:
        return "EUR"

    if "£" in combined or "GBP" in combined_upper:
        return "GBP"

    if "JPY" in combined_upper or "円" in combined:
        return "JPY"

    if "CNY" in combined_upper or "RMB" in combined_upper or "CN¥" in combined_upper:
        return "CNY"

    # '$'만 있는 경우에는 영문 영수증의 일반적인 경우로 USD 처리.
    if "$" in combined:
        return "USD"

    # 기존 한국 영수증을 깨지 않도록 기본값은 KRW.
    return "KRW"


def parse_decimal_amount(value):
    """
    외화 금액용 파서.

    예:
    1,250.00 -> 1250.00
    1.250,00 -> 1250.00
    ฿1,250.00 -> 1250.00
    $25.50 -> 25.50

    기존 KRW 정수 파서는 별도로 유지한다.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except Exception:
            return None

    text = str(value).strip()

    if not text:
        return None

    negative = False

    if text.startswith("-") or ("(" in text and ")" in text):
        negative = True

    # 숫자와 소수/천단위 구분 기호만 남긴다.
    cleaned = re.sub(r"[^0-9,.-]", "", text)
    cleaned = cleaned.replace("-", "")

    if not cleaned or not re.search(r"[0-9]", cleaned):
        return None

    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")

    decimal_separator = None

    if last_comma >= 0 and last_dot >= 0:
        # 가장 마지막 기호를 소수점으로 본다.
        decimal_separator = "," if last_comma > last_dot else "."

    elif last_comma >= 0:
        tail = cleaned[last_comma + 1:]
        if len(tail) in (1, 2) and tail.isdigit():
            decimal_separator = ","

    elif last_dot >= 0:
        tail = cleaned[last_dot + 1:]
        if len(tail) in (1, 2) and tail.isdigit():
            decimal_separator = "."

    if decimal_separator:
        split_pos = cleaned.rfind(decimal_separator)
        integer_part = re.sub(r"[^0-9]", "", cleaned[:split_pos]) or "0"
        decimal_part = re.sub(r"[^0-9]", "", cleaned[split_pos + 1:])
        normalized = f"{integer_part}.{decimal_part}"
    else:
        normalized = re.sub(r"[^0-9]", "", cleaned)

    if not normalized:
        return None

    try:
        amount = Decimal(normalized)
        return -amount if negative else amount
    except InvalidOperation:
        return None


def get_previous_day_kst():

    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    return (now_kst.date() - timedelta(days=1))


def fetch_fx_rate_to_krw(currency, requested_date=None):
    """
    Frankfurter v2를 사용해 1 외화당 KRW 환율을 조회한다.

    사용자가 요청한 기준은 '어제'.
    해당 날짜 데이터가 없으면 공휴일/휴일을 고려해
    최대 7일 이전까지 가장 가까운 값을 찾는다.

    API 장애 시 임의 환율을 만들지 않고 None을 반환한다.
    """

    currency = normalize_currency_code(currency)

    if currency == "KRW":
        target_date = requested_date or get_previous_day_kst()
        return 1.0, target_date.isoformat()

    if currency not in SUPPORTED_FOREIGN_CURRENCIES:
        return None, None

    target_date = requested_date or get_previous_day_kst()

    for offset in range(0, 8):

        query_date = target_date - timedelta(days=offset)

        cache_key = f"{currency}_KRW_{query_date.isoformat()}"

        if cache_key in st.session_state.fx_rate_cache:
            cached = st.session_state.fx_rate_cache[cache_key]
            return cached.get("rate"), cached.get("date")

        params = urllib.parse.urlencode(
            {
                "date": query_date.isoformat(),
                "base": currency.lower(),
                "quotes": "krw",
            }
        )

        url = f"https://api.frankfurter.dev/v2/rates?{params}"

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Receipt-AI/1.0",
                    "Accept": "application/json",
                },
            )

            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))

            if isinstance(payload, list) and payload:
                matched = next(
                    (
                        item
                        for item in payload
                        if str(item.get("quote", "")).upper() == "KRW"
                    ),
                    payload[0],
                )

                rate = safe_float(matched.get("rate"))
                actual_date = str(matched.get("date") or query_date.isoformat())

                if rate > 0:
                    st.session_state.fx_rate_cache[cache_key] = {
                        "rate": rate,
                        "date": actual_date,
                    }
                    return rate, actual_date

        except Exception:
            # 다음 이전 날짜를 시도한다.
            continue

    return None, None


def apply_currency_conversion(result):
    """
    AI는 환율을 계산하지 않는다.
    AI가 읽은 원통화/원금액을 앱에서 KRW로 환산한다.
    """

    row = dict(result or {})

    source_amount_text = str(
        row.get("source_amount_text", "")
        or ""
    ).strip()

    currency = normalize_currency_code(
        row.get("currency"),
        source_amount_text,
    )

    original_amount = parse_decimal_amount(
        row.get("original_amount")
    )

    if original_amount is None:
        original_amount = parse_decimal_amount(source_amount_text)

    if original_amount is None:
        original_amount = parse_decimal_amount(row.get("amount"))

    row["currency"] = currency
    row["original_amount"] = (
        float(original_amount)
        if original_amount is not None
        else None
    )

    row["exchange_rate"] = None
    row["exchange_rate_date"] = ""
    row["exchange_rate_status"] = ""
    row["fx_review_reason"] = ""
    row["fx_conversion_ok"] = False

    if original_amount is None or original_amount <= 0:
        row["amount"] = 0
        row["needs_review"] = True
        row["fx_review_reason"] = "최종 결제금액을 정확히 읽지 못했습니다."
        return row

    if currency == "KRW":
        row["amount"] = int(round(float(original_amount)))
        row["exchange_rate"] = 1.0
        row["exchange_rate_date"] = ""
        row["exchange_rate_status"] = "원화"
        row["fx_conversion_ok"] = True
        return row

    rate, rate_date = fetch_fx_rate_to_krw(currency)

    if rate is None:
        row["amount"] = 0
        row["exchange_rate_status"] = "환율 확인 필요"
        row["needs_review"] = True
        row["fx_review_reason"] = (
            f"{currency} → KRW 전일 환율을 불러오지 못했습니다. "
            "임의 환산하지 않았습니다."
        )
        return row

    converted_krw = int(
        round(
            float(original_amount)
            *
            float(rate)
        )
    )

    row["amount"] = converted_krw
    row["exchange_rate"] = float(rate)
    row["exchange_rate_date"] = rate_date or ""
    row["exchange_rate_status"] = "전일 환율 자동 환산"
    row["fx_conversion_ok"] = True

    return row


# =========================================================
# 9. AI 원문 보호
# =========================================================

def normalize_text(value):

    return re.sub(
        r"\s+",
        "",
        str(
            value
            or ""
        )
        .strip()
        .lower(),
    )


def normalize_amount_digits(value):

    digits = re.sub(
        r"[^0-9]",
        "",
        str(
            value
            or ""
        ),
    )

    if not digits:
        return None

    try:
        return int(digits)

    except Exception:
        return None


def validate_ai_source_integrity(row):

    issues = []

    currency = normalize_currency_code(
        row.get("currency"),
        row.get("source_amount_text", ""),
    )

    merchant = str(
        row.get("merchant", "")
        or ""
    ).strip()

    source_merchant = str(
        row.get("source_merchant_text", "")
        or ""
    ).strip()

    source_amount_text = str(
        row.get("source_amount_text", "")
        or ""
    ).strip()

    if not merchant:
        issues.append(
            "상호 정보가 비어 있습니다."
        )

    # 상호 원문 표기와 정리된 상호를 서로 비교하지 않는다.
    # 영어/태국어/브랜드 표기 차이는 오류 사유가 아니다.
    if not source_merchant:
        issues.append(
            "영수증 원문에서 상호를 직접 확인하지 못했습니다."
        )

    if currency == "KRW":

        amount = safe_int(
            row.get("amount", 0)
        )

        source_amount = normalize_amount_digits(
            source_amount_text
        )

        if amount <= 0:
            issues.append(
                "최종 결제금액을 확인할 수 없습니다."
            )

        elif source_amount is None:
            issues.append(
                "영수증 원문에서 결제금액을 직접 확인하지 못했습니다."
            )

        elif amount != source_amount:
            issues.append(
                "AI 입력 금액과 영수증 원문 금액이 일치하지 않습니다."
            )

    else:

        original_amount = parse_decimal_amount(
            row.get("original_amount")
        )

        source_original = parse_decimal_amount(
            source_amount_text
        )

        if original_amount is None or original_amount <= 0:
            issues.append(
                "외화 최종 결제금액을 확인할 수 없습니다."
            )

        elif source_original is None:
            issues.append(
                "영수증 원문에서 외화 결제금액을 직접 확인하지 못했습니다."
            )

        elif abs(original_amount - source_original) > Decimal("0.01"):
            issues.append(
                "AI가 읽은 외화 금액과 영수증 원문 금액이 일치하지 않습니다."
            )

        if not safe_bool(
            row.get("fx_conversion_ok", False)
        ):
            issues.append(
                str(
                    row.get(
                        "fx_review_reason",
                        "전일 환율 환산을 확인할 수 없습니다.",
                    )
                    or
                    "전일 환율 환산을 확인할 수 없습니다."
                )
            )

        if safe_int(row.get("amount", 0)) <= 0:
            issues.append(
                "원화 환산금액을 확인할 수 없습니다."
            )

    return (
        len(issues) == 0,
        " / ".join(
            dict.fromkeys(issues)
        ),
    )


def add_ai_safety_status(df):

    result = df.copy()

    if (
        "manual_override"
        not in result.columns
    ):
        result[
            "manual_override"
        ] = False

    result[
        "ai_source_safe"
    ] = False

    result[
        "ai_safety_issue"
    ] = ""

    for idx, row in result.iterrows():

        if safe_bool(
            row.get(
                "manual_override",
                False,
            )
        ):

            result.at[
                idx,
                "ai_source_safe",
            ] = True

            continue

        safe, issue = (
            validate_ai_source_integrity(
                row
            )
        )

        result.at[
            idx,
            "ai_source_safe",
        ] = safe

        result.at[
            idx,
            "ai_safety_issue",
        ] = issue

    return result


# =========================================================
# 9. 좌표 검증
# =========================================================

def valid_bbox(bbox):

    if not isinstance(
        bbox,
        list,
    ):
        return False

    if len(bbox) != 4:
        return False

    try:

        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]

    except Exception:

        return False

    return (
        0 <= x1 < x2 <= 1000
        and
        0 <= y1 < y2 <= 1000
    )


def valid_quad(quad):

    if not isinstance(
        quad,
        list,
    ):
        return False

    if len(quad) != 4:
        return False

    try:

        points = []

        for point in quad:

            if (
                not isinstance(
                    point,
                    list,
                )
                or
                len(point)
                !=
                2
            ):
                return False

            x = float(
                point[0]
            )

            y = float(
                point[1]
            )

            if not (
                0 <= x <= 1000
                and
                0 <= y <= 1000
            ):
                return False

            points.append(
                [
                    x,
                    y,
                ]
            )

    except Exception:

        return False

    area = abs(
        cv2.contourArea(
            np.array(
                points,
                dtype=np.float32,
            )
        )
    )

    return area >= 3000


# =========================================================
# 10. 중복 검사
# =========================================================

def add_duplicate_flags(df):

    result = df.copy()

    result[
        "duplicate_suspect"
    ] = False

    result[
        "_merchant_norm"
    ] = (
        result[
            "merchant"
        ]
        .fillna("")
        .astype(str)
        .str.replace(
            " ",
            "",
            regex=False,
        )
        .str.lower()
    )

    grouped = result.groupby(
        [
            "date",
            "amount",
            "_merchant_norm",
        ],
        dropna=False,
    )

    for _, group in grouped:

        if len(group) >= 2:

            result.loc[
                group.index,
                "duplicate_suspect",
            ] = True

    return result.drop(
        columns=[
            "_merchant_norm"
        ]
    )


# =========================================================
# 11. 증빙 상태
# =========================================================

def add_final_evidence_status(df):

    result = df.copy()

    if (
        "evidence_ok"
        not in result.columns
    ):

        result[
            "evidence_ok"
        ] = False

    if (
        "evidence_manual_approved"
        not in result.columns
    ):

        result[
            "evidence_manual_approved"
        ] = False

    if (
        "data_manual_approved"
        not in result.columns
    ):

        result[
            "data_manual_approved"
        ] = False

    result[
        "evidence_final_ok"
    ] = (
        (
            result[
                "evidence_ok"
            ]
            ==
            True
        )
        |
        (
            result[
                "evidence_manual_approved"
            ]
            ==
            True
        )
    )

    return result


# =========================================================
# 12. 확인 필요
# =========================================================

def row_attention_reasons(row):

    reasons = []

    if (
        not safe_bool(
            row.get(
                "ai_source_safe",
                False,
            )
        )
        and
        not safe_bool(
            row.get(
                "manual_override",
                False,
            )
        )
    ):

        reasons.append(
            str(
                row.get(
                    "ai_safety_issue",
                    "",
                )
                or
                "AI 원문 대조가 필요합니다."
            )
        )

    if not safe_bool(
        row.get(
            "confirmed",
            False,
        )
    ):

        confidence = safe_float(
            row.get(
                "confidence",
                0,
            )
        )

        if confidence < 0.90:

            reasons.append(
                f"AI 인식 신뢰도가 {confidence * 100:.0f}%입니다."
            )

        else:

            reasons.append(
                "영수증 내용을 최종 확인해주세요."
            )

    if not safe_bool(
        row.get(
            "evidence_final_ok",
            False,
        )
    ):

        issue = str(
            row.get(
                "evidence_issue",
                "",
            )
            or ""
        ).strip()

        reasons.append(
            issue
            or
            "증빙 이미지를 확인해주세요."
        )

    if safe_bool(
        row.get(
            "duplicate_suspect",
            False,
        )
    ):

        reasons.append(
            "중복 업로드 가능성이 있습니다."
        )

    return reasons


def row_needs_attention(row):

    return bool(
        row_attention_reasons(
            row
        )
    )


def row_status_label(row):

    return (
        "확인 필요"
        if row_needs_attention(
            row
        )
        else
        "정상"
    )


def row_next_action(row):

    actions = []

    if (
        not safe_bool(
            row.get(
                "ai_source_safe",
                False,
            )
        )
        and
        not safe_bool(
            row.get(
                "manual_override",
                False,
            )
        )
    ):

        actions.append(
            "원문 확인"
        )

    if not safe_bool(
        row.get(
            "evidence_final_ok",
            False,
        )
    ):

        actions.append(
            "증빙 확인"
        )

    if not safe_bool(
        row.get(
            "confirmed",
            False,
        )
    ):

        actions.append(
            "내용 확인"
        )

    if safe_bool(
        row.get(
            "duplicate_suspect",
            False,
        )
    ):

        actions.append(
            "중복 확인"
        )

    return (
        " / ".join(
            actions
        )
        if actions
        else
        "완료"
    )


# =========================================================
# 13. 사용자 액션
# =========================================================

def set_flash(
    message,
    kind="success",
):

    st.session_state.flash_message = {
        "message":
            message,
        "type":
            kind,
    }


def confirm_receipt(
    receipt_id
):

    df = (
        st.session_state
        .receipt_df
        .copy()
    )

    mask = (
        df[
            "receipt_id"
        ]
        ==
        receipt_id
    )

    if not mask.any():
        return False

    df.loc[
        mask,
        "confirmed",
    ] = True

    df.loc[
        mask,
        "data_manual_approved",
    ] = True

    df.loc[
        mask,
        "manual_override",
    ] = True

    # 사용자가 검증 화면의 실제 영수증 이미지를 보고
    # "직접 확인 후 확정"을 누른 경우, 내용과 증빙을 동시에 승인한다.
    # 따라서 별도의 "증빙 정상 승인" 버튼을 다시 누를 필요가 없다.
    df.loc[
        mask,
        "evidence_manual_approved",
    ] = True

    # 중요:
    # 복수 영수증 crop이 자동 증빙검증을 통과하지 못한 상태라면
    # 사용자는 검증 화면에서 원본 전체 이미지를 보고 확정하게 된다.
    # 이 경우 최종 PDF/JPG에도 잘린 crop을 다시 쓰지 않고
    # 사용자가 실제 확인한 원본 전체 이미지를 사용한다.
    current_evidence_ok = (
        df.loc[
            mask,
            "evidence_ok",
        ]
        .apply(
            safe_bool
        )
    )

    current_crop_mode = (
        df.loc[
            mask,
            "crop_mode",
        ]
        .astype(
            str
        )
    )

    force_full_mask = (
        (
            current_crop_mode
            !=
            "full_image"
        )
        &
        (
            current_evidence_ok
            ==
            False
        )
    )

    if "force_full_image_export" not in df.columns:

        df[
            "force_full_image_export"
        ] = False

    df.loc[
        mask,
        "force_full_image_export",
    ] = (
        force_full_mask
        .values
    )

    df.loc[
        mask,
        "needs_review",
    ] = False

    df.loc[
        mask,
        "review_question",
    ] = ""

    df = add_ai_safety_status(
        df
    )

    df = add_duplicate_flags(
        df
    )

    df = add_final_evidence_status(
        df
    )

    st.session_state.receipt_df = df

    invalidate_generated_files()

    set_flash(
        "영수증 내용과 증빙 이미지를 직접 확인하여 최종 확정했습니다."
    )

    return True


def approve_evidence(
    receipt_id
):

    df = (
        st.session_state
        .receipt_df
        .copy()
    )

    mask = (
        df[
            "receipt_id"
        ]
        ==
        receipt_id
    )

    if not mask.any():
        return False

    df.loc[
        mask,
        "evidence_manual_approved",
    ] = True

    df = add_final_evidence_status(
        df
    )

    st.session_state.receipt_df = df

    invalidate_generated_files()

    set_flash(
        "증빙 이미지를 정상 승인했습니다."
    )

    return True


# =========================================================
# 14. AI 분석
# =========================================================

def prepare_first_pass_ai_image(
    file_bytes,
    max_side=FIRST_PASS_AI_MAX_SIDE,
    quality=FIRST_PASS_AI_JPEG_QUALITY,
):
    """
    1차 AI 분석 전송용 사본만 축소한다.

    - 원본 file_bytes는 절대 변경하지 않는다.
    - 긴 변이 max_side보다 큰 경우에만 비율 유지 축소한다.
    - API 전송용 사본은 JPEG로 통일한다.
    - EXIF 회전을 먼저 반영한다.

    실패 시에는 원본 bytes와 원래 mime type을 그대로 사용할 수 있도록
    (file_bytes, None)을 반환한다.
    """

    try:
        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )

        image = ImageOps.exif_transpose(
            image
        ).convert(
            "RGB"
        )

        width, height = image.size
        longest = max(
            width,
            height,
        )

        if longest > max_side:
            scale = (
                max_side
                /
                longest
            )

            image = image.resize(
                (
                    max(
                        1,
                        int(
                            round(
                                width
                                *
                                scale
                            )
                        ),
                    ),
                    max(
                        1,
                        int(
                            round(
                                height
                                *
                                scale
                            )
                        ),
                    ),
                ),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=False,
        )

        return (
            output.getvalue(),
            "image/jpeg",
        )

    except Exception:
        return (
            file_bytes,
            None,
        )


def get_receipt_analysis_cache_key(
    file_bytes,
):

    return (
        get_file_hash(
            file_bytes
        )
        +
        "_receipt_v12_amountsafe"
    )


def analyze_receipt_image_uncached(
    file_name,
    file_bytes,
    mime_type,
):
    """
    1차 AI 분석의 순수 worker 함수.

    중요:
    - Streamlit session_state를 건드리지 않는다.
    - worker마다 OpenAI client를 따로 만든다.
    - 병렬 실행 안전성을 높인다.
    """

    # 원본 증빙은 그대로 두고, AI에 전송하는 사본만 축소한다.
    ai_file_bytes, optimized_mime_type = (
        prepare_first_pass_ai_image(
            file_bytes
        )
    )

    ai_mime_type = (
        optimized_mime_type
        or
        mime_type
        or
        "image/jpeg"
    )

    encoded = (
        base64
        .b64encode(
            ai_file_bytes
        )
        .decode(
            "utf-8"
        )
    )

    prompt = """
너는 대한민국 회사 경비정산용 영수증 판독 AI다.

한국어뿐 아니라 영어 및 태국어 영수증도 판독한다.
영문/태국 영수증이라고 해서 한국 영수증 형식을 억지로 적용하지 않는다.

가장 중요한 원칙은 원문 보존이다.

영수증에 실제로 보이지 않는 금액, 상호, 날짜, 품목을
추측하거나 수정하거나 만들어내면 절대로 안 된다.

금액을 반올림하지 않는다.
합계를 임의로 다시 계산하지 않는다.
상호를 다른 공식 브랜드명으로 바꾸지 않는다.
품목을 추측해서 추가하지 않는다.
환율을 추측하거나 직접 계산하지 않는다.

영어 영수증에서는 TOTAL, GRAND TOTAL, AMOUNT PAID, TOTAL PAID 같은 최종 결제금액을 우선한다.
SUBTOTAL, TAX, VAT, SERVICE CHARGE를 최종 결제금액으로 착각하지 않는다.

태국어 영수증에서는 ยอดรวม, ยอดสุทธิ, รวม, จำนวนเงิน, ยอดชำระ, ชำระเงิน 및 บาท/฿ 표기를 최종 결제금액 판단에 활용한다.
VAT/ภาษี가 따로 표시되더라도 최종 합계 금액을 선택한다.

불확실하면 needs_review=true.

source_merchant_text:
영수증에 실제 인쇄된 상호.

source_date_text:
영수증에 실제 인쇄된 날짜.

source_amount_text:
최종 결제금액 위치에 실제 인쇄된 문자열.

items_text:
실제로 읽히는 품목명만 배열로 반환.

currency:
영수증에 표시된 실제 결제 통화의 ISO 4217 코드.
예: KRW, THB, USD, EUR, JPY.
한국 원화면 KRW. 태국 바트면 THB.
통화를 확신하지 못하면 보이는 기호/문구를 근거로 가장 명확한 통화만 선택하고, 정말 판단할 수 없으면 needs_review=true.

original_amount:
영수증에 표시된 실제 최종 결제금액을 원통화 기준 숫자로 반환한다.
소수점이 있으면 반드시 보존한다.
예: ฿1,250.00 -> 1250.00
$25.50 -> 25.50

amount:
원화 영수증은 실제 최종 결제금액 숫자.
외화 영수증은 original_amount와 같은 원통화 숫자를 그대로 반환한다.
환율 계산은 절대로 하지 않는다. 앱이 별도로 처리한다.

한 이미지에 여러 영수증이 있으면 각각 별도 객체로 반환한다.
같은 영수증을 중복 반환하지 않는다.

receipt_quad:
영수증 종이 전체의 네 모서리.
영수증 일부가 아니라 종이 전체 길이를 반드시 포함한다.
좌상 → 우상 → 우하 → 좌하.
좌표는 0~1000.

bbox:
영수증 전체 종이를 모두 포함하는 영역.
절대로 글자 영역만 잡지 않는다.
[x1,y1,x2,y2]
좌표는 0~1000.

amount_bbox:
최종 결제금액(source_amount_text)이 실제로 인쇄된 영역만 감싸는 사각형.
SUBTOTAL, VAT, TAX가 아니라 실제 최종 결제금액 위치를 잡는다.
[x1,y1,x2,y2]
좌표는 0~1000.
최종 결제금액 위치를 확신하지 못하면 null.

위치를 확신하지 못하면 억지로 좌표를 만들지 말고 null.

category:
유류비
주차비
통행료
식대
카페/음료
교통비
숙박비
접대비
기타

카페/음료:
커피, 차, 티, 라떼, 에스프레소, 주스,
스무디, 탄산음료 등 음료 구매가 주목적인 경우.

식대:
식당, 도시락, 한식, 중식, 일식, 양식,
분식, 김밥, 햄버거, 배달음식 등 식사가 주목적인 경우.

카페에서 음식과 음료가 섞여 애매하면
needs_review=true.

반환:

date
merchant
source_merchant_text
source_date_text
currency
original_amount
amount
source_amount_text
items_text
category
confidence
needs_review
review_question
complete_visible
bbox
amount_bbox
receipt_quad

반드시 JSON 배열만 출력한다.
"""

    worker_client = OpenAI(
        api_key=API_KEY,
        timeout=35.0,
        max_retries=1,
    )

    response = worker_client.responses.create(
        model="gpt-5.6-luna",
        store=False,
        input=[
            {
                "role":
                    "user",
                "content":
                    [
                        {
                            "type":
                                "input_text",
                            "text":
                                prompt,
                        },
                        {
                            "type":
                                "input_image",
                            "image_url":
                                f"data:{ai_mime_type};base64,{encoded}",
                            "detail":
                                "high",
                        },
                    ],
            }
        ],
    )

    return json.loads(
        clean_json(
            response.output_text
        )
    )


def analyze_receipt_image(
    file_name,
    file_bytes,
    mime_type,
):
    """
    기존 단일 호출 호환용 wrapper.
    메인 스레드에서 호출할 때는 기존 분석 cache를 그대로 사용한다.
    """

    cache_key = (
        get_receipt_analysis_cache_key(
            file_bytes
        )
    )

    if (
        cache_key
        in
        st.session_state.analysis_cache
    ):

        return (
            copy.deepcopy(
                st.session_state
                .analysis_cache[
                    cache_key
                ]
            )
        )

    result = (
        analyze_receipt_image_uncached(
            file_name,
            file_bytes,
            mime_type,
        )
    )

    st.session_state.analysis_cache[
        cache_key
    ] = copy.deepcopy(
        result
    )

    return result


# =========================================================
# 15. 안전한 Crop 영역 생성
# =========================================================
#
# 중요 원칙
#
# 사진 1장 = 영수증 1장
# → AI bbox / quad를 아예 사용하지 않음
# → 원본 전체 이미지를 증빙으로 사용
#
# 사진 1장 = 영수증 여러 장
# → bbox와 quad를 합쳐 가장 넓은 영역 사용
# → 상하좌우 안전 여백 추가
#
# 출력용 증빙에서 영수증을 예쁘게 펴기 위한
# perspective warp는 사용하지 않음.
#
# =========================================================

def prepare_crop_boxes(df):

    result = df.copy()

    result[
        "crop_bbox"
    ] = None

    result[
        "crop_ok"
    ] = False

    result[
        "crop_mode"
    ] = ""

    result[
        "crop_geometry_issue"
    ] = ""

    result[
        "source_receipt_count"
    ] = 0

    for source_file, group in result.groupby(
        "source_file"
    ):

        receipt_count = len(
            group
        )

        # -------------------------------------------------
        # 사진 한 장에 영수증 한 장
        #
        # 좌표를 절대 사용하지 않음.
        # 원본 전체 사용.
        # -------------------------------------------------

        if receipt_count == 1:

            idx = group.index[0]

            result.at[
                idx,
                "source_receipt_count",
            ] = 1

            result.at[
                idx,
                "crop_ok",
            ] = True

            result.at[
                idx,
                "crop_mode",
            ] = "full_image"

            result.at[
                idx,
                "crop_geometry_issue",
            ] = ""

            continue


        # -------------------------------------------------
        # 여러 영수증 사진
        # -------------------------------------------------

        for idx, row in group.iterrows():

            result.at[
                idx,
                "source_receipt_count",
            ] = receipt_count

            candidates = []

            bbox = row.get(
                "bbox"
            )

            amount_bbox = row.get(
                "amount_bbox"
            )

            quad = row.get(
                "receipt_quad"
            )

            # bbox 후보
            if valid_bbox(
                bbox
            ):

                x1, y1, x2, y2 = map(
                    float,
                    bbox,
                )

                candidates.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

            # 최종 청구금액 영역은 절대 crop 밖으로 나가면 안 된다.
            # AI가 영수증 전체 bbox를 조금 짧게 잡더라도
            # amount_bbox를 crop 후보에 포함해 최종금액 영역을 강제로 보존한다.
            if valid_bbox(
                amount_bbox
            ):

                ax1, ay1, ax2, ay2 = map(
                    float,
                    amount_bbox,
                )

                candidates.append(
                    (
                        ax1,
                        ay1,
                        ax2,
                        ay2,
                    )
                )

            # quad를 직사각형으로 변환해서 후보 추가
            if valid_quad(
                quad
            ):

                xs = [
                    float(
                        point[
                            0
                        ]
                    )
                    for point in quad
                ]

                ys = [
                    float(
                        point[
                            1
                        ]
                    )
                    for point in quad
                ]

                candidates.append(
                    (
                        min(
                            xs
                        ),
                        min(
                            ys
                        ),
                        max(
                            xs
                        ),
                        max(
                            ys
                        ),
                    )
                )

            if not candidates:

                result.at[
                    idx,
                    "crop_ok",
                ] = False

                result.at[
                    idx,
                    "crop_mode",
                ] = "failed"

                result.at[
                    idx,
                    "crop_geometry_issue",
                ] = (
                    "한 사진에 여러 영수증이 있지만 "
                    "해당 영수증 위치를 정확히 찾지 못했습니다."
                )

                continue

            # bbox와 quad 중 더 넓은 범위를 모두 포함
            x1 = min(
                item[
                    0
                ]
                for item in candidates
            )

            y1 = min(
                item[
                    1
                ]
                for item in candidates
            )

            x2 = max(
                item[
                    2
                ]
                for item in candidates
            )

            y2 = max(
                item[
                    3
                ]
                for item in candidates
            )

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )

            if (
                box_width < 40
                or
                box_height < 60
            ):

                result.at[
                    idx,
                    "crop_ok",
                ] = False

                result.at[
                    idx,
                    "crop_mode",
                ] = "failed"

                result.at[
                    idx,
                    "crop_geometry_issue",
                ] = (
                    "인식된 영수증 영역이 너무 작습니다."
                )

                continue


            # -------------------------------------------------
            # 안전 여백
            #
            # 특히 세로 방향을 넉넉하게 잡음.
            # AI가 끝부분을 조금 덜 잡아도 잘리지 않도록.
            # -------------------------------------------------

            padding_x = max(
                10,
                box_width
                *
                0.04,
            )

            padding_y = max(
                24,
                box_height
                *
                0.12,
            )

            safe_box = [
                max(
                    0,
                    x1 - padding_x,
                ),
                max(
                    0,
                    y1 - padding_y,
                ),
                min(
                    1000,
                    x2 + padding_x,
                ),
                min(
                    1000,
                    y2 + padding_y,
                ),
            ]

            result.at[
                idx,
                "crop_bbox",
            ] = safe_box

            result.at[
                idx,
                "crop_ok",
            ] = True

            result.at[
                idx,
                "crop_mode",
            ] = "safe_bbox"

            result.at[
                idx,
                "crop_geometry_issue",
            ] = ""

    return result


# =========================================================
# 16. 원본 이미지 로드
# =========================================================

def load_source_image(
    source_file
):

    if (
        source_file
        not in
        st.session_state.uploaded_images
    ):

        return None

    raw = (
        st.session_state
        .uploaded_images[
            source_file
        ][
            "bytes"
        ]
    )

    try:

        image = Image.open(
            io.BytesIO(
                raw
            )
        )

        image = ImageOps.exif_transpose(
            image
        )

        return image.convert(
            "RGB"
        )

    except Exception:

        return None


# =========================================================
# 17. 안전한 영수증 이미지 생성
# =========================================================
#
# 중요:
#
# 원근변환 안 함
# Warp 안 함
# 곡률보정 안 함
#
# 증빙에서 내용이 사라지는 위험을 없애기 위해
# 원본 픽셀 범위 자체를 보존함.
#
# =========================================================

def crop_receipt(row):

    image = load_source_image(
        row[
            "source_file"
        ]
    )

    if image is None:
        return None

    width, height = (
        image.size
    )

    crop_mode = str(
        row.get(
            "crop_mode",
            "",
        )
        or ""
    )

    # -----------------------------------------------------
    # 사람이 검증 화면의 원본 전체 이미지를 직접 확인한 뒤
    # 확정한 특수 케이스.
    #
    # 잘린 crop을 PDF/JPG에 다시 쓰지 않고 원본 전체를 사용한다.
    # -----------------------------------------------------

    if safe_bool(
        row.get(
            "force_full_image_export",
            False,
        )
    ):

        return image.copy()


    # -----------------------------------------------------
    # 단일 영수증
    # 원본 이미지 전체 사용
    # -----------------------------------------------------

    if (
        crop_mode
        ==
        "full_image"
    ):

        return image.copy()


    # -----------------------------------------------------
    # 여러 영수증
    # 안전 bbox 사용
    # -----------------------------------------------------

    bbox = row.get(
        "crop_bbox"
    )

    if valid_bbox(
        bbox
    ):

        x1, y1, x2, y2 = map(
            float,
            bbox,
        )

        left = max(
            0,
            int(
                width
                *
                x1
                /
                1000
            ),
        )

        top = max(
            0,
            int(
                height
                *
                y1
                /
                1000
            ),
        )

        right = min(
            width,
            int(
                width
                *
                x2
                /
                1000
            ),
        )

        bottom = min(
            height,
            int(
                height
                *
                y2
                /
                1000
            ),
        )

        if (
            right > left
            and
            bottom > top
        ):

            return image.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )


    # -----------------------------------------------------
    # 과거 세션 호환
    #
    # source_receipt_count == 1이면
    # 무조건 원본 사용
    # -----------------------------------------------------

    source_receipt_count = safe_int(
        row.get(
            "source_receipt_count",
            0,
        )
    )

    if (
        source_receipt_count
        ==
        1
    ):

        return image.copy()

    return None


# =========================================================
# 18. Export 이미지
# =========================================================
#
# 예전처럼 dewarp / warp / 곡률보정을 하지 않는다.
#
# 이 부분이:
# 1. 영수증 잘림 방지
# 2. 이미지 왜곡 방지
# 3. 파일 생성시간 단축
#
# 세 가지를 동시에 해결한다.
#
# =========================================================

def get_display_receipt_image(
    row
):
    """
    검증 화면 전용 이미지.

    원칙:
    - 자동 증빙검증을 통과한 정상 crop만 crop으로 보여준다.
    - crop이 잘렸거나 최종금액 확인이 실패한 경우에는
      사용자가 판단할 수 있도록 원본 전체 이미지를 보여준다.
    - 이 함수는 화면 표시 전용이다.
    """

    evidence_ok = safe_bool(
        row.get(
            "evidence_ok",
            False,
        )
    )

    crop_mode = str(
        row.get(
            "crop_mode",
            "",
        )
        or ""
    )

    # 단일 영수증은 원래부터 원본 전체이므로 그대로 표시.
    if (
        crop_mode
        ==
        "full_image"
    ):

        return crop_receipt(
            row
        )

    # 복수 영수증 crop은 자동 증빙 검증을 통과한 경우에만 표시.
    if evidence_ok:

        image = crop_receipt(
            row
        )

        if image is not None:

            return image

    # crop 실패/잘림/최종금액 미확인 등 검증이 필요한 경우:
    # 원본 전체를 표시해서 사람이 실제 증빙을 보고 판단할 수 있게 한다.
    source_file = str(
        row.get(
            "source_file",
            "",
        )
        or ""
    )

    if not source_file:

        return None

    return load_source_image(
        source_file
    )


def get_export_receipt_image(
    row
):

    image = crop_receipt(
        row
    )

    if image is None:
        return None

    return image.convert(
        "RGB"
    )


# =========================================================
# 19. AI 증빙 검증용 이미지
# =========================================================

def pil_to_data_url(
    image
):

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=84,
        optimize=False,
    )

    encoded = (
        base64
        .b64encode(
            output.getvalue()
        )
        .decode(
            "utf-8"
        )
    )

    return (
        "data:image/jpeg;base64,"
        +
        encoded
    )


# =========================================================
# 20. 자동 증빙 승인
# =========================================================

def can_auto_approve_evidence(
    row
):
    """
    증빙 자동 승인 판단.

    절대 조건:
    - 사진 1장에 영수증 1장(full_image)은 기존 고신뢰도 자동승인 가능.
    - 사진 1장에 여러 영수증이 있는 safe_bbox crop은
      신뢰도가 99%여도 자동승인하지 않는다.
      반드시 2차 증빙 검증에서 '최종 청구금액이 실제로 보이는지' 확인한다.

    이유:
    데이터 인식 신뢰도가 높아도 crop 좌표가 짧게 잡히면
    PDF/JPG 증빙에서 최종금액이 사라질 수 있기 때문이다.
    """

    confidence = safe_float(
        row.get(
            "confidence",
            0,
        )
    )

    source_safe = safe_bool(
        row.get(
            "ai_source_safe",
            False,
        )
    )

    crop_ok = safe_bool(
        row.get(
            "crop_ok",
            False,
        )
    )

    amount_ok = (
        safe_int(
            row.get(
                "amount",
                0,
            )
        )
        >
        0
    )

    merchant_ok = bool(
        str(
            row.get(
                "merchant",
                "",
            )
            or
            ""
        ).strip()
    )

    date_ok = bool(
        str(
            row.get(
                "date",
                "",
            )
            or
            ""
        ).strip()
    )

    crop_mode = str(
        row.get(
            "crop_mode",
            "",
        )
        or ""
    )

    source_receipt_count = safe_int(
        row.get(
            "source_receipt_count",
            0,
        )
    )

    # 복수 영수증 crop은 무조건 AI 증빙 재검증으로 보낸다.
    if (
        source_receipt_count
        >
        1
        or
        crop_mode
        ==
        "safe_bbox"
    ):

        return False

    # 단일 영수증 full_image만 고신뢰도 우회 허용.
    if (
        crop_mode
        ==
        "full_image"
        and
        confidence
        >=
        HIGH_CONFIDENCE_EVIDENCE_AUTO_APPROVE
        and
        source_safe
        and
        crop_ok
        and
        amount_ok
        and
        merchant_ok
        and
        date_ok
    ):

        return True

    return (
        crop_mode
        ==
        "full_image"
        and
        confidence
        >=
        0.93
        and
        source_safe
        and
        safe_bool(
            row.get(
                "complete_visible",
                False,
            )
        )
        and
        not safe_bool(
            row.get(
                "needs_review",
                False,
            )
        )
        and
        crop_ok
        and
        amount_ok
    )


# =========================================================
# 21. AI 증빙 검증
# =========================================================

def validate_receipt_crop(
    row
):

    crop = crop_receipt(
        row
    )

    if crop is None:

        return {
            "evidence_ok":
                False,
            "evidence_confidence":
                0,
            "evidence_issue":
                "영수증 이미지를 만들 수 없습니다.",
        }

    max_side = max(
        crop.size
    )

    if (
        max_side
        >
        1200
    ):

        scale = (
            1200
            /
            max_side
        )

        crop = crop.resize(
            (
                max(
                    1,
                    int(
                        crop.width
                        *
                        scale
                    ),
                ),
                max(
                    1,
                    int(
                        crop.height
                        *
                        scale
                    ),
                ),
            ),
            Image.Resampling.BILINEAR,
        )

    expected_source_amount = str(
        row.get(
            "source_amount_text",
            "",
        )
        or ""
    ).strip()

    expected_original_amount = row.get(
        "original_amount",
        None,
    )

    expected_currency = str(
        row.get(
            "currency",
            "",
        )
        or ""
    ).strip()

    expected_krw_amount = safe_int(
        row.get(
            "amount",
            0,
        )
    )

    prompt = f"""
회사 경비정산용 영수증 증빙 이미지다.

이 단계에서는 금액, 상호, 날짜, 품목을 수정하거나 생성하지 않는다.

오직 증빙 이미지가 실무적으로 사용할 수 있는지만 판단한다.

[절대 조건 - 최종 청구금액]
이 crop 안에는 실제 최종 청구/결제금액이 반드시 완전하게 보여야 한다.
숫자의 일부가 잘렸거나, TOTAL/합계/결제금액 줄이 crop 밖으로 나갔거나,
SUBTOTAL/VAT/TAX만 보이고 실제 최종금액이 안 보이면 무조건 실패다.

기대 원문 금액:
{expected_source_amount}

기대 원통화 금액:
{expected_original_amount} {expected_currency}

정산 원화 금액:
{expected_krw_amount} KRW

외화 영수증에서는 원화환산금액이 영수증에 직접 인쇄되어 있을 필요는 없다.
대신 원통화의 실제 최종 결제금액이 완전하게 보여야 한다.

정상으로 인정:
- 영수증 전체 사진을 그대로 사용
- 배경이 일부 보임
- 영수증 주변 여백이 보임
- 영수증이 약간 기울어짐
- 종이가 조금 휘어 있음
- 사진 경계에 영수증이 가까움

확인 필요:
- 영수증 실제 위/아래 일부가 잘림
- 날짜/상호/최종금액이 잘려 읽을 수 없음
- 다른 영수증이 본 영수증을 크게 가림
- 증빙으로 사용하기 어려울 정도로 흐림

JSON 객체만 출력:

{{
 "primary_receipt_clear": true,
 "materially_truncated": false,
 "another_receipt_materially_visible": false,
 "important_content_obscured": false,
 "merchant_readable": true,
 "date_readable": true,
 "amount_readable": true,
 "final_amount_visible": true,
 "final_amount_matches_expected": true,
 "usable_as_evidence": true,
 "confidence": 0.95,
 "issue": ""
}}
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            store=False,
            input=[
                {
                    "role":
                        "user",
                    "content":
                    [
                        {
                            "type":
                                "input_text",
                            "text":
                                prompt,
                        },
                        {
                            "type":
                                "input_image",
                            "image_url":
                                pil_to_data_url(
                                    crop
                                ),
                            "detail":
                                "high",
                        },
                    ],
                }
            ],
        )

        result = json.loads(
            clean_json_object(
                response.output_text
            )
        )

    except Exception:

        return {
            "evidence_ok":
                False,
            "evidence_confidence":
                0,
            "evidence_issue":
                "자동 재검증을 완료하지 못했습니다.",
        }

    confidence = safe_float(
        result.get(
            "confidence",
            0,
        )
    )

    ok = (
        safe_bool(
            result.get(
                "primary_receipt_clear"
            )
        )
        and
        not safe_bool(
            result.get(
                "materially_truncated"
            )
        )
        and
        not safe_bool(
            result.get(
                "another_receipt_materially_visible"
            )
        )
        and
        not safe_bool(
            result.get(
                "important_content_obscured"
            )
        )
        and
        safe_bool(
            result.get(
                "merchant_readable"
            )
        )
        and
        safe_bool(
            result.get(
                "date_readable"
            )
        )
        and
        safe_bool(
            result.get(
                "amount_readable"
            )
        )
        and
        safe_bool(
            result.get(
                "final_amount_visible"
            )
        )
        and
        safe_bool(
            result.get(
                "final_amount_matches_expected"
            )
        )
        and
        safe_bool(
            result.get(
                "usable_as_evidence"
            )
        )
        and
        confidence
        >=
        0.78
    )

    issue = str(
        result.get(
            "issue",
            "",
        )
        or ""
    ).strip()

    if (
        not ok
        and
        not issue
    ):

        issue = (
            "영수증 이미지를 직접 확인해주세요."
        )

    return {
        "evidence_ok":
            ok,
        "evidence_confidence":
            confidence,
        "evidence_issue":
            issue,
    }


# =========================================================
# 22. 빠른 증빙 검증
# =========================================================

def run_evidence_validation_fast(
    df
):

    result = df.copy()

    result[
        "evidence_ok"
    ] = False

    result[
        "evidence_confidence"
    ] = 0.0

    result[
        "evidence_issue"
    ] = ""

    result[
        "evidence_manual_approved"
    ] = False

    result[
        "evidence_check_method"
    ] = ""

    auto_count = 0
    review_indices = []

    for idx, row in result.iterrows():

        if can_auto_approve_evidence(
            row
        ):

            result.at[
                idx,
                "evidence_ok",
            ] = True

            result.at[
                idx,
                "evidence_confidence",
            ] = safe_float(
                row.get(
                    "confidence",
                    0,
                )
            )

            result.at[
                idx,
                "evidence_check_method",
            ] = "자동 승인"

            auto_count += 1

        else:

            review_indices.append(
                idx
            )

    if review_indices:

        progress = st.progress(
            0
        )

        status = st.empty()

        total = len(
            review_indices
        )

        for n, idx in enumerate(
            review_indices
        ):

            row = result.loc[
                idx
            ]

            status.info(
                (
                    "추가 확인이 필요한 영수증만 재검증 중 · "
                    f"{n + 1}/{total}"
                )
            )

            if not safe_bool(
                row.get(
                    "crop_ok",
                    False,
                )
            ):

                result.at[
                    idx,
                    "evidence_issue",
                ] = (
                    row.get(
                        "crop_geometry_issue",
                        "",
                    )
                    or
                    "영수증 영역을 정확히 찾지 못했습니다."
                )

                result.at[
                    idx,
                    "evidence_check_method",
                ] = "수동 확인"

            else:

                check = (
                    validate_receipt_crop(
                        row
                    )
                )

                result.at[
                    idx,
                    "evidence_ok",
                ] = check[
                    "evidence_ok"
                ]

                result.at[
                    idx,
                    "evidence_confidence",
                ] = check[
                    "evidence_confidence"
                ]

                result.at[
                    idx,
                    "evidence_issue",
                ] = check[
                    "evidence_issue"
                ]

                result.at[
                    idx,
                    "evidence_check_method",
                ] = "AI 재검증"

            progress.progress(
                (
                    n + 1
                )
                /
                total
            )

        status.empty()
        progress.empty()

    return (
        result,
        auto_count,
        len(
            review_indices
        ),
    )


# =========================================================
# 23. 최종 정산 대상
# =========================================================

def get_final_ready_df(
    df
):

    result = add_ai_safety_status(
        df
    )

    result = add_final_evidence_status(
        result
    )

    result = add_duplicate_flags(
        result
    )

    return result[
        (
            result[
                "confirmed"
            ]
            ==
            True
        )
        &
        (
            result[
                "evidence_final_ok"
            ]
            ==
            True
        )
        &
        (
            result[
                "duplicate_suspect"
            ]
            !=
            True
        )
        &
        (
            (
                result[
                    "ai_source_safe"
                ]
                ==
                True
            )
            |
            (
                result[
                    "manual_override"
                ]
                ==
                True
            )
        )
    ].copy()


# =========================================================
# 24. Export Cache
# =========================================================

def make_export_signature(
    df
):

    parts = []

    for _, row in df.iterrows():

        parts.append(
            (
                str(
                    row.get(
                        "receipt_id",
                        "",
                    )
                ),
                str(
                    row.get(
                        "date",
                        "",
                    )
                ),
                str(
                    row.get(
                        "amount",
                        "",
                    )
                ),
                str(
                    row.get(
                        "confirmed",
                        "",
                    )
                ),
                str(
                    row.get(
                        "evidence_final_ok",
                        "",
                    )
                ),
                str(
                    row.get(
                        "manual_override",
                        "",
                    )
                ),
                str(
                    row.get(
                        "crop_mode",
                        "",
                    )
                ),
                str(
                    row.get(
                        "crop_bbox",
                        "",
                    )
                ),
                str(
                    row.get(
                        "amount_bbox",
                        "",
                    )
                ),
                str(
                    row.get(
                        "force_full_image_export",
                        "",
                    )
                ),
            )
        )

    raw = repr(
        parts
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()


def build_export_image_cache(
    df
):

    final_df = get_final_ready_df(
        df
    )

    final_df[
        "_sort"
    ] = pd.to_datetime(
        final_df[
            "date"
        ],
        errors="coerce",
    )

    final_df = (
        final_df
        .sort_values(
            "_sort"
        )
        .drop(
            columns=[
                "_sort"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    signature = make_export_signature(
        final_df
    )

    if (
        st.session_state.export_cache_signature
        ==
        signature
        and
        st.session_state.export_image_cache
    ):

        return (
            final_df,
            st.session_state.export_image_cache,
        )

    cache = {}

    total = len(
        final_df
    )

    if total == 0:

        return (
            final_df,
            cache,
        )

    progress = st.progress(
        0
    )

    status = st.empty()

    for position, (
        _,
        row,
    ) in enumerate(
        final_df.iterrows()
    ):

        receipt_id = str(
            row[
                "receipt_id"
            ]
        )

        status.info(
            (
                "증빙 이미지 준비 중 · "
                f"{position + 1}/{total}"
            )
        )

        image = get_export_receipt_image(
            row
        )

        if image is not None:

            # ---------------------------------------------
            # 출력에서 필요한 해상도보다 지나치게 큰 경우만 축소
            #
            # 비율만 축소할 뿐 절대 crop하지 않음
            # ---------------------------------------------

            max_dimension = max(
                image.size
            )

            if (
                max_dimension
                >
                2200
            ):

                scale = (
                    2200
                    /
                    max_dimension
                )

                image = image.resize(
                    (
                        max(
                            1,
                            int(
                                image.width
                                *
                                scale
                            ),
                        ),
                        max(
                            1,
                            int(
                                image.height
                                *
                                scale
                            ),
                        ),
                    ),
                    Image.Resampling.BILINEAR,
                )

            cache[
                receipt_id
            ] = image

        progress.progress(
            (
                position + 1
            )
            /
            total
        )

    progress.empty()
    status.empty()

    st.session_state.export_image_cache = (
        cache
    )

    st.session_state.export_cache_signature = (
        signature
    )

    return (
        final_df,
        cache,
    )


# =========================================================
# 25. JPG 생성
# =========================================================

def create_long_jpg(
    final_df,
    image_cache,
):

    target_width = 900

    receipts = []

    for _, row in final_df.iterrows():

        receipt_id = str(
            row[
                "receipt_id"
            ]
        )

        source = image_cache.get(
            receipt_id
        )

        if source is None:
            continue

        image = source.copy()

        # 확대는 하지 않음
        if (
            image.width
            >
            target_width
        ):

            ratio = (
                target_width
                /
                image.width
            )

            image = image.resize(
                (
                    target_width,
                    max(
                        1,
                        int(
                            image.height
                            *
                            ratio
                        ),
                    ),
                ),
                Image.Resampling.BILINEAR,
            )

        receipts.append(
            (
                row,
                image,
            )
        )

    if not receipts:
        return None

    header_height = 40
    gap = 18

    max_width = max(
        image.width
        for _, image in receipts
    )

    total_height = (
        20
        +
        sum(
            header_height
            +
            image.height
            +
            gap
            for _, image in receipts
        )
    )

    canvas = Image.new(
        "RGB",
        (
            max_width
            +
            40,
            total_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    y = 20

    for row, image in receipts:

        draw.text(
            (
                20,
                y + 8,
            ),
            (
                f"{row['date']}  "
                f"{safe_int(row['amount']):,}원  "
                f"{row['category']}"
            ),
            fill="black",
        )

        y += (
            header_height
        )

        x = (
            (
                canvas.width
                -
                image.width
            )
            //
            2
        )

        canvas.paste(
            image,
            (
                x,
                y,
            ),
        )

        y += (
            image.height
            +
            gap
        )

    output = io.BytesIO()

    canvas.save(
        output,
        format="JPEG",
        quality=91,
        optimize=False,
    )

    output.seek(
        0
    )

    return output


# =========================================================
# 26. PDF 생성
# =========================================================
#
# 한 페이지 최대 8개
#
# [1][2][3][4]
# [5][6][7][8]
#
# 핵심:
#
# 이미지 자체를 절대 crop하지 않는다.
# 칸 안에 전체 이미지가 들어오도록 축소만 한다.
#
# =========================================================

def create_pdf(
    final_df,
    image_cache,
):

    receipts = []

    for _, row in final_df.iterrows():

        receipt_id = str(
            row[
                "receipt_id"
            ]
        )

        image = image_cache.get(
            receipt_id
        )

        if image is None:
            continue

        receipts.append(
            (
                row,
                image,
            )
        )

    if not receipts:
        return None


    # -----------------------------------------------------
    # PDF LAYOUT
    # -----------------------------------------------------
    # A4 가로 / 4열 x 2행 / 최대 8개
    #
    # 핵심:
    # - 모든 영수증이 동일한 크기의 고정 슬롯(card)에 들어감
    # - 원본 종횡비 유지
    # - crop / 강제 늘이기 / 찌그러뜨리기 없음
    # - 큰 영수증은 슬롯 안에 들어오도록 축소
    # - 작은 영수증도 동일 슬롯 안에 중앙 배치
    # - 캡션, 이미지 슬롯의 상단/하단/열 폭 모두 동일
    # -----------------------------------------------------

    page_width = 1754
    page_height = 1240

    columns = 4
    rows = 2
    per_page = 8

    margin_x = 24
    margin_y = 24

    gap_x = 14
    gap_y = 20

    caption_height = 28
    caption_gap = 6

    # 각 영수증이 들어갈 고정 셀
    cell_width = int(
        (
            page_width
            -
            margin_x * 2
            -
            gap_x * (columns - 1)
        )
        /
        columns
    )

    cell_height = int(
        (
            page_height
            -
            margin_y * 2
            -
            gap_y * (rows - 1)
        )
        /
        rows
    )

    # 영수증 이미지 자체가 들어가는 고정 슬롯
    slot_padding_x = 10
    slot_padding_y = 10

    slot_width = (
        cell_width
        -
        slot_padding_x * 2
    )

    slot_height = (
        cell_height
        -
        caption_height
        -
        caption_gap
        -
        slot_padding_y * 2
    )

    pages = []


    for page_start in range(
        0,
        len(receipts),
        per_page,
    ):

        page = Image.new(
            "RGB",
            (
                page_width,
                page_height,
            ),
            "white",
        )

        draw = ImageDraw.Draw(
            page
        )

        current_receipts = receipts[
            page_start:
            page_start + per_page
        ]


        for slot_index, (
            row,
            source_image,
        ) in enumerate(
            current_receipts
        ):

            row_index = (
                slot_index
                //
                columns
            )

            column_index = (
                slot_index
                %
                columns
            )

            cell_x = (
                margin_x
                +
                column_index
                *
                (
                    cell_width
                    +
                    gap_x
                )
            )

            cell_y = (
                margin_y
                +
                row_index
                *
                (
                    cell_height
                    +
                    gap_y
                )
            )


            # ---------------------------------------------
            # 캡션 위치 고정
            # ---------------------------------------------

            caption = (
                f"{row['date']}  "
                f"{safe_int(row['amount']):,}원"
            )

            draw.text(
                (
                    cell_x + 4,
                    cell_y + 2,
                ),
                caption,
                fill="#333333",
            )


            # ---------------------------------------------
            # 고정 이미지 슬롯 생성
            # ---------------------------------------------
            # 슬롯 자체 크기는 모든 영수증에서 완전히 동일하다.
            # 영수증 원본은 비율을 유지한 채 슬롯 안에 contain한다.
            # ---------------------------------------------

            tile = Image.new(
                "RGB",
                (
                    slot_width,
                    slot_height,
                ),
                "white",
            )

            image = source_image.convert(
                "RGB"
            )

            width_ratio = (
                slot_width
                /
                max(
                    1,
                    image.width,
                )
            )

            height_ratio = (
                slot_height
                /
                max(
                    1,
                    image.height,
                )
            )

            # 이미지가 작은 경우에도 PDF에서 너무 작게 보이지 않도록
            # 슬롯 범위 안에서는 확대를 허용한다.
            scale = min(
                width_ratio,
                height_ratio,
            )

            target_width = max(
                1,
                int(
                    round(
                        image.width
                        *
                        scale
                    )
                ),
            )

            target_height = max(
                1,
                int(
                    round(
                        image.height
                        *
                        scale
                    )
                ),
            )

            if (
                target_width
                !=
                image.width

                or

                target_height
                !=
                image.height
            ):

                resized = image.resize(
                    (
                        target_width,
                        target_height,
                    ),
                    Image.Resampling.LANCZOS,
                )

            else:

                resized = image


            # 영수증을 동일 고정 슬롯 중앙에 배치
            receipt_x = max(
                0,
                (
                    slot_width
                    -
                    target_width
                )
                //
                2,
            )

            receipt_y = max(
                0,
                (
                    slot_height
                    -
                    target_height
                )
                //
                2,
            )

            tile.paste(
                resized,
                (
                    receipt_x,
                    receipt_y,
                ),
            )


            # ---------------------------------------------
            # 고정 슬롯을 페이지의 동일 위치에 붙여넣음
            # ---------------------------------------------

            slot_x = (
                cell_x
                +
                slot_padding_x
            )

            slot_y = (
                cell_y
                +
                caption_height
                +
                caption_gap
                +
                slot_padding_y
            )

            page.paste(
                tile,
                (
                    slot_x,
                    slot_y,
                ),
            )


            # ---------------------------------------------
            # 아주 연한 슬롯 테두리
            # 영수증 크기가 달라도 각 칸의 기준선이 눈에 보이도록 함
            # ---------------------------------------------

            draw.rectangle(
                (
                    slot_x,
                    slot_y,
                    slot_x + slot_width - 1,
                    slot_y + slot_height - 1,
                ),
                outline="#F2F2F2",
                width=1,
            )


            # 열 구분선
            if column_index < columns - 1:

                separator_x = (
                    cell_x
                    +
                    cell_width
                    +
                    gap_x
                    //
                    2
                )

                draw.line(
                    (
                        separator_x,
                        cell_y,
                        separator_x,
                        cell_y + cell_height,
                    ),
                    fill="#F3F3F3",
                    width=1,
                )


        # 행 구분선
        separator_y = (
            margin_y
            +
            cell_height
            +
            gap_y
            //
            2
        )

        draw.line(
            (
                margin_x,
                separator_y,
                page_width - margin_x,
                separator_y,
            ),
            fill="#ECECEC",
            width=1,
        )

        pages.append(
            page
        )


    output = io.BytesIO()

    pages[0].save(
        output,
        format="PDF",
        save_all=True,
        append_images=(
            pages[1:]
        ),
        resolution=150.0,
    )

    output.seek(
        0
    )

    return output


# =========================================================
# 27. Excel 생성
# =========================================================

def create_excel(
    final_df
):

    work_df = (
        final_df.copy()
    )

    required_columns = [
        "date",
        "merchant",
        "category",
        "amount",
        "confidence",
        "source_merchant_text",
        "source_date_text",
        "currency",
        "original_amount",
        "exchange_rate",
        "exchange_rate_date",
        "source_amount_text",
        "items_text",
        "evidence_final_ok",
        "evidence_check_method",
    ]

    for column in required_columns:

        if (
            column
            not in
            work_df.columns
        ):

            work_df[
                column
            ] = ""

    detail = work_df[
        required_columns
    ].copy()

    detail[
        "items_text"
    ] = (
        detail[
            "items_text"
        ]
        .apply(
            lambda value:
                " / ".join(
                    str(item)
                    for item in value
                )
                if
                isinstance(
                    value,
                    list,
                )
                else
                str(
                    value
                    or ""
                )
        )
    )

    detail.columns = [
        "날짜",
        "상호",
        "계정과목",
        "금액",
        "AI신뢰도",
        "영수증원문_상호",
        "영수증원문_날짜",
        "원통화",
        "외화금액",
        "적용환율_KRW",
        "환율기준일",
        "영수증원문_금액",
        "영수증원문_품목",
        "증빙상태",
        "증빙검증방식",
    ]

    detail[
        "증빙상태"
    ] = (
        detail[
            "증빙상태"
        ]
        .map(
            {
                True:
                    "정상",
                False:
                    "확인 필요",
            }
        )
    )

    summary = (
        work_df
        .groupby(
            "category"
        )[
            "amount"
        ]
        .sum()
        .reset_index()
    )

    summary.columns = [
        "계정과목",
        "합계금액",
    ]

    total = pd.DataFrame(
        {
            "구분":
                [
                    "전체 합계"
                ],
            "금액":
                [
                    safe_int(
                        work_df[
                            "amount"
                        ]
                        .fillna(
                            0
                        )
                        .sum()
                    )
                ],
        }
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        detail.to_excel(
            writer,
            index=False,
            sheet_name="영수증내역",
        )

        summary.to_excel(
            writer,
            index=False,
            sheet_name="계정과목별합계",
        )

        total.to_excel(
            writer,
            index=False,
            sheet_name="전체합계",
        )

    output.seek(
        0
    )

    return output


# =========================================================
# 28. HERO
# =========================================================

if (
    st.session_state.receipt_df
    is None
):

    st.html(
        """
<div class="tw-hero">
    <div class="tw-hero-title">
        영수증 정산,<br>
        이제 사진만 올리세요.
    </div>

    <div class="tw-hero-subtitle">
        읽고, 분류하고, 날짜순으로 정리합니다.
    </div>

    <div class="tw-hero-description">
        여러 장의 영수증을 순서 없이 업로드해도
        AI가 실제 영수증에 표시된 날짜 · 상호 · 결제금액을 읽어 정리합니다.<br>
        증빙 이미지는 원본 보존을 우선하며,
        커피 · 차 · 카페 음료는 식대와 별도로 구분합니다.
    </div>
</div>
"""
    )

else:

    month_text = html.escape(
        get_settlement_month(
            st.session_state.receipt_df
        )
    )

    st.html(
        f"""
<div class="tw-compact-header">
    <div class="tw-compact-title">
        영수증 AI 자동정산
    </div>

    <div class="tw-compact-meta">
        {month_text}
    </div>
</div>
"""
    )


# =========================================================
# 29. 업로드
# =========================================================

if (
    st.session_state.receipt_df
    is None
):

    section(
        "영수증을 선택하세요.",
        (
            "JPG 또는 PNG 이미지를 여러 장 한 번에 업로드할 수 있습니다. "
            "한 사진 안에 여러 영수증이 있어도 각각 분리해 분석합니다."
        ),
    )


files = st.file_uploader(
    "영수증 이미지",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


if files:

    uploaded_images = {}

    for file in files:

        uploaded_images[
            file.name
        ] = {
            "bytes":
                file.getvalue(),
            "mime_type":
                file.type,
        }

    st.session_state.uploaded_images = (
        uploaded_images
    )

    st.success(
        f"{len(files)}장의 이미지를 불러왔습니다."
    )

    with st.expander(
        "원본 이미지 보기",
        expanded=False,
    ):

        columns = st.columns(
            5
        )

        for n, file in enumerate(
            files
        ):

            with columns[
                n % 5
            ]:

                st.image(
                    uploaded_images[
                        file.name
                    ][
                        "bytes"
                    ],
                    use_container_width=True,
                )

                st.caption(
                    file.name
                )


# =========================================================
# 30. 분석 실행
# =========================================================

if (
    files
    and
    st.session_state.receipt_df
    is None
):

    a, b, c = st.columns(
        [
            2,
            1.2,
            2,
        ]
    )

    with b:

        analyze_button = st.button(
            "AI 분석 시작",
            type="primary",
            use_container_width=True,
        )

    if analyze_button:

        results_all = []

        status = st.empty()

        progress = st.progress(
            0
        )

        # -------------------------------------------------
        # 1차 AI 분석: 최대 4장 병렬 처리
        #
        # worker thread 안에서는 OpenAI 호출만 수행하고,
        # Streamlit session_state / 환율 처리 / 결과 저장은
        # 메인 스레드에서 처리한다.
        # -------------------------------------------------

        analysis_jobs = []
        cached_jobs = []

        for n, file in enumerate(
            files
        ):

            file_data = (
                st.session_state
                .uploaded_images[
                    file.name
                ]
            )

            cache_key = (
                get_receipt_analysis_cache_key(
                    file_data[
                        "bytes"
                    ]
                )
            )

            job = {
                "index": n,
                "file_name": file.name,
                "bytes": file_data[
                    "bytes"
                ],
                "mime_type": file_data[
                    "mime_type"
                ],
                "cache_key": cache_key,
            }

            if (
                cache_key
                in
                st.session_state.analysis_cache
            ):

                cached_jobs.append(
                    job
                )

            else:

                analysis_jobs.append(
                    job
                )

        completed_files = 0
        ordered_results = []

        # 기존 cache가 있는 이미지는 API 호출 없이 즉시 사용
        for job in cached_jobs:

            cached_results = copy.deepcopy(
                st.session_state
                .analysis_cache[
                    job[
                        "cache_key"
                    ]
                ]
            )

            ordered_results.append(
                (
                    job[
                        "index"
                    ],
                    job,
                    cached_results,
                )
            )

            completed_files += 1

            progress.progress(
                completed_files
                /
                len(files)
            )

        if analysis_jobs:

            status.info(
                (
                    "1단계 · 영수증 원문 병렬 분석 중 "
                    f"0/{len(analysis_jobs)} "
                    f"(최대 {FIRST_PASS_MAX_WORKERS}장 동시)"
                )
            )

            max_workers = min(
                FIRST_PASS_MAX_WORKERS,
                len(
                    analysis_jobs
                ),
            )

            with ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:

                future_map = {}

                for job in analysis_jobs:

                    future = executor.submit(
                        analyze_receipt_image_uncached,
                        job[
                            "file_name"
                        ],
                        job[
                            "bytes"
                        ],
                        job[
                            "mime_type"
                        ],
                    )

                    future_map[
                        future
                    ] = job

                finished_new = 0

                for future in as_completed(
                    future_map
                ):

                    job = future_map[
                        future
                    ]

                    try:

                        results = future.result()

                        # session_state cache 저장은 메인 스레드에서만
                        st.session_state.analysis_cache[
                            job[
                                "cache_key"
                            ]
                        ] = copy.deepcopy(
                            results
                        )

                        ordered_results.append(
                            (
                                job[
                                    "index"
                                ],
                                job,
                                results,
                            )
                        )

                    except Exception as e:

                        st.error(
                            (
                                f"{job['file_name']} 분석 실패: "
                                f"{e}"
                            )
                        )

                    finished_new += 1
                    completed_files += 1

                    status.info(
                        (
                            "1단계 · 영수증 원문 병렬 분석 중 "
                            f"{finished_new}/{len(analysis_jobs)} "
                            f"(최대 {FIRST_PASS_MAX_WORKERS}장 동시)"
                        )
                    )

                    progress.progress(
                        completed_files
                        /
                        len(files)
                    )

        # 병렬 완료 순서와 무관하게 업로드 순서대로 결과 처리
        ordered_results.sort(
            key=lambda item: item[
                0
            ]
        )

        for _, job, results in ordered_results:

            for position, result in enumerate(
                results
            ):

                # 외국어/외화 영수증은 원통화 금액을 보존한 뒤
                # 앱에서 전일 환율 기준 KRW로 환산한다.
                # 환율 cache/session_state는 메인 스레드에서만 처리한다.
                result = apply_currency_conversion(
                    copy.deepcopy(
                        result
                    )
                )

                result[
                    "source_file"
                ] = (
                    job[
                        "file_name"
                    ]
                )

                result[
                    "receipt_id"
                ] = (
                    f"{get_file_hash(job['bytes'])[:12]}"
                    f"_{position}"
                )

                results_all.append(
                    result
                )

        status.success(
            (
                "1단계 · 영수증 원문 분석 완료 "
                f"{len(files)}/{len(files)}"
            )
        )

        if not results_all:

            status.error(
                "분석 가능한 영수증을 찾지 못했습니다."
            )

            progress.empty()

        else:

            status.info(
                "2단계 · 영수증 원문과 결과를 대조하고 있습니다."
            )

            df = pd.DataFrame(
                results_all
            )

            required = [
                "receipt_id",
                "date",
                "merchant",
                "source_merchant_text",
                "source_date_text",
                "currency",
                "original_amount",
                "amount",
                "source_amount_text",
                "exchange_rate",
                "exchange_rate_date",
                "exchange_rate_status",
                "fx_conversion_ok",
                "fx_review_reason",
                "items_text",
                "category",
                "confidence",
                "needs_review",
                "review_question",
                "complete_visible",
                "bbox",
                "amount_bbox",
                "receipt_quad",
                "source_file",
                "force_full_image_export",
            ]

            for column in required:

                if (
                    column
                    not in
                    df.columns
                ):

                    if (
                        column
                        ==
                        "items_text"
                    ):

                        df[
                            column
                        ] = [
                            []
                            for _
                            in range(
                                len(df)
                            )
                        ]

                    elif (
                        column
                        ==
                        "force_full_image_export"
                    ):

                        df[
                            column
                        ] = False

                    else:

                        df[
                            column
                        ] = None

            df[
                "amount"
            ] = pd.to_numeric(
                df[
                    "amount"
                ],
                errors="coerce",
            )

            df[
                "confidence"
            ] = pd.to_numeric(
                df[
                    "confidence"
                ],
                errors="coerce",
            )

            df[
                "original_amount"
            ] = pd.to_numeric(
                df[
                    "original_amount"
                ],
                errors="coerce",
            )

            df[
                "exchange_rate"
            ] = pd.to_numeric(
                df[
                    "exchange_rate"
                ],
                errors="coerce",
            )

            df[
                "needs_review"
            ] = (
                df[
                    "needs_review"
                ]
                .apply(
                    safe_bool
                )
            )

            df[
                "complete_visible"
            ] = (
                df[
                    "complete_visible"
                ]
                .apply(
                    safe_bool
                )
            )

            df[
                "manual_override"
            ] = False

            df = add_ai_safety_status(
                df
            )

            # 핵심값이 정상이라면 AI가 needs_review=true를 반환했다는 이유만으로
            # 자동 확정을 막지 않는다.
            # 실제 확인 필요 여부는 원문 안전성, 핵심값, 신뢰도, 증빙 상태로 판단한다.
            df[
                "confirmed"
            ] = (
                (
                    df[
                        "confidence"
                    ]
                    >=
                    0.90
                )
                &
                (
                    df[
                        "ai_source_safe"
                    ]
                    ==
                    True
                )
                &
                df[
                    "date"
                ].notna()
                &
                df[
                    "merchant"
                ].notna()
                &
                (
                    df[
                        "amount"
                    ]
                    >
                    0
                )
            )

            df[
                "data_manual_approved"
            ] = False

            # 핵심:
            # 단일 영수증은 여기서 무조건 full_image
            df = prepare_crop_boxes(
                df
            )

            status.info(
                (
                    "3단계 · 확실한 증빙은 자동 승인하고 "
                    "필요한 항목만 추가 확인합니다."
                )
            )

            (
                df,
                auto_count,
                recheck_count,
            ) = (
                run_evidence_validation_fast(
                    df
                )
            )

            df = add_ai_safety_status(
                df
            )

            df = add_duplicate_flags(
                df
            )

            df = add_final_evidence_status(
                df
            )

            df[
                "_sort"
            ] = pd.to_datetime(
                df[
                    "date"
                ],
                errors="coerce",
            )

            df = (
                df
                .sort_values(
                    "_sort",
                    na_position="last",
                )
                .drop(
                    columns=[
                        "_sort"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            st.session_state.receipt_df = (
                df
            )

            st.session_state.deleted_row = (
                None
            )

            invalidate_generated_files()

            progress.empty()

            status.success(
                (
                    f"분석 완료 · 자동 승인 {auto_count}건"
                    +
                    (
                        f" · 추가 검증 {recheck_count}건"
                        if recheck_count
                        else ""
                    )
                )
            )

            st.rerun()


# =========================================================
# 31. 결과 화면
# =========================================================

if (
    st.session_state.receipt_df
    is not None
):

    if st.session_state.flash_message:

        flash = (
            st.session_state
            .flash_message
        )

        st.toast(
            flash[
                "message"
            ],
            icon=(
                "✅"
                if
                flash[
                    "type"
                ]
                ==
                "success"
                else
                "⚠️"
            ),
        )

        st.session_state.flash_message = (
            None
        )

    df = (
        st.session_state
        .receipt_df
        .copy()
    )

    df = add_ai_safety_status(
        df
    )

    df = add_final_evidence_status(
        df
    )

    df = add_duplicate_flags(
        df
    )

    st.session_state.receipt_df = (
        df
    )

    review_mask = df.apply(
        row_needs_attention,
        axis=1,
    )

    review_df = df[
        review_mask
    ].copy()

    final_df = get_final_ready_df(
        df
    )

    if review_df.empty:

        st.html(
            f"""
<div class="tw-ready">
    <div class="tw-ready-label">
        정산 준비 완료
    </div>
    <div class="tw-ready-title">
        {len(df)}장의 영수증을 정리했습니다.
    </div>
    <div class="tw-ready-description">
        확인 필요한 항목이 없습니다.
    </div>
</div>
"""
        )

    else:

        st.html(
            f"""
<div class="tw-ready">
    <div class="tw-ready-label">
        AI 분석 완료
    </div>
    <div class="tw-ready-title">
        {len(df)}장의 영수증을 읽었습니다.
    </div>
    <div class="tw-ready-description">
        최종 반영 가능 {len(final_df)}건 · 확인 필요 {len(review_df)}건
    </div>
</div>
"""
        )


    # =====================================================
    # KPI
    # =====================================================

    k1, k2, k3, k4 = st.columns(
        4
    )

    with k1:

        st.metric(
            "전체 영수증",
            f"{len(df)}건",
        )

    with k2:

        st.metric(
            "최종 반영 가능",
            f"{len(final_df)}건",
        )

    with k3:

        st.metric(
            "확인 필요",
            f"{len(review_df)}건",
        )

    with k4:

        st.metric(
            "정산 합계",
            (
                f"{safe_int(final_df['amount'].fillna(0).sum()):,}원"
            ),
        )


    # =====================================================
    # 분석 결과
    # =====================================================

    section(
        "분석 결과",
        "원문 대조가 끝난 영수증을 날짜순으로 정리합니다.",
    )

    f1, f2, f3 = st.columns(
        [
            2.4,
            1,
            1,
        ]
    )

    with f1:

        search_text = st.text_input(
            "검색",
            placeholder="상호 검색",
            label_visibility="collapsed",
        )

    with f2:

        category_filter = st.selectbox(
            "계정과목",
            [
                "전체",
                *CATEGORIES,
            ],
            label_visibility="collapsed",
        )

    with f3:

        status_filter = st.selectbox(
            "상태",
            [
                "전체",
                "확인 필요",
                "정상",
                "중복 의심",
                "원문 대조 필요",
            ],
            label_visibility="collapsed",
        )

    filtered_df = (
        df.copy()
    )

    if search_text:

        filtered_df = filtered_df[
            filtered_df[
                "merchant"
            ]
            .fillna("")
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False,
            )
        ]

    if (
        category_filter
        !=
        "전체"
    ):

        filtered_df = filtered_df[
            filtered_df[
                "category"
            ]
            ==
            category_filter
        ]

    if (
        status_filter
        ==
        "확인 필요"
    ):

        filtered_df = filtered_df[
            filtered_df.apply(
                row_needs_attention,
                axis=1,
            )
        ]

    elif (
        status_filter
        ==
        "정상"
    ):

        filtered_df = filtered_df[
            ~filtered_df.apply(
                row_needs_attention,
                axis=1,
            )
        ]

    elif (
        status_filter
        ==
        "중복 의심"
    ):

        filtered_df = filtered_df[
            filtered_df[
                "duplicate_suspect"
            ]
            ==
            True
        ]

    elif (
        status_filter
        ==
        "원문 대조 필요"
    ):

        filtered_df = filtered_df[
            (
                filtered_df[
                    "ai_source_safe"
                ]
                !=
                True
            )
            &
            (
                filtered_df[
                    "manual_override"
                ]
                !=
                True
            )
        ]

    display = filtered_df[
        [
            "date",
            "merchant",
            "amount",
            "category",
            "confidence",
        ]
    ].copy()

    display[
        "상태"
    ] = filtered_df.apply(
        row_status_label,
        axis=1,
    )

    display[
        "다음 조치"
    ] = filtered_df.apply(
        row_next_action,
        axis=1,
    )

    display.columns = [
        "날짜",
        "상호",
        "금액",
        "계정과목",
        "AI신뢰도",
        "상태",
        "다음 조치",
    ]

    display[
        "금액"
    ] = (
        display[
            "금액"
        ]
        .fillna(0)
        .astype(int)
        .map(
            lambda x:
                f"{x:,}원"
        )
    )

    display[
        "AI신뢰도"
    ] = (
        display[
            "AI신뢰도"
        ]
        .fillna(0)
        .map(
            lambda x:
                f"{x * 100:.0f}%"
        )
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=min(
            560,
            38
            *
            (
                len(display)
                +
                1
            ),
        ),
    )


    # =====================================================
    # 상세 확인
    # =====================================================

    section(
        "상세 확인",
        (
            "증빙 보존을 최우선으로 처리합니다. "
            "사진 한 장에 영수증이 하나면 원본 전체 이미지를 사용합니다."
        ),
    )

    filtered_review = (
        filtered_df[
            filtered_df.apply(
                row_needs_attention,
                axis=1,
            )
        ]
        .copy()
    )

    filtered_normal = (
        filtered_df[
            ~filtered_df.apply(
                row_needs_attention,
                axis=1,
            )
        ]
        .copy()
    )


    def render_receipts(
        detail_df,
        attention=False,
    ):

        for _, row in detail_df.iterrows():

            receipt_id = str(
                row[
                    "receipt_id"
                ]
            )

            merchant = str(
                row.get(
                    "merchant",
                    "",
                )
                or ""
            )

            date = str(
                row.get(
                    "date",
                    "",
                )
                or ""
            )

            amount = safe_int(
                row.get(
                    "amount",
                    0,
                )
            )

            evidence_ok = safe_bool(
                row.get(
                    "evidence_final_ok",
                    False,
                )
            )

            ai_source_safe = safe_bool(
                row.get(
                    "ai_source_safe",
                    False,
                )
            )

            manual_override = safe_bool(
                row.get(
                    "manual_override",
                    False,
                )
            )

            title = (
                (
                    "🚨 확인 필요 | "
                    if attention
                    else
                    "✓ "
                )
                +
                f"{date} "
                +
                f"{merchant} "
                +
                f"{amount:,}원"
            )

            with st.expander(
                title,
                expanded=False,
            ):

                left, right = st.columns(
                    [
                        1.6,
                        1,
                    ],
                    gap="large",
                )

                with left:

                    st.subheader(
                        merchant
                    )

                    st.caption(
                        date
                    )

                    st.header(
                        f"{amount:,}원"
                    )

                    st.write(
                        row.get(
                            "category",
                            "",
                        )
                    )

                    currency = normalize_currency_code(
                        row.get("currency"),
                        row.get("source_amount_text", ""),
                    )

                    if currency != "KRW":
                        original_amount = safe_float(
                            row.get("original_amount", 0)
                        )
                        exchange_rate = safe_float(
                            row.get("exchange_rate", 0)
                        )
                        exchange_rate_date = str(
                            row.get("exchange_rate_date", "")
                            or ""
                        )

                        st.caption(
                            f"외화 결제 {original_amount:,.2f} {currency} · "
                            f"1 {currency} = {exchange_rate:,.6f}원 · "
                            f"환율 기준 {exchange_rate_date or '확인 필요'}"
                        )

                    crop_mode = str(
                        row.get(
                            "crop_mode",
                            "",
                        )
                        or ""
                    )

                    if (
                        crop_mode
                        ==
                        "full_image"
                    ):

                        st.caption(
                            "증빙 이미지: 원본 전체 보존"
                        )

                    elif (
                        crop_mode
                        ==
                        "safe_bbox"
                    ):

                        st.caption(
                            "증빙 이미지: 복수 영수증 안전 분리"
                        )

                    st.caption(
                        (
                            "AI 신뢰도 "
                            f"{safe_float(row.get('confidence', 0)) * 100:.0f}%"
                            " · "
                            f"{get_confidence_state(row.get('confidence', 0))}"
                        )
                    )

                    show_source_data(
                        row
                    )

                    if (
                        not ai_source_safe
                        and
                        not manual_override
                    ):

                        show_safety_warning(
                            row.get(
                                "ai_safety_issue",
                                "",
                            )
                            or
                            "AI 결과와 원문을 확인해주세요."
                        )

                    elif manual_override:

                        show_safety_warning(
                            "사용자가 직접 확인하여 승인한 항목입니다."
                        )

                    if attention:

                        reasons = (
                            row_attention_reasons(
                                row
                            )
                        )

                        actions = []

                        if (
                            not ai_source_safe
                            and
                            not manual_override
                        ):

                            actions.append(
                                "원문 대조"
                            )

                        if not evidence_ok:

                            actions.append(
                                "이미지 확인"
                            )

                        if not safe_bool(
                            row.get(
                                "confirmed",
                                False,
                            )
                        ):

                            actions.append(
                                "내용 확인"
                            )

                        if safe_bool(
                            row.get(
                                "duplicate_suspect",
                                False,
                            )
                        ):

                            actions.append(
                                "중복 확인"
                            )

                        actions.append(
                            "불필요하면 삭제"
                        )

                        show_attention(
                            (
                                " / ".join(
                                    reasons
                                )
                                if reasons
                                else
                                "추가 확인 필요"
                            ),
                            " · ".join(
                                actions
                            ),
                        )

                    if evidence_ok:

                        show_ok()

                    elif str(
                        row.get(
                            "evidence_issue",
                            "",
                        )
                        or ""
                    ).strip():

                        show_neutral(
                            (
                                "증빙 확인 필요 · "
                                +
                                str(
                                    row.get(
                                        "evidence_issue",
                                        "",
                                    )
                                )
                            )
                        )

                    buttons = st.columns(
                        4
                    )

                    with buttons[0]:

                        if safe_bool(
                            row.get(
                                "confirmed",
                                False,
                            )
                        ):

                            st.button(
                                "확정 완료",
                                key=
                                    f"confirmed_{receipt_id}",
                                disabled=True,
                                use_container_width=True,
                            )

                        else:

                            if st.button(
                                "직접 확인 후 확정",
                                key=
                                    f"confirm_{receipt_id}",
                                type="primary",
                                use_container_width=True,
                            ):

                                if confirm_receipt(
                                    receipt_id
                                ):

                                    st.rerun()

                    with buttons[1]:

                        if evidence_ok:

                            st.button(
                                "증빙 확인 완료",
                                key=
                                    f"evidence_done_{receipt_id}",
                                disabled=True,
                                use_container_width=True,
                            )

                        else:

                            if st.button(
                                "증빙 정상 승인",
                                key=
                                    f"evidence_{receipt_id}",
                                use_container_width=True,
                            ):

                                if approve_evidence(
                                    receipt_id
                                ):

                                    st.rerun()

                    with buttons[2]:

                        if st.button(
                            "수정",
                            key=
                                f"edit_{receipt_id}",
                            use_container_width=True,
                        ):

                            st.session_state.edit_receipt_id = (
                                receipt_id
                            )

                            st.rerun()

                    with buttons[3]:

                        if st.button(
                            "삭제",
                            key=
                                f"delete_{receipt_id}",
                            use_container_width=True,
                        ):

                            original = (
                                st.session_state
                                .receipt_df
                                .copy()
                            )

                            deleted = original[
                                original[
                                    "receipt_id"
                                ]
                                ==
                                receipt_id
                            ].copy()

                            remaining = original[
                                original[
                                    "receipt_id"
                                ]
                                !=
                                receipt_id
                            ].copy()

                            st.session_state.deleted_row = (
                                deleted
                            )

                            st.session_state.receipt_df = (
                                remaining
                                .reset_index(
                                    drop=True
                                )
                            )

                            invalidate_generated_files()

                            set_flash(
                                "영수증 1건을 삭제했습니다."
                            )

                            st.rerun()


                    # -------------------------------------
                    # 수정
                    # -------------------------------------

                    if (
                        st.session_state.edit_receipt_id
                        ==
                        receipt_id
                    ):

                        st.divider()

                        st.subheader(
                            "내용 수정"
                        )

                        st.info(
                            (
                                "이 값은 사용자가 직접 수정하는 값입니다. "
                                "저장을 눌러야 반영됩니다."
                            )
                        )

                        with st.form(
                            f"edit_form_{receipt_id}"
                        ):

                            edit_date = st.text_input(
                                "날짜",
                                value=date,
                            )

                            edit_merchant = st.text_input(
                                "상호",
                                value=merchant,
                            )

                            edit_amount = st.number_input(
                                "금액",
                                min_value=0,
                                value=amount,
                                step=100,
                            )

                            category = row.get(
                                "category",
                                "기타",
                            )

                            if (
                                category
                                not in
                                CATEGORIES
                            ):

                                category = (
                                    "기타"
                                )

                            edit_category = st.selectbox(
                                "계정과목",
                                CATEGORIES,
                                index=
                                    CATEGORIES.index(
                                        category
                                    ),
                            )

                            edit_confirmed = st.checkbox(
                                "저장 후 확정",
                                value=True,
                            )

                            fc1, fc2 = st.columns(
                                2
                            )

                            with fc1:

                                save_button = (
                                    st.form_submit_button(
                                        "수정 내용 저장",
                                        type="primary",
                                        use_container_width=True,
                                    )
                                )

                            with fc2:

                                cancel_button = (
                                    st.form_submit_button(
                                        "취소",
                                        use_container_width=True,
                                    )
                                )

                        if save_button:

                            updated = (
                                st.session_state
                                .receipt_df
                                .copy()
                            )

                            mask = (
                                updated[
                                    "receipt_id"
                                ]
                                ==
                                receipt_id
                            )

                            updated.loc[
                                mask,
                                "date",
                            ] = edit_date

                            updated.loc[
                                mask,
                                "merchant",
                            ] = edit_merchant

                            updated.loc[
                                mask,
                                "amount",
                            ] = edit_amount

                            updated.loc[
                                mask,
                                "category",
                            ] = edit_category

                            updated.loc[
                                mask,
                                "confirmed",
                            ] = edit_confirmed

                            updated.loc[
                                mask,
                                "data_manual_approved",
                            ] = edit_confirmed

                            updated.loc[
                                mask,
                                "manual_override",
                            ] = True

                            if edit_confirmed:

                                updated.loc[
                                    mask,
                                    "needs_review",
                                ] = False

                                updated.loc[
                                    mask,
                                    "review_question",
                                ] = ""

                            updated = add_ai_safety_status(
                                updated
                            )

                            updated = add_duplicate_flags(
                                updated
                            )

                            updated = add_final_evidence_status(
                                updated
                            )

                            updated[
                                "_sort"
                            ] = pd.to_datetime(
                                updated[
                                    "date"
                                ],
                                errors="coerce",
                            )

                            updated = (
                                updated
                                .sort_values(
                                    "_sort",
                                    na_position="last",
                                )
                                .drop(
                                    columns=[
                                        "_sort"
                                    ]
                                )
                                .reset_index(
                                    drop=True
                                )
                            )

                            st.session_state.receipt_df = (
                                updated
                            )

                            st.session_state.edit_receipt_id = (
                                None
                            )

                            invalidate_generated_files()

                            set_flash(
                                "수정 내용을 저장했습니다."
                            )

                            st.rerun()

                        if cancel_button:

                            st.session_state.edit_receipt_id = (
                                None
                            )

                            st.rerun()


                # -----------------------------------------
                # 영수증 이미지
                # -----------------------------------------

                with right:

                    display_image = get_display_receipt_image(
                        row
                    )

                    if display_image is not None:

                        st.image(
                            display_image,
                            use_container_width=True,
                        )

                        crop_mode = str(
                            row.get(
                                "crop_mode",
                                "",
                            )
                            or
                            ""
                        )

                        if (
                            crop_mode
                            !=
                            "full_image"
                            and
                            not safe_bool(
                                row.get(
                                    "evidence_ok",
                                    False,
                                )
                            )
                        ):

                            st.caption(
                                "자동 crop이 최종 증빙 조건을 통과하지 못해 원본 전체 이미지를 표시하고 있습니다. 직접 확정하면 이 원본 전체가 PDF/JPG 증빙에 사용됩니다."
                            )

                    else:

                        st.warning(
                            "원본 이미지 데이터 자체를 불러올 수 없습니다."
                        )


    if not filtered_review.empty:

        st.error(
            (
                f"확인 필요한 영수증 "
                f"{len(filtered_review)}건입니다."
            )
        )

        render_receipts(
            filtered_review,
            attention=True,
        )


    if not filtered_normal.empty:

        section(
            "정상 영수증",
            "원문 대조 또는 사용자 확인이 완료된 항목입니다.",
        )

        render_receipts(
            filtered_normal,
            attention=False,
        )


    # =====================================================
    # 삭제 되돌리기
    # =====================================================

    if (
        st.session_state.deleted_row
        is not None
    ):

        u1, u2 = st.columns(
            [
                5,
                1,
            ]
        )

        with u1:

            st.caption(
                "최근 영수증 1건을 삭제했습니다."
            )

        with u2:

            if st.button(
                "되돌리기",
                use_container_width=True,
            ):

                restored = pd.concat(
                    [
                        st.session_state.receipt_df,
                        st.session_state.deleted_row,
                    ],
                    ignore_index=True,
                )

                restored = add_ai_safety_status(
                    restored
                )

                restored = add_duplicate_flags(
                    restored
                )

                restored = add_final_evidence_status(
                    restored
                )

                st.session_state.receipt_df = (
                    restored
                )

                st.session_state.deleted_row = (
                    None
                )

                invalidate_generated_files()

                st.rerun()


    # =====================================================
    # 정산 요약
    # =====================================================

    latest_df = (
        st.session_state
        .receipt_df
        .copy()
    )

    final_df = get_final_ready_df(
        latest_df
    )

    if not final_df.empty:

        section(
            "정산 요약",
            (
                "원문 대조 또는 사용자 확인이 완료된 "
                "영수증만 합산합니다."
            ),
        )

        summary = (
            final_df
            .groupby(
                "category"
            )[
                "amount"
            ]
            .sum()
            .reset_index()
        )

        summary.columns = [
            "계정과목",
            "합계금액",
        ]

        summary[
            "합계금액"
        ] = (
            summary[
                "합계금액"
            ]
            .fillna(0)
            .astype(int)
            .map(
                lambda x:
                    f"{x:,}원"
            )
        )

        s1, s2 = st.columns(
            [
                2.3,
                1,
            ]
        )

        with s1:

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True,
            )

        with s2:

            st.metric(
                "전체 총합",
                (
                    f"{safe_int(final_df['amount'].fillna(0).sum()):,}원"
                ),
            )


    # =====================================================
    # 최종 증빙 미리보기
    # =====================================================

    if not final_df.empty:

        with st.expander(
            "최종 증빙 영수증 이미지 확인",
            expanded=False,
        ):

            preview = (
                final_df.copy()
            )

            preview[
                "_sort"
            ] = pd.to_datetime(
                preview[
                    "date"
                ],
                errors="coerce",
            )

            preview = (
                preview
                .sort_values(
                    "_sort"
                )
            )

            preview_cols = st.columns(
                4
            )

            for n, (
                _,
                row,
            ) in enumerate(
                preview.iterrows()
            ):

                with preview_cols[
                    n % 4
                ]:

                    crop = get_export_receipt_image(
                        row
                    )

                    if crop is not None:

                        st.image(
                            crop,
                            use_container_width=True,
                        )

                    st.caption(
                        (
                            f"{row['date']} · "
                            f"{safe_int(row['amount']):,}원 · "
                            f"{row['category']}"
                        )
                    )


    # =====================================================
    # 파일 생성
    # =====================================================

    if not final_df.empty:

        section(
            "완료",
            (
                "PDF는 한 페이지에 최대 8개의 영수증을 "
                "4열 × 2행으로 배치하며, "
                "영수증 원본 비율과 전체 길이를 유지합니다."
            ),
        )

        settlement_month = (
            get_settlement_month(
                final_df
            )
        )

        if (
            st.session_state.generated_files
            is None
        ):

            g1, g2, g3 = st.columns(
                [
                    2,
                    1.2,
                    2,
                ]
            )

            with g2:

                if st.button(
                    "정산 파일 생성",
                    type="primary",
                    use_container_width=True,
                ):

                    with st.spinner(
                        "정산 파일을 생성하고 있습니다."
                    ):

                        current_df = (
                            st.session_state
                            .receipt_df
                            .copy()
                        )

                        export_df, image_cache = (
                            build_export_image_cache(
                                current_df
                            )
                        )

                        jpg = create_long_jpg(
                            export_df,
                            image_cache,
                        )

                        pdf = create_pdf(
                            export_df,
                            image_cache,
                        )

                        excel = create_excel(
                            export_df
                        )

                        st.session_state.generated_files = {
                            "jpg":
                                (
                                    jpg.getvalue()
                                    if jpg
                                    else None
                                ),

                            "pdf":
                                (
                                    pdf.getvalue()
                                    if pdf
                                    else None
                                ),

                            "excel":
                                (
                                    excel.getvalue()
                                    if excel
                                    else None
                                ),
                        }

                    set_flash(
                        "정산 파일 생성이 완료되었습니다."
                    )

                    st.rerun()

        else:

            st.success(
                "정산 파일 생성이 완료되었습니다."
            )

            ready = (
                st.session_state
                .generated_files
            )

            d1, d2, d3 = st.columns(
                3
            )

            with d1:

                if ready[
                    "jpg"
                ]:

                    st.download_button(
                        "JPG 다운로드",
                        data=
                            ready[
                                "jpg"
                            ],
                        file_name=
                            f"{settlement_month}_영수증_날짜순.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )

            with d2:

                if ready[
                    "pdf"
                ]:

                    st.download_button(
                        "PDF 다운로드",
                        data=
                            ready[
                                "pdf"
                            ],
                        file_name=
                            f"{settlement_month}_영수증_증빙.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            with d3:

                if ready[
                    "excel"
                ]:

                    st.download_button(
                        "Excel 다운로드",
                        data=
                            ready[
                                "excel"
                            ],
                        file_name=
                            f"{settlement_month}_경비정산.xlsx",
                        mime=
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

            r1, r2, r3 = st.columns(
                [
                    2,
                    1,
                    2,
                ]
            )

            with r2:

                if st.button(
                    "파일 다시 생성",
                    use_container_width=True,
                ):

                    # 이미지 캐시는 유지
                    # PDF/JPG/Excel만 다시 생성
                    st.session_state.generated_files = (
                        None
                    )

                    st.rerun()


    # =====================================================
    # 새 정산
    # =====================================================

    st.divider()

    reset1, reset2, reset3 = st.columns(
        [
            2,
            1,
            2,
        ]
    )

    with reset2:

        if st.button(
            "새 정산 시작",
            use_container_width=True,
        ):

            st.session_state.receipt_df = None
            st.session_state.uploaded_images = {}
            st.session_state.deleted_row = None
            st.session_state.edit_receipt_id = None
            st.session_state.generated_files = None
            st.session_state.flash_message = None
            st.session_state.action_errors = {}

            st.session_state.export_image_cache = {}
            st.session_state.export_cache_signature = None

            st.rerun()


# =========================================================
# 32. Footer
# =========================================================

st.html(
    """
<div class="tw-footer">
    Receipt AI · Internal Expense Support Tool
</div>
"""
)
