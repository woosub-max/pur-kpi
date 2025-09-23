# -*- coding: utf-8 -*-
"""
미입고 KPI 대시보드 (Streamlit · Robust Reader · 업로드 히스토리 '영구 저장' + 삭제 버튼)
- 더존 발주현황 업로드 → KPI/차트/필터 → 엑셀 보고서(요약+상세+원본) 다운로드
- 업로드 히스토리: ./uploads 폴더에 파일을 저장해 '재접속해도' 목록에서 선택/삭제 가능
- 다중 업로드 가능(동일 파일명은 최신본으로 치환), 선택 삭제 버튼 제공
- 본문 로직은 기존 test2.py를 토대로 보강(파일 저장/불러오기만 추가)
"""

import io, os, csv, calendar, time, json, re
from pathlib import Path
from datetime import date, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def halt_app():
    """Safely stop execution both in Streamlit runtime and bare execution."""
    try:
        st.stop()
    except Exception:
        raise SystemExit(0)

# ───────────────────────── 기본 설정 ─────────────────────────
st.set_page_config(page_title="미입고 KPI 대시보드(Pro)", page_icon="📦", layout="wide")

ROOT_DIR   = Path(__file__).parent if "__file__" in globals() else Path(".")
UPLOAD_DIR = ROOT_DIR / "uploads"
MANIFEST   = UPLOAD_DIR / "manifest.json"
UPLOAD_DIR.mkdir(exist_ok=True)

def month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])

def add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))

# 더존 컬럼 후보 & 상태 라벨(사이드바에서 바꿀 수 있음)
COL = {
    "group":   ["제품군(1)","제품군","품목군","대분류"],
    "po_no":   ["발주번호","PO번호","PO_NO","주문번호"],
    "po_date": ["발주일자","발주일","PO_DATE","주문일자"],
    "item":    ["품목명","품목","내역","ITEM_NAME"],
    "vendor":  ["거래처명","거래처","공급사"],
    "pgroup":  ["구매그룹","구매그룹명","구매 그룹","Buyer Group","구매그룹코드"],
    "due":     ["발주납기일자","납기일자","납기일","DUE_DATE"],
    "rcv_date":["입고일자","입고일","RCV_DATE"],
    "po_qty":  ["발주수량","PO수량","발주 수량"],
    "rcv_qty": ["입고수량","RCV수량","검사합격수량","실입고수량"],
    "status":  ["입고구분","입고상태","입고상태명","진행상태"],
}
COMPLETE = {"입고완료","완료"}
PARTIAL  = {"부분입고","부분","부분완료"}
OPEN     = {"미입고","대기","미완료"}
OPEN_OR_PARTIAL = OPEN | PARTIAL

# ──────────────── 업로드 히스토리 (영구 저장) ───────────────
def _slug(s: str) -> str:
    s = re.sub(r"[^\w.\-가-힣 ]+", "_", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:140] if len(s) > 140 else s

def _load_manifest() -> list:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text("utf-8"))
        except Exception:
            return []
    return []

def _save_manifest(rows: list):
    MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def list_uploads() -> list:
    rows = _load_manifest()
    rows = [r for r in rows if (UPLOAD_DIR / r["path"]).exists()]  # 유효한 항목만
    rows.sort(key=lambda r: r["uploaded_at"], reverse=True)        # 최신 먼저
    _save_manifest(rows)
    return rows

def save_upload(file) -> dict:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = f"{ts}__{_slug(file.name)}"
    path = UPLOAD_DIR / safe_name
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    rows = _load_manifest()
    # 동일 원본 이름은 최신본으로 치환(이름 기준 중복 제거)
    rows = [r for r in rows if r["name"] != file.name]
    rec = {
        "id": f"{ts}_{int(time.time()*1000)}",
        "name": file.name,
        "path": safe_name,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rows.append(rec)
    _save_manifest(rows)
    return rec

def delete_upload(rec_id: str):
    rows = _load_manifest()
    remain = []
    for r in rows:
        if r["id"] == rec_id:
            try:
                (UPLOAD_DIR / r["path"]).unlink(missing_ok=True)
            except Exception:
                pass
        else:
            remain.append(r)
    _save_manifest(remain)

# ──────────────── 업로드 파일 판독 ────────────────
@st.cache_data(show_spinner=False)
def read_any(_uploaded_file) -> pd.DataFrame:
    """UploadedFile → DataFrame (극내구성 판독기)"""
    import io, csv
    from bs4 import BeautifulSoup

    raw = _uploaded_file.read()
    if not raw:
        raise RuntimeError("빈 파일입니다.")
    name = _uploaded_file.name.lower()
    head = raw[:8]

    def is_zip(h: bytes):   return h.startswith(b"PK\x03\x04")
    def is_ole(h: bytes):   return h.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")

    # 1) pandas + openpyxl
    if is_zip(head) or name.endswith((".xlsx",".xlsm",".xltx")):
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass
        # 2) openpyxl 수동 파싱
        try:
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
            ws = wb[wb.sheetnames[0]]
            rows = list(ws.iter_rows(values_only=True))
            header_idx = None
            for i, r in enumerate(rows[:20]):
                non_empty = sum(1 for x in r if (x is not None and str(x).strip() != ""))
                if non_empty >= 2:
                    header_idx = i; break
            if header_idx is None:
                header_idx = 0
            header = [str(c).strip() if c is not None else "" for c in rows[header_idx]]
            data = rows[header_idx+1:]
            df = pd.DataFrame(data, columns=header)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass

    # 3) xls / xlsb
    if is_ole(head) or name.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="xlrd")  # xlrd==1.2.0
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass
    if name.endswith(".xlsb"):
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="pyxlsb")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass

    # 4) HTML / MHTML / XML
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    if ("<html" in text.lower() or "mime-version" in text.lower()
        or name.endswith((".htm",".html",".xml",".mht",".mhtml"))):
        try:
            tables = pd.read_html(io.StringIO(text))
            if tables:
                df = tables[0]; df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            try:
                soup = BeautifulSoup(text, "html.parser")
                table = soup.find("table")
                if table:
                    tables = pd.read_html(str(table))
                    if tables:
                        df = tables[0]; df.columns = [str(c).strip() for c in df.columns]
                        return df
            except Exception:
                pass

    # 5) CSV (인코딩/구분자/sep= 자동)
    try:
        head_txt = raw[:256].decode("utf-8-sig", errors="ignore")
        if head_txt.lower().startswith("sep="):
            sep = head_txt.split("=",1)[1].strip()[:1] or ","
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig",
                             sep=sep, skiprows=1, engine="python", on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            return df
    except Exception:
        pass

    encodings = ["utf-8-sig","cp949","ms949","euc-kr","utf-8"]
    try:
        sample = raw[:4096].decode("utf-8", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
            delims = [dialect.delimiter]
        except Exception:
            delims = [",",";","\t","|"]
    except Exception:
        delims = [",",";","\t","|"]

    for enc in encodings:
        for delim in delims:
            for skip in range(0, 6):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=delim,
                                     skiprows=skip, engine="python", on_bad_lines="skip")
                    if df.shape[1] >= 2:
                        df.columns = [str(c).strip() for c in df.columns]
                        return df
                except Exception:
                    continue

    sig = head.hex(" ")
    raise RuntimeError(
        "파일 형식을 자동으로 판독하지 못했습니다.\n"
        f"- 파일명: { _uploaded_file.name }\n"
        f"- 헤더 시그니처: {sig}\n"
        "엑셀에서 '다른 이름으로 저장' → .xlsx (통합문서)로 저장하거나, 더존에서 CSV로 다시 추출해 업로드 해주세요.\n"
        "암호화된(비밀번호) 통합문서는 지원되지 않습니다."
    )

# ─────────────── 표준화 & 미입고 로직 ───────────────
def _pick(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def build_base(df: pd.DataFrame) -> pd.DataFrame:
    m = {k: _pick(df, v) for k, v in COL.items()}
    if not m["po_qty"]:
        df["발주수량"] = 0; m["po_qty"] = "발주수량"
    if not m["rcv_qty"]:
        df["입고수량"] = 0; m["rcv_qty"] = "입고수량"

    base = pd.DataFrame()
    base["제품군"]      = (df[m["group"]].astype(str).str.strip().replace({"": "미지정"}).fillna("미지정")
                          if m["group"] else "미지정")
    base["발주번호"]    = df[m["po_no"]]   if m["po_no"]   else ""
    base["품목명"]      = df[m["item"]]    if m["item"]    else ""
    base["거래처명"]    = df[m["vendor"]]  if m["vendor"]  else ""
    if m["pgroup"]:
        pg = df[m["pgroup"]].fillna("").astype(str).str.strip()
        base["구매그룹"] = pg
    else:
        base["구매그룹"] = ""
    base["발주일자"]    = pd.to_datetime(df[m["po_date"]],  errors="coerce") if m["po_date"]  else pd.NaT
    base["발주납기일자"]= pd.to_datetime(df[m["due"]],      errors="coerce") if m["due"]      else pd.NaT
    base["입고일자"]    = pd.to_datetime(df[m["rcv_date"]], errors="coerce") if m["rcv_date"] else pd.NaT
    base["발주수량"]    = _to_num(df[m["po_qty"]])
    base["입고수량"]    = _to_num(df[m["rcv_qty"]])
    base["미입고수량"]  = (base["발주수량"] - base["입고수량"]).clip(lower=0)
    base["입고구분"]    = (df[m["status"]].astype(str).str.strip() if m["status"] else pd.NA)

    stcol = base["입고구분"].fillna("").astype(str).str.strip()
    base["상태_표준"] = np.where(stcol.isin(COMPLETE), "입고완료",
                         np.where(stcol.isin(PARTIAL), "부분입고",
                         np.where(stcol.isin(OPEN), "미입고",
                                  np.where(stcol=="", "미표시", "기타"))))
    return base

def complete_now_series(base: pd.DataFrame) -> pd.Series:
    if base["입고구분"].notna().any():
        st_series = base["입고구분"].fillna("").astype(str).str.strip()
        return st_series.isin(COMPLETE)
    return (base["입고수량"] >= base["발주수량"])

def complete_by_cutoff(base: pd.DataFrame, cutoff) -> pd.Series:
    cutoff = pd.to_datetime(cutoff)
    return (base["입고수량"] >= base["발주수량"]) & base["입고일자"].notna() & (base["입고일자"] <= cutoff)

def backlog_by_cutoff(base: pd.DataFrame, cutoff) -> pd.Series:
    cutoff = pd.to_datetime(cutoff)
    due_ok = base["발주납기일자"].notna() & (base["발주납기일자"] <= cutoff)
    if base["입고구분"].notna().any():
        st_series = base["입고구분"].fillna("").astype(str).str.strip()
        return due_ok & st_series.isin(OPEN_OR_PARTIAL)
    else:
        return due_ok & (~complete_by_cutoff(base, cutoff))

def kpi_summary(base: pd.DataFrame, prev_eom: date, curr_eom: date) -> pd.DataFrame:
    total_lines   = base.groupby("제품군").size()
    received_curr = complete_now_series(base).groupby(base["제품군"]).sum()
    prev_overdue  = backlog_by_cutoff(base, prev_eom)
    curr_overdue  = backlog_by_cutoff(base, curr_eom)

    A_cnt = curr_overdue.groupby(base["제품군"]).sum()
    B_cnt = ((~prev_overdue) & curr_overdue).groupby(base["제품군"]).sum()
    prev_cnt = prev_overdue.groupby(base["제품군"]).sum()
    AB_cnt   = A_cnt.add(B_cnt, fill_value=0)

    idx = sorted(total_lines.index.unique().tolist())
    out = pd.DataFrame(index=idx)
    out["계(전체)"]         = total_lines
    out["입고건수(당월말)"]  = received_curr.reindex(idx, fill_value=0)
    out["전월(미입고)"]      = prev_cnt.reindex(idx, fill_value=0)
    out["당월(A)"]          = A_cnt.reindex(idx, fill_value=0)
    out["전월 比"]           = (out["당월(A)"] - out["전월(미입고)"]).astype(int)\
                               .map(lambda x: f"△{abs(x)}" if x<0 else (f"▲{x}" if x>0 else "0"))
    out["신규(B)"]          = B_cnt.reindex(idx, fill_value=0)
    out["계(A+B)"]          = AB_cnt.reindex(idx, fill_value=0)  # 필요 시 제거 가능
    out["적기입고율(당월말)"] = (out["입고건수(당월말)"] / out["계(전체)"]).fillna(0.0)

    total = pd.DataFrame(out.sum(numeric_only=True)).T
    total.index = ["합계"]
    total["전월 比"] = ""
    total["적기입고율(당월말)"] = out["입고건수(당월말)"].sum() / max(out["계(전체)"].sum(), 1)
    return pd.concat([out, total]).reset_index().rename(columns={"index":"제품군"})

def detail_at(base: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    cutoff = pd.to_datetime(cutoff)
    mask = backlog_by_cutoff(base, cutoff)
    D = base.loc[mask].copy()
    if D.empty: return D
    D["지연일수"] = (cutoff - pd.to_datetime(D["발주납기일자"])).dt.days
    cols = ["제품군","발주번호","거래처명","구매그룹","품목명","발주일자","발주납기일자",
            "입고일자","발주수량","입고수량","미입고수량","입고구분","지연일수"]
    return D[cols].sort_values(["지연일수","발주납기일자"], ascending=[False, True])

# ─────────────── 엑셀 보고서 생성 (원본 test2.py 형식 유지) ───────────────
def build_excel(summary, raw_df, prev_eom, curr_eom, det_prev, det_curr) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active; ws.title = "월말요약"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=12)
    ws.cell(row=1, column=1,
            value=f"<월말 미입고 요약>  (전월말: {pd.to_datetime(prev_eom):%Y-%m-%d}, 당월말: {pd.to_datetime(curr_eom):%Y-%m-%d})"
    ).font = Font(size=13, bold=True)

    r1, r2 = 3, 4
    header = PatternFill("solid", fgColor="F2F2F2")
    center = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="999999")

    ws.cell(row=r1, column=1, value="제품군")
    ws.cell(row=r1, column=2, value="계(전체)")
    ws.cell(row=r1, column=3, value="입고건수(당월말)")
    ws.merge_cells(start_row=r1, start_column=4, end_row=r1, end_column=6); ws.cell(row=r1, column=4, value="미입고")
    ws.merge_cells(start_row=r1, start_column=7, end_row=r1, end_column=8); ws.cell(row=r1, column=7, value="미입고")
    ws.cell(row=r1, column=9, value="적기입고율(당월말)")
    for c in [1,2,3,9]: ws.merge_cells(start_row=r1, start_column=c, end_row=r2, end_column=c)
    for i, lab in enumerate(["전월(미입고)","당월(A)","전월 比","신규(B)","계(A+B)"], start=4):
        ws.cell(row=r2, column=i, value=lab)

    for r in (r1, r2):
        for c in range(1, 10):
            cell = ws.cell(row=r, column=c)
            cell.font = Font(bold=True); cell.alignment = center
            cell.fill = header
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ordered = ["제품군","계(전체)","입고건수(당월말)","전월(미입고)","당월(A)","전월 比","신규(B)","계(A+B)","적기입고율(당월말)"]
    for rr in dataframe_to_rows(summary[ordered], index=False, header=False): ws.append(rr)

    max_row = ws.max_row
    for c in range(1,10):
        ws.column_dimensions[get_column_letter(c)].width = 14
        for r in range(r2+1, max_row+1):
            cell = ws.cell(row=r, column=c)
            cell.alignment = center
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for r in range(r2+1, max_row+1):
        for c in [2,3,4,5,7,8]: ws.cell(row=r, column=c).number_format = '#,##0'
        ws.cell(row=r, column=9).number_format = '0.0%'
    ws.freeze_panes = "A5"

    ws_raw = wb.create_sheet("현황(원본)")
    if raw_df is None or raw_df.empty:
        ws_raw.cell(row=1, column=1, value="원본 표가 비어있습니다.")
    else:
        ws_raw.append(list(raw_df.columns))
        for rr in dataframe_to_rows(raw_df, index=False, header=False): ws_raw.append(rr)
        for c in range(1, len(raw_df.columns)+1): ws_raw.cell(row=1, column=c).font = Font(bold=True)
        ws_raw.freeze_panes = "A2"

    ws_p = wb.create_sheet("미입고_상세(전월말)")
    if det_prev.empty:
        ws_p.cell(row=1, column=1, value="전월말 기준 미입고가 없습니다.")
    else:
        ws_p.append(list(det_prev.columns))
        for rr in dataframe_to_rows(det_prev, index=False, header=False): ws_p.append(rr)
        for c in range(1, len(det_prev.columns)+1): ws_p.cell(row=1, column=c).font = Font(bold=True)
        ws_p.freeze_panes = "A2"

    ws_c = wb.create_sheet("미입고_상세(당월말)")
    if det_curr.empty:
        ws_c.cell(row=1, column=1, value="당월말 기준 미입고가 없습니다.")
    else:
        ws_c.append(list(det_curr.columns))
        for rr in dataframe_to_rows(det_curr, index=False, header=False): ws_c.append(rr)
        for c in range(1, len(det_curr.columns)+1): ws_c.cell(row=1, column=c).font = Font(bold=True)
        ws_c.freeze_panes = "A2"

    bio = io.BytesIO(); wb.save(bio); bio.seek(0)
    return bio.read()

# ───────────────────────── 사이드바: 업로드 & 히스토리 ─────────────────────────
st.sidebar.header("📦 데이터 업로드 & 기준일")

# 1) 새 파일 업로드(다중) → ./uploads 저장 + manifest 갱신
upfiles = st.sidebar.file_uploader(
    "발주현황 파일 업로드 (.xlsx/.xls/.xlsb/.csv/HTML/XML)",
    type=["xlsx","xls","xlsm","xltx","xlsb","csv","htm","html","xml","mht","mhtml"],
    accept_multiple_files=True
)
if upfiles:
    for f in upfiles:
        rec = save_upload(f)
    st.sidebar.success("업로드/저장 완료! (좌측 '업로드 히스토리'에 반영됨)")

# 2) 업로드 히스토리: 선택해서 불러오기 & 삭제
st.sidebar.subheader("📂 업로드 히스토리")
hist = list_uploads()
if hist:
    labels = [f"{i+1}. {h['name']}  ·  {h['uploaded_at']}" for i, h in enumerate(hist)]
    idx = st.sidebar.selectbox("이전 업로드 불러오기", range(len(hist)), format_func=lambda i: labels[i])
    col_a, col_b = st.sidebar.columns([1,1])
    use_hist   = col_a.button("이 파일 불러오기")
    del_hist   = col_b.button("선택 파일 삭제")
    if del_hist:
        delete_upload(hist[idx]["id"])
        st.sidebar.warning("선택 파일을 삭제했습니다. (새로고침 시 목록 반영)")
else:
    st.sidebar.info("저장된 업로드가 없습니다.")

# 3) 기준일
query_date = st.sidebar.date_input("조회 기준일", date.today())
curr_eom = month_end(add_months(query_date, -1))
prev_eom = month_end(add_months(query_date, -2))
st.sidebar.info(f"전월말: **{prev_eom}**, 당월말: **{curr_eom}**")

# 4) 상태 라벨 커스터마이즈
with st.sidebar.expander("상태 라벨 커스터마이즈", expanded=False):
    comp_str = st.text_input("입고완료 라벨(쉼표)", "입고완료,완료")
    part_str = st.text_input("부분입고 라벨(쉼표)", "부분입고,부분,부분완료")
    open_str = st.text_input("미입고 라벨(쉼표)", "미입고,대기,미완료")
COMPLETE = set(s.strip() for s in comp_str.split(",") if s.strip())
PARTIAL  = set(s.strip() for s in part_str.split(",") if s.strip())
OPEN     = set(s.strip() for s in open_str.split(",") if s.strip())
OPEN_OR_PARTIAL = OPEN | PARTIAL

st.title("📊 미입고 KPI 대시보드")

# ───────────────────────── 데이터 로딩 ─────────────────────────
def read_path(path: Path) -> pd.DataFrame:
    class _MemUpload:
        def __init__(self, name, raw): self.name, self._raw = name, raw
        def read(self): return self._raw
    raw = path.read_bytes()
    return read_any(_MemUpload(path.name, raw))

raw_df = None
if hist:
    chosen = hist[idx]
    if use_hist or True:  # 기본으로 선택된 항목 사용
        try:
            raw_df = read_path(UPLOAD_DIR / chosen["path"])
        except Exception as e:
            st.error(f"히스토리 파일 읽기 오류: {e}")
            halt_app()
else:
    st.info("좌측에서 파일을 업로드하거나 히스토리에서 선택해 주세요.")
    halt_app()

# ───────────────────────── 표준화·필터·지표 ─────────────────────────
if raw_df is None:
    st.error("데이터를 불러오지 못했습니다. 파일을 업로드 후 다시 시도해 주세요.")
    halt_app()

base = build_base(raw_df.copy())

st.subheader("🔎 필터")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    prods = st.multiselect("제품군", sorted(base["제품군"].dropna().unique().tolist()))
with c2:
    vendors = st.multiselect("거래처명", sorted(base["거래처명"].astype(str).dropna().unique().tolist())[:5000])
with c3:
    pg_opts = sorted(
        x for x in base["구매그룹"].fillna("").astype(str).str.strip().unique().tolist()
        if x and x.lower() != "nan"
    )
    purchase_groups = st.multiselect("구매그룹", pg_opts)
with c4:
    statuses = st.multiselect("입고구분", sorted(base["입고구분"].astype(str).dropna().unique().tolist()))
with c5:
    state_std = st.multiselect("상태(표준)", ["입고완료","부분입고","미입고","미표시","기타"])

due_range = None
order_range = None
due_range_default = None
order_range_default = None
date_cols = st.columns(2)
with date_cols[0]:
    due_series = base["발주납기일자"].dropna()
    if not due_series.empty:
        due_range_default = (due_series.min().date(), due_series.max().date())
        due_range = st.date_input(
            "발주납기일자 범위",
            value=due_range_default,
            min_value=due_range_default[0],
            max_value=due_range_default[1],
            format="YYYY-MM-DD",
        )
    else:
        st.date_input(
            "발주납기일자 범위",
            value=(date.today(), date.today()),
            format="YYYY-MM-DD",
            disabled=True,
            help="발주납기일자가 없어 필터를 사용할 수 없습니다.",
        )
with date_cols[1]:
    order_series = base["발주일자"].dropna()
    if not order_series.empty:
        order_range_default = (order_series.min().date(), order_series.max().date())
        order_range = st.date_input(
            "발주일자 범위",
            value=order_range_default,
            min_value=order_range_default[0],
            max_value=order_range_default[1],
            format="YYYY-MM-DD",
        )
    else:
        st.date_input(
            "발주일자 범위",
            value=(date.today(), date.today()),
            format="YYYY-MM-DD",
            disabled=True,
            help="발주일자가 없어 필터를 사용할 수 없습니다.",
        )

flt = base.copy()
if prods:    flt = flt[flt["제품군"].isin(prods)]
if vendors:  flt = flt[flt["거래처명"].astype(str).isin(vendors)]
if purchase_groups:
    flt = flt[flt["구매그룹"].fillna("").astype(str).str.strip().isin(purchase_groups)]
if statuses: flt = flt[flt["입고구분"].astype(str).isin(statuses)]
if state_std:flt = flt[flt["상태_표준"].isin(state_std)]
if due_range_default and isinstance(due_range, (list, tuple)) and len(due_range) == 2:
    due_start, due_end = due_range
    if due_start and due_end:
        due_start_ts = pd.to_datetime(due_start)
        due_end_ts = pd.to_datetime(due_end)
        due_mask = flt["발주납기일자"].notna() & flt["발주납기일자"].between(due_start_ts, due_end_ts)
        if tuple(due_range) == due_range_default:
            flt = flt[due_mask | flt["발주납기일자"].isna()]
        else:
            flt = flt[due_mask]
if order_range_default and isinstance(order_range, (list, tuple)) and len(order_range) == 2:
    order_start, order_end = order_range
    if order_start and order_end:
        order_start_ts = pd.to_datetime(order_start)
        order_end_ts = pd.to_datetime(order_end)
        order_mask = flt["발주일자"].notna() & flt["발주일자"].between(order_start_ts, order_end_ts)
        if tuple(order_range) == order_range_default:
            flt = flt[order_mask | flt["발주일자"].isna()]
        else:
            flt = flt[order_mask]

st.caption(f"필터 적용 결과: {len(flt):,} 행")

summary = kpi_summary(flt, prev_eom, curr_eom)
total_row = summary[summary["제품군"]=="합계"].iloc[0]

m = st.columns(6)
m[0].metric("계(전체)", f"{int(total_row['계(전체)']):,}")
m[1].metric("입고건수(당월말)", f"{int(total_row['입고건수(당월말)']):,}")
m[2].metric("전월(미입고)", f"{int(total_row['전월(미입고)']):,}")
m[3].metric("당월(A)", f"{int(total_row['당월(A)']):,}")
m[4].metric("신규(B)", f"{int(total_row['신규(B)']):,}")
m[5].metric("적기입고율", f"{float(total_row['적기입고율(당월말)']):.1%}")

st.subheader("📈 시각화")
curr_over_mask = backlog_by_cutoff(flt, curr_eom)
bar_df = flt.loc[curr_over_mask].groupby("제품군").size().reset_index(name="미입고(당월말)")
st.plotly_chart(px.bar(bar_df.sort_values("미입고(당월말)", ascending=False),
                       x="제품군", y="미입고(당월말)", text_auto=True,
                       title=f"제품군별 미입고(당월말: {curr_eom})"),
                use_container_width=True)

topN = st.slider("거래처 Top N (당월말 미입고)", 5, 30, 10)
vendor_df = flt.loc[curr_over_mask].groupby("거래처명").size().reset_index(name="미입고(당월말)")
vendor_df = vendor_df.sort_values("미입고(당월말)", ascending=False).head(topN)
st.plotly_chart(px.bar(vendor_df, x="거래처명", y="미입고(당월말)", text_auto=True, title="거래처 Top 미입고"),
                use_container_width=True)

status_df = flt["상태_표준"].value_counts().reset_index()
status_df.columns = ["상태","건수"]
st.plotly_chart(px.pie(status_df, names="상태", values="건수", hole=0.5, title="상태 분포"),
                use_container_width=True)

st.subheader("📋 당월말 미입고 상세")
detail_prev = detail_at(flt, prev_eom)
detail_curr = detail_at(flt, curr_eom)
st.dataframe(detail_curr.head(200), use_container_width=True)

st.subheader("📑 월말 요약표")
disp = summary.copy()
disp["적기입고율(당월말)"] = (disp["적기입고율(당월말)"]*100).round(1).astype(str) + "%"
st.dataframe(disp, use_container_width=True)

# ───────────────────────── 다운로드 ─────────────────────────
st.subheader("⬇️ 다운로드")
xlsx_bytes = build_excel(summary, raw_df, prev_eom, curr_eom, detail_prev, detail_curr)
st.download_button(
    "엑셀 보고서 다운로드 (요약+상세+원본 .xlsx)",
    data=xlsx_bytes,
    file_name=f"미입고_월말리포트_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.download_button(
    "필터 결과 CSV 다운로드",
    data=flt.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"발주현황_필터결과_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)
st.caption("※ 업로드한 파일은 ./uploads 폴더에 저장됩니다. 히스토리에서 선택/삭제 가능합니다.")
