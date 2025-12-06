import os
import re
import time
from datetime import datetime, date
from email.message import EmailMessage
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except Exception:  # optional dependency
    SendGridAPIClient = None
    Mail = None


st.set_page_config(page_title="미입고 메일 발송", page_icon="📧", layout="wide")


def load_sample_data() -> pd.DataFrame:
    today = date.today()
    sample_rows = []
    vendors = ["에이테크", "비비상사", "씨엔디"]
    product_groups = ["전자부품", "포장재", "금속"]
    purch_groups = ["PG01", "PG02"]
    for idx in range(1, 25):
        vendor = np.random.choice(vendors)
        pg = np.random.choice(product_groups)
        purch = np.random.choice(purch_groups)
        order_date = pd.Timestamp(today.replace(day=1)) - pd.Timedelta(days=np.random.randint(0, 60))
        due_date = order_date + pd.Timedelta(days=np.random.randint(7, 35))
        order_qty = int(np.random.randint(10, 200))
        open_qty = int(order_qty * np.random.uniform(0.2, 1.0))
        sample_rows.append(
            {
                "product_group": pg,
                "vendor_name": vendor,
                "purch_group": purch,
                "material_code": f"MAT{idx:04d}",
                "material_name": f"샘플 품목 {idx}",
                "po_no": f"PO{20240000+idx}",
                "order_qty": order_qty,
                "open_qty": open_qty,
                "po_date": order_date.date(),
                "due_date": due_date.date(),
                "vendor_email": f"test+{idx}@example.com",
            }
        )
    return pd.DataFrame(sample_rows)


def is_valid_email(address: str) -> bool:
    if not address:
        return False
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", address) is not None


def build_items_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>미입고 내역이 없습니다.</p>"
    display_df = df[
        ["material_code", "material_name", "po_no", "order_qty", "open_qty", "po_date", "due_date"]
    ].copy()
    display_df.rename(
        columns={
            "material_code": "품목코드",
            "material_name": "품목명",
            "po_no": "발주번호",
            "order_qty": "발주수량",
            "open_qty": "미입고수량",
            "po_date": "발주일",
            "due_date": "납기일",
        },
        inplace=True,
    )
    return display_df.to_html(index=False, justify="center", border=1)


def render_template(template: str, placeholders: Dict[str, str]) -> str:
    html = template
    for key, val in placeholders.items():
        html = html.replace(f"{{{{{key}}}}}", val)
    return html


def send_via_smtp(msg: EmailMessage, dry_run: bool) -> None:
    if dry_run:
        return
    host = os.environ.get("SMTP_HOST") or st.secrets.get("SMTP_HOST", None)
    port = int(os.environ.get("SMTP_PORT") or st.secrets.get("SMTP_PORT", 587))
    user = os.environ.get("SMTP_USER") or st.secrets.get("SMTP_USER", None)
    password = os.environ.get("SMTP_PASS") or st.secrets.get("SMTP_PASS", None)
    if not (host and user and password):
        raise RuntimeError("SMTP 설정을 확인해주세요.")
    import smtplib

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def send_via_sendgrid(to_addr: str, subject: str, html_body: str, dry_run: bool) -> None:
    if dry_run:
        return
    api_key = os.environ.get("SENDGRID_API_KEY") or st.secrets.get("SENDGRID_API_KEY", None)
    if not api_key:
        raise RuntimeError("SendGrid API Key가 설정되지 않았습니다.")
    if SendGridAPIClient is None or Mail is None:
        raise RuntimeError("sendgrid 패키지가 설치되지 않았습니다.")
    message = Mail(from_email="no-reply@example.com", to_emails=to_addr, subject=subject, html_content=html_body)
    sg = SendGridAPIClient(api_key)
    sg.send(message)


def send_email(to_addr: str, subject: str, html_body: str, dry_run: bool = True) -> None:
    if not is_valid_email(to_addr):
        raise ValueError("수신자 이메일이 유효하지 않습니다.")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_USER") or st.secrets.get("SMTP_USER", "no-reply@example.com")
    msg["To"] = to_addr
    msg.add_alternative(html_body, subtype="html")

    if os.environ.get("SENDGRID_API_KEY") or st.secrets.get("SENDGRID_API_KEY", None):
        send_via_sendgrid(to_addr, subject, html_body, dry_run)
    else:
        send_via_smtp(msg, dry_run)


def build_vendor_email(vendor: str, vendor_df: pd.DataFrame, subject_tpl: str, body_tpl: str) -> Dict[str, str]:
    items_table = build_items_table(vendor_df)
    placeholders = {
        "vendor_name": vendor,
        "today": datetime.now().strftime("%Y-%m-%d"),
        "items_table": items_table,
    }
    subject = render_template(subject_tpl, placeholders)
    body_html = render_template(body_tpl, placeholders)
    return {"subject": subject, "body_html": body_html}


def main():
    st.title("📧 미입고 업체별 자동 메일 발송")
    st.caption("필터 → 거래처별 그룹 → 개별 이메일 자동 발송")

    df = load_sample_data()

    with st.sidebar:
        st.header("필터")
        prod_selected = st.multiselect("제품군", sorted(df["product_group"].unique()))
        vendor_selected = st.multiselect("거래처명", sorted(df["vendor_name"].unique()))
        purch_selected = st.multiselect("구매그룹", sorted(df["purch_group"].unique()))
        dry_run = st.toggle("드라이런 모드 (보내지 않고 로그만)", value=True)
        delay = st.slider("건당 지연(초)", 0.0, 1.0, 0.5, 0.1, help="대량 발송 시 스팸 방지")

    flt = df.copy()
    if prod_selected:
        flt = flt[flt["product_group"].isin(prod_selected)]
    if vendor_selected:
        flt = flt[flt["vendor_name"].isin(vendor_selected)]
    if purch_selected:
        flt = flt[flt["purch_group"].isin(purch_selected)]

    st.subheader("필터 결과 미리보기")
    st.dataframe(flt.head(200), use_container_width=True)

    st.subheader("거래처별 요약")
    summary = (
        flt.groupby("vendor_name")
        .agg(items=("material_code", "nunique"), open_qty=("open_qty", "sum"), email=("vendor_email", "first"))
        .reset_index()
        .sort_values("open_qty", ascending=False)
    )
    st.dataframe(summary, use_container_width=True)

    vendors_available = summary["vendor_name"].tolist()
    selected_vendors = st.multiselect("메일을 보낼 업체 선택", vendors_available, default=vendors_available)

    st.subheader("이메일 템플릿")
    default_subject = "[미입고 안내] {{vendor_name}} - {{today}} 기준"
    default_body = (
        "<p>안녕하세요, {{vendor_name}} 담당자님.</p>"
        "<p>{{today}} 기준 미입고 내역을 공유드립니다. 확인 부탁드립니다.</p>"
        "{{items_table}}"
        "<p>문의가 있으시면 회신 부탁드립니다.</p>"
    )
    subject_tpl = st.text_input("제목 템플릿", value=default_subject)
    body_tpl = st.text_area("본문 템플릿 (HTML 지원, {{items_table}} 자리표시자 포함)", value=default_body, height=220)

    preview_vendor = st.selectbox("미리보기용 업체", selected_vendors if selected_vendors else vendors_available)
    if st.button("미리보기", disabled=not preview_vendor):
        vendor_df = flt[flt["vendor_name"] == preview_vendor]
        email_preview = build_vendor_email(preview_vendor, vendor_df, subject_tpl, body_tpl)
        st.markdown(f"**제목:** {email_preview['subject']}")
        st.markdown(email_preview["body_html"], unsafe_allow_html=True)

    st.divider()
    st.subheader("메일 발송")
    send_mode = st.radio("발송 대상", ["선택 업체", "전체 업체"], horizontal=True)

    result_rows: List[Dict[str, str]] = []
    if st.button("메일 보내기", type="primary"):
        target_vendors = selected_vendors if send_mode == "선택 업체" else vendors_available
        if not target_vendors:
            st.warning("보낼 업체가 없습니다.")
        for vendor in target_vendors:
            vendor_df = flt[flt["vendor_name"] == vendor]
            email_addr = vendor_df["vendor_email"].iloc[0] if not vendor_df.empty else ""
            status = "성공"
            detail = ""
            if not is_valid_email(email_addr):
                status = "건너뜀"
                detail = "유효한 이메일 없음"
            else:
                email_content = build_vendor_email(vendor, vendor_df, subject_tpl, body_tpl)
                try:
                    send_email(email_addr, email_content["subject"], email_content["body_html"], dry_run=dry_run)
                    detail = "드라이런" if dry_run else "발송 완료"
                except Exception as e:  # pylint: disable=broad-except
                    status = "실패"
                    detail = str(e)
            result_rows.append({"거래처": vendor, "이메일": email_addr, "결과": status, "비고": detail})
            if delay:
                time.sleep(delay)
        if result_rows:
            st.success("처리가 완료되었습니다. 결과를 확인하세요.")
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

    st.caption("이 코드는 예시로, 실제 SMTP/SendGrid 정보는 환경 변수 또는 Streamlit secrets에 설정하세요.")


if __name__ == "__main__":
    main()
