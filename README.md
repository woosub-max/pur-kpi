# 미입고 자재 메일 발송 Streamlit 앱

거래처별 미입고 내역을 필터링하여 HTML 이메일로 자동 발송하는 Streamlit 예제입니다. 샘플 DataFrame이 포함되어 있어 즉시 UI를 확인할 수 있으며, SMTP 또는 SendGrid 설정을 넣으면 실제 발송도 가능합니다.

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 환경 변수 / Secrets 설정
민감 정보는 코드에 하드코딩하지 않고 환경 변수 또는 Streamlit `secrets.toml`을 사용합니다. 다음 중 하나를 선택하세요.

### 옵션 A: SMTP
- 환경 변수
  ```bash
  export SMTP_HOST=smtp.gmail.com
  export SMTP_PORT=587
  export SMTP_USER=you@example.com
  export SMTP_PASS=app_password
  ```
- `.streamlit/secrets.toml`
  ```toml
  SMTP_HOST = "smtp.gmail.com"
  SMTP_PORT = 587
  SMTP_USER = "you@example.com"
  SMTP_PASS = "app_password"
  ```

### 옵션 B: SendGrid
- 환경 변수
  ```bash
  export SENDGRID_API_KEY=your_sendgrid_key
  ```
- `.streamlit/secrets.toml`
  ```toml
  SENDGRID_API_KEY = "your_sendgrid_key"
  ```

## 주요 기능
- 제품군/거래처명/구매그룹 멀티셀렉트 필터
- 거래처별 그룹 요약(품목 수, 총 미입고 수량)
- 업체 선택 후 제목/본문 템플릿에 `{{vendor_name}}`, `{{today}}`, `{{items_table}}` 치환
- HTML 테이블 포함 본문 미리보기
- 선택/전체 업체 발송, 드라이런 모드 및 발송 지연 설정
- 수신자 이메일 누락·형식 오류·발송 오류에 대한 결과 테이블 표시

## 테스트 팁
- 기본 드라이런 모드에서 “메일 보내기”를 누르면 실제 발송 없이 결과 로그만 확인합니다.
- SMTP/SendGrid 설정 후 드라이런을 끄고 테스트 계정으로 1건 이상 발송되는지 확인하세요.
