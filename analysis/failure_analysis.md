# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Anh Trí
**MSSV:** 2A202602730
**Khóa:** K4 - Track 3B
**LLM judge/generator:** DeepSeek `deepseek-chat` via `OPENAI_BASE_URL=https://api.deepseek.com/v1` (fallback OpenAI khi để trống base_url)
**Ngày chạy:** pipeline `main.py` — total ~527s

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6467 | 0.8333 | +0.1867 |
| Answer Relevancy | NaN | NaN | NaN |
| Context Precision | 0.7000 | 0.7250 | +0.0250 |
| Context Recall | 0.8500 | 0.7750 | -0.0750 |

> 3/4 metrics đo được đạt ≥ 0.70 (faithfulness, precision, recall).
> `answer_relevancy = NaN` ở **cả 2 pipeline** — không phải do retrieval mà do DeepSeek trả
> `400 Invalid n value (currently only n = 1 is supported)` khi RAGAS gọi judge với `n=3`
> (xem log `Exception raised in Job[...]: BadRequestError ... n value`). Metric này bị loại khỏi so sánh.

Chi tiết chunking (26 docs; 2 PDF scan `BCTC.pdf`, `Nghi_dinh_so_13...pdf` bị bỏ qua vì không có text layer — cần OCR):

| Strategy | Chunks | Avg | Min | Max |
|----------|--------|-----|-----|-----|
| basic | 51 | 410 | 273 | 565 |
| semantic | 208 | 99 | 6 | 354 |
| hierarchical (children) | 100/87* | 240 | 23 | 255 |
| structure | 106 | 196 | 86 | 788 |

> \* 100 children khi chạy trên docs thật (pipeline probe), 87 khi chạy `compare_strategies` trên text nối — cùng bậc độ lớn.
> Rerank (`bge-reranker-v2-m3`): query "nghỉ phép" xếp đúng `12 ngày/năm (0.991)` > `mật khẩu (0.001)` > `VPN (0.0)`; latency CPU ~172ms avg (166 min / 181 max, n=3).

## Bottom-5 Failures

> Lưu ý phương pháp: `failures` trong `reports/ragas_report.json` đều có `score: NaN` vì
> `failure_analysis()` tính avg gồm cả `answer_relevancy=NaN` → NaN lan truyền, thứ tự bottom-10
> giữ nguyên thứ tự test_set. Do đó 5 case dưới được chọn theo **loại lỗi đại diện**
> (version-conflict, negation, multi-hop numeric, ambiguous) và đã verify bằng probe
> `HybridSearch + rerank + deepseek-chat` thực tế (xem Got/Context).

### #1 — Version conflict (v2023 vs v2024)
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày (v2024 hiện hành; v2023 12 ngày đã bị thay thế).
- **Got:** "Theo chính sách nghỉ phép năm 2024 ... **15 ngày phép năm** có lương." (đúng đáp án, nhưng mong manh)
- **Worst metric:** context_precision
- **Error Tree:** Output đúng → Context đúng một phần? → Cả 2 version đều lọt top-3 (v2023 + v2024) → Rerank xếp **v2023 lên rank 0**, v2024 rank 1 → LLM phải tự chọn version.
- **Root cause:** Retrieval không có version/metadata filter; corpus chứa 2 bản superseded (`nghi_phep_nam_v2023.md`, `mat_khau_v1.md`).
- **Suggested fix:** Add reranking or metadata filter — index `version/effective_date/is_current`, filter `is_current=true` hoặc prepend "Phiên bản 2.0 hiện hành" (M5 contextual) + recency boost.

### #2 — Version conflict (password policy)
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** 12 ký tự (v2.0); v1.0 8 ký tự đã bị thay thế.
- **Got:** "12 ký tự." (đúng nhưng context top-3 vẫn chứa cả v1.0 + v2.0)
- **Worst metric:** context_precision
- **Error Tree:** Output đúng → Context thừa (cả old + new) → Query OK → thiếu version-aware retrieval.
- **Root cause:** Giống #1 — BM25/dense chấm điểm lexical cao cho cả 2 bản vì cùng từ khóa "mật khẩu/ký tự".
- **Suggested fix:** Giống #1 + HyQA questions chứa version ("mật khẩu v2.0 bao nhiêu ký tự?").

### #3 — Negation (thử việc)
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG — phải xin nghỉ không lương, trưởng phòng duyệt.
- **Got:** "nhân viên thử việc **không được nghỉ phép năm** ... phải xin nghỉ không lương..." (đúng)
- **Worst metric:** faithfulness (nhạy với phủ định — chỉ cần thiếu chữ "không" là sai hoàn toàn)
- **Error Tree:** Output đúng → Context đúng (chunk phủ định rank 0 sau rerank) → Query OK → rủi ro còn lại nằm ở generator.
- **Root cause:** Câu phủ định dễ bị LLM bỏ sót "KHÔNG" khi tóm tắt; faithfulness metric trừng phạt nặng.
- **Suggested fix:** Tighten prompt, lower temperature — giữ nguyên system prompt "Trả lời CHỈ dựa trên context", thêm "giữ nguyên từ phủ định KHÔNG/không được".

### #4 — Multi-hop numeric (Senior 9 năm + lương)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày; lương Senior (P3-P4) 20–35 triệu.
- **Got:** "**không có thông tin về mức lương** ... 9 năm thâm niên được hưởng **18 ngày phép** (15 + 1 ngày/3 năm)" — đúng nửa phép, mất nửa lương.
- **Worst metric:** context_recall
- **Error Tree:** Output sai một nửa → Context thiếu (top-3 toàn chunk phép năm, không có chunk `bang_luong_2024.md`) → Query multi-hop nhưng retrieve single-hop top-3.
- **Root cause:** 1 query cần 2 facts ở 2 files khác nhau; hybrid top-20 → rerank top-3 vẫn thiên về 1 chủ đề dominant.
- **Suggested fix:** Improve chunking or add BM25 — query decomposition (tách thành 2 sub-queries: phép + lương) hoặc tăng top_k / diverse retrieval (maximal marginal relevance).

### #5 — Ambiguous / cross-file synthesis (phân loại lương)
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Dữ liệu Bí mật (cấp 3), mã hóa khi truyền, need-to-know (tổng hợp từ quy chế lương + chính sách phân loại dữ liệu).
- **Got:** "Thông tin lương là dữ liệu **Bí mật**." (đúng nhãn, thiếu cấp 3 + nghĩa vụ mã hóa)
- **Worst metric:** context_recall
- **Error Tree:** Output đúng một phần → Context thiếu chunk thứ 2 (policy phân loại chi tiết) → Query mơ hồ ("cấp độ" không nêu tên văn bản).
- **Root cause:** Đáp án cần synthesis 2 nguồn; enrichment hiện tại (combined 1-call) chưa nối cross-document.
- **Suggested fix:** Improve chunking or add BM25 — contextual prepend tên văn bản nguồn + HyQA cầu nối ("lương thuộc cấp mấy theo phân loại dữ liệu?").

## Case Study (cho presentation)

**Question chọn phân tích:** #4 — Senior 9 năm thâm niên (multi-hop numeric, lỗi rõ nhất: đúng 50%).

**Error Tree walkthrough:**
1. Output đúng? → **Không hoàn toàn** (18 ngày đúng, lương sai/missing).
2. Context đúng? → **Không đủ** — cả 3 contexts đều về "thâm niên/phép năm", zero chunk về "bảng lương Senior 20–35 triệu".
3. Query rewrite OK? → Query gốc gộp 2 ý ("bao nhiêu ngày phép VÀ lương khoảng nào") — retriever hiểu thành 1 vector, chủ đề "phép" áp đảo "lương".
4. Fix ở bước: **retrieval (recall)** — decompose query hoặc retrieve-then-merge per sub-question; dự phòng: tăng `HYBRID_TOP_K`/`RERANK_TOP_K` hoặc dùng MMR để ép diversity.

**Nếu có thêm 1 giờ, sẽ optimize:**
- (30') Version filter: thêm `is_current` metadata khi index, loại `*_v2023`, `*_v1` khỏi top-k mặc định (giải quyết #1, #2 — nhóm lỗi chiếm đa số test_set version-type).
- (20') Query decomposition cho câu multi-hop (`Và` → 2 sub-queries, RRF-merge) (giải quyết #4, #5).
- (10') Sửa `failure_analysis()` bỏ qua NaN khi tính avg để bottom-N có thứ tự thật thay vì giữ nguyên thứ tự test_set.
