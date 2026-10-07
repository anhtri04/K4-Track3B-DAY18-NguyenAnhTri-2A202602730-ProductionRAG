# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Anh Trí
**MSSV:** 2A202602730
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 2026-10-05
**LLM:** DeepSeek `deepseek-chat` qua `OPENAI_BASE_URL=https://api.deepseek.com/v1` (trống base_url → fallback OpenAI)

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` (`all-MiniLM-L6-v2`, threshold 0.85) | Threshold mặc định trên corpus thật cho 208 chunks avg 99 chars vs basic 51 chunks avg 410 — semantic cắt mịn theo câu, giữ ý nhưng sinh quá nhiều mảnh nhỏ (min 6 chars); phù hợp đoạn ngắn, không phải default production. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` (parent 2048 + child 256) | Children (~87–100 chunks, avg 240) nhỏ hơn parents rõ rệt; `parent_id` link hợp lệ 100%. Đây là strategy pipeline dùng để index — retrieve child (precision) rồi có thể trả parent (context). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` (parse `#{1,3}`) | 106 chunks avg 196, giữ nguyên header trong text + `section` metadata; không cắt giữa table/list. Tốt nhất cho corpus markdown chính sách (mỗi file 1 chính sách). |
| Vietnamese segmentation | M2 | `segment_vietnamese()` (underthesea + `replace("_"," ")`) | `nghỉ_phép` → `nghỉ phép`; nếu quên replace, BM25 `split(" ")` coi `nghỉ_phép` là 1 token trong khi query là 2 token → recall lexical sụp. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` (`Σ 1/(k+rank+1)`, k=60) + `HybridSearch` (BM25 top-20 + bge-m3/Qdrant top-20) | BM25 bắt số/ngày chính xác ("120 ngày", "12 ký tự"), dense bắt paraphrase; RRF hợp nhất không cần chuẩn hóa score. Query "nghỉ phép" trả đúng chunk phép năm đầu tiên ở cả 2 nhánh. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` (`bge-reranker-v2-m3` via `sentence_transformers.CrossEncoder`) | Top-20 → top-3; probe "nghỉ phép" cho scores 0.991 vs 0.001 vs 0.0 — phân biệt dứt khoát. Latency CPU ~172ms avg (166–181ms, n=3). Không dùng `FlagEmbedding.FlagReranker` vì crash với transformers>=5 (`XLMRobertaTokenizer`). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` (ragas 0.1.22 + `datasets.Dataset`) | Production: faithfulness 0.8333 (+0.1867 vs naive 0.6467), precision 0.725 (+0.025), recall 0.775 (−0.075). `answer_relevancy=NaN` cả 2 pipeline vì DeepSeek không hỗ trợ `n>1` (RAGAS judge gọi `n=3` → `400 Invalid n value`). Vẫn đạt 3/4 metrics ≥0.70. |
| Failure diagnosis | M4 | `failure_analysis()` (Diagnostic Tree bottom-N) | Map worst-metric → diagnosis/fix (faithfulness→hallucination/tighten prompt; recall→missing chunks/BM25; precision→irrelevant/rerank). Phát hiện bug: avg dính NaN làm mọi score NaN — cần bỏ qua NaN khi sort. |
| Contextual enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` (1 call/chunk: summary+questions+context+metadata) | Combined mode 1 API call thay vì 4 — tiết kiệm cost/toàn bộ `enrich_chunks()` vẫn chạy qua DeepSeek base_url custom. Fallback extractive khi mất key (giữ `SAMPLE in result`, metadata default `policy/vi`). Enrichment giúp chunk mang tên version ("Phiên bản 2.0 hiện hành") → giảm lỗi version-conflict (#1, #2). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `Exception raised in Job[...]: BadRequestError(Error code: 400 - {'error': {'message': 'Invalid n value (currently only n = 1 is supported)'}})` — lặp lại hàng chục lần trong phase `Evaluating: .../80` của RAGAS khi dùng DeepSeek.
  - `BCTC.pdf` / `Nghi_dinh_so_13...pdf` bị skip: `PDF scan ảnh, không có text layer (cần OCR)`.
  - `answer_relevancy: NaN` ở cả naive và production report; `failures[].score: NaN` toàn bộ.
- **Nguyên nhân gốc rễ & Cách debug:**
  - RAGAS `answer_relevancy` sinh nhiều completions với `n=3`; DeepSeek API chỉ cho `n=1` → request 400 cho riêng metric này, các metric khác vẫn chạy. Xác định bằng cách đọc log Job song song với `Evaluating` bar + đối chiếu `reports/*.json` (chỉ metric này NaN). Kết luận: giới hạn của judge endpoint, không phải bug retrieval — ghi nhận trong báo cáo thay vì cố "fix số".
  - PDF scan: `pypdf.PdfReader.extract_text()` trả `""` → `load_documents()` bỏ qua có cảnh báo (đúng thiết kế text-based RAG). Ghi nhận cần OCR ngoài phạm vi lab.
  - NaN lan truyền: `failure_analysis()` tính mean gồm NaN → sort vô nghĩa. Debug bằng `json.load` report + in scores. Fix đề xuất: lọc NaN trước khi avg/sort (ghi trong failure_analysis "Nếu có thêm 1 giờ").
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Thiếu: RAGAS `evaluate(..., llm=...)` + `LangchainLLMWrapper(ChatOpenAI(...))` cho custom base_url; khác biệt `qdrant_client.query_points()` vs `search()`; `CrossEncoder` vs `FlagReranker`.
  - Cách bổ sung: đọc signature `inspect.signature(evaluate)`, `dir(ragas.llms)`; test cô lập từng module bằng `pytest tests/test_mX.py -v` trước khi chạy pipeline 9 phút; probe 5 câu hỏi thật qua `HybridSearch+rernk+LLM` để có Got/Context thật cho báo cáo thay vì đoán.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: [Điền tên project RAG của bạn]

#### 1. Hiện trạng
- **Pipeline hiện tại:** Paragraph chunking + dense-only + top-3 trả lời trực tiếp (tương đương `naive_baseline.py`).
- **Vấn đề / Bottlenecks đang gặp:** (a) lẫn version cũ/mới khi corpus có nhiều bản; (b) câu multi-hop thiếu 1 vế (recall); (c) không có eval định lượng nên mọi cải tiến đều "cảm tính"; (d) latency rerank chưa đo.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical (parent 2048/child 256) làm default; tài liệu markdown nhiều header thì bật structure-aware để giữ section + bảng. Không dùng semantic đại trà (quá vụn: 208 chunks).
2. **Search retrieval:** Hybrid BM25 (có `segment_vietnamese` + replace `_`) + dense bge-m3, fuse bằng RRF k=60. BM25 xử số/từ khóa chính xác, dense xử paraphrase tiếng Việt.
3. **Reranking:** Có — `bge-reranker-v2-m3` top-20→top-3 cho QA chính xác; đo latency (~172ms CPU) và cân nhắc tắt với query đơn giản hoặc chuyển lightweight ranker nếu p95 vượt SLA.
4. **Evaluation:** RAGAS 4 metrics trên bộ 20 câu phân loại sẵn (lookup/version/negation/multi-hop/numeric/ambiguous) + `failure_analysis` bottom-N theo Diagnostic Tree. Lưu ý chọn judge endpoint hỗ trợ `n>1` nếu cần `answer_relevancy`, nếu không thì document NaN như lab này.
5. **Enrichment:** Combined single-call (summary + 3 HyQA + 1 context + metadata JSON) cho production; giá trị lớn nhất là prepend version/tên văn bản để chống version-conflict. Giữ fallback extractive khi mất API key.

#### 3. Timeline triển khai
- **Tuần 1:** Thêm version metadata (`version/is_current/effective_date`) + filter lúc retrieve; backfill contextual prepend cho corpus hiện có; dựng bộ eval 20–30 câu theo 6 loại của lab.
- **Tuần 2:** Triển khai Hybrid + RRF + rerank sau đo latency; thêm query decomposition cho câu có "và/cả hai" (multi-hop); chạy A/B naive vs production, mục tiêu ≥3 metrics ≥0.70 như rubric, faithfulness ≥0.85 lấy bonus.
