# ThreatMamba — Bản tái hiện mã nguồn mở của ThreatMAMBA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c.svg)](https://pytorch.org/)

Bản tái hiện từ đầu của **ThreatMAMBA** (Ge và cộng sự, *IEEE TIFS*, 2026 —
[10.1109/TIFS.2026.3685967](https://doi.org/10.1109/TIFS.2026.3685967)), một pipeline
đầu-cuối cho bài toán **quy kết tấn công mạng**: đọc một báo cáo Cyber Threat Intelligence,
trích xuất IOC và TTP, dựng đồ thị tri thức an ninh mạng theo trục thời gian, rồi quy kết
báo cáo về một nhóm tấn công — kèm giải thích theo từng node dựa trên MITRE ATT&CK.

Toàn bộ các giai đoạn đều đã được cài đặt và chạy được. Repo này cũng ghi lại ngay trong mã
nguồn những lỗi cấu hình đã làm hỏng lần chạy đầu tiên và cách chẩn đoán chúng — đó chính là
phần thực tế nhất của việc tái hiện bài báo này.

> 🇬🇧 English version: [README.md](README.md)

---

## Mục lục

- [Hệ thống làm gì](#hệ-thống-làm-gì)
- [Pipeline](#pipeline)
- [Kết quả](#kết-quả)
- [Cài đặt](#cài-đặt)
- [Chạy pipeline](#chạy-pipeline)
- [Giao diện demo](#giao-diện-demo)
- [Cách đọc kết quả](#cách-đọc-kết-quả)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Kiểm thử](#kiểm-thử)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Phạm vi và điều chỉnh so với bài báo](#phạm-vi-và-điều-chỉnh-so-với-bài-báo)
- [Trích dẫn](#trích-dẫn)
- [Ghi nhận và giấy phép](#ghi-nhận-và-giấy-phép)

---

## Hệ thống làm gì

Cho một báo cáo CTI tiếng Anh ở dạng văn bản thuần, hệ thống trả về:

- danh sách các **nhóm tấn công** khả nghi được xếp hạng kèm xác suất từng lớp;
- các **IOC** tìm được trong văn bản (13 loại), cộng thêm những gì tra cứu trực tiếp từ các
  nguồn threat intel (Time, Geo-location, CMD, API);
- các **TTP** (tactic và technique theo MITRE ATT&CK) dự đoán ở mức câu;
- một **đồ thị tri thức an ninh mạng (CSKG)** gồm 20 loại node và 11 loại quan hệ, sắp theo
  trục thời gian của báo cáo;
- **điểm đóng góp của từng node**, để người phân tích thấy được bằng chứng nào dẫn tới kết
  luận quy kết;
- **đường cong độ bền (robustness)** cho biết dự đoán giữ vững đến đâu khi chỉ có 20–100%
  đầu của timeline — tức là ở giai đoạn sớm của một cuộc điều tra.

---

## Pipeline

```
Báo cáo CTI (văn bản)
   │
   ├─ Giai đoạn 1  chuẩn bị dữ liệu ..... làm sạch, lọc nhóm (>=30 tài liệu), chia 8:2
   │
   ├─ Giai đoạn 2  trích IOC ............ ioc-finder + regex -> 13 loại IOC
   │               daemon IOCHunter ..... chọn method bằng UCT (Eq. 1) -> VirusTotal /
   │                                      AlienVault OTX / RapidDNS -> LLM sàng lọc
   │                                      -> node Time, Geo-location, CMD, API
   │
   ├─ Giai đoạn 3  trích TTP ............ SecureBERT (đóng băng) -> chiếu -> SSM
   │                                      -> Gaussian attention theo nhãn (Eq. 2-5)
   │
   ├─ Giai đoạn 4  dựng CSKG ............ 20 loại node, 11 quan hệ, xương sống thời gian (Eq. 6)
   │
   ├─ Giai đoạn 5  classifier ........... GAT quan hệ (Eq. 9-11) ‖ MAMBA trên chuỗi state
   │                                      (Eq. 12-14) -> AvgPool‖MaxPool (Eq. 16)
   │                                      -> MLP sigmoid (Eq. 17)
   │                                      loss = BCE + λ·InfoNCE (Eq. 18-20)
   │
   ├─ Giai đoạn 6  đánh giá & XAI ....... F1, Top-k, robustness (Eq. 27), chất lượng biểu
   │                                      diễn (Eq. 24-26), đối chiếu ATT&CK (Eq. 28-29)
   │
   └─ Giai đoạn 7  giao diện demo ....... ứng dụng Streamlit 4 trang
```

### Ghi chú kiến trúc

**GAT quan hệ.** Mỗi trong 11 loại quan hệ có vector attention riêng (Eq. 10); 3 tầng,
4 head, số chiều 48, có residual và jump link. Cài đặt bằng các phép scatter thuần PyTorch
nên không cần `torch_geometric`.

**Nhánh MAMBA.** Một state space model chọn lọc 3 tầng chạy trên chuỗi node State — đây
chính là phần mã hóa thứ tự thời gian của cuộc tấn công. Kernel CUDA chính chủ `mamba-ssm`
được dùng khi có sẵn; ngoài ra có sẵn `SimpleSSM` thuần PyTorch (conv nhân quả + gating chọn
lọc + quét tuyến tính) cho máy không có GPU phù hợp.

**Học đối chiếu (contrastive).** Mẫu dương thêm nhiễu Gaussian lên feature và bỏ ngẫu nhiên
cạnh theo Bernoulli, nhưng giữ nguyên xương sống thời gian (Eq. 7). Mẫu âm ghép timeline của
hai báo cáo khác nhãn với độ lệch φ ∈ (−100, 100)\\{0} rồi nối lại các quan hệ kill-chain của
ATT&CK trên đồ thị đã hợp nhất (Eq. 8).

---

## Kết quả

Số liệu đo trên chính bản tái hiện này. Dữ liệu là tập CTI công bố kèm bài báo: 10.510 dòng →
3.589 tài liệu đủ dài → **20 nhóm / 3.144 tài liệu** sau khi giữ các nhóm có ít nhất 30 tài
liệu (2.515 train / 629 test).

| Thành phần | Chỉ số | Giá trị |
|---|---|---|
| GĐ 1 — parse ATT&CK v14 | techniques / tactics / groups | 625 / 14 / 142 |
| GĐ 3 — module TTP (D1) | 12.330 mẫu, 14 tactics, 540 cột technique (333 cột có ≥5 mẫu dương) | — |
| GĐ 4 — CSKG | trung bình node / cạnh mỗi đồ thị | ~37 / ~96 (bài báo ~53 / ~147) |
| GĐ 4 — CSKG không có node TTP | trung bình node / cạnh | ~13 / ~18 |
| GĐ 6 — ánh xạ nhóm sang ATT&CK | số nhóm khớp được intrusion-set | 15 / 20 |

Tiêu chí nghiệm thu Giai đoạn 5 dùng ở đây là **macro-F1 > 0,35** và **Top-3 > 0,60**;
lệnh `python -m src.evaluate all` in ra PASS/FAIL theo hai ngưỡng này và ghi
`outputs/metrics_all.csv`.

Bốn trong hai mươi nhóm của tập dữ liệu (Nitro, SEA, Scarab, TwoForOne) không có
intrusion-set tương ứng trong ATT&CK v14 nên được báo là không khớp, thay vì bị chấm điểm âm
thầm. Các alias đã kiểm chứng: Energetic Bear → Dragonfly, Hidden Cobra → Lazarus Group,
Quedagh → Sandworm Team, Tick → BRONZE BUTLER, Waterbug → Turla, MageCart → FIN6.

> **Để tái hiện đúng số liệu.** Giai đoạn 4 **bắt buộc** chạy sau Giai đoạn 3. Dựng CSKG khi
> chưa train module TTP chỉ cho ~13 node mỗi đồ thị thay vì ~37, và mọi con số phía sau đều
> thay đổi theo.

---

## Cài đặt

**Môi trường tham chiếu:** Windows 11 + WSL2 (Ubuntu 22.04/24.04) với GPU NVIDIA (phát triển
trên RTX 5060 Ti 16 GB, Blackwell `sm_120`), khoảng 25 GB trống. Linux thuần cũng chạy được.
Cần tài khoản miễn phí của VirusTotal và AlienVault OTX; RapidDNS không cần key.

```bash
git clone https://github.com/<tên-tài-khoản>/threatmamba-repro
cd threatmamba-repro
bash scripts/setup_env.sh
```

`setup_env.sh` tạo `.venv` với **Python 3.11**, cài **PyTorch 2.8.0 + cu128** (hỗ trợ
Blackwell `sm_120`) và toàn bộ thư viện cho Giai đoạn 1–7.

Ba điều quan trọng ở bước này:

**Python phải là 3.10–3.13, tốt nhất 3.11.** Python 3.14 chưa có wheel cho `mamba-ssm`,
`causal-conv1d` và nhiều thư viện ML — đây là nguyên nhân số một gây lỗi cài đặt.

**Script cố ý không cài `mamba-ssm`.** Wheel dựng sẵn chỉ compile tới `sm_100`, nên trên GPU
`sm_120` sẽ lỗi *no kernel image is available*. Muốn có kernel thật phải build từ nguồn (xem
bên dưới).

**Dự án chạy đầy đủ Giai đoạn 1–7 mà không cần `mamba-ssm`**, chỉ cần bật fallback:

```bash
export THREATMAMBA_SSM=simple      # nên thêm vào ~/.bashrc
```

Fallback là SSM thuần PyTorch: train chậm hơn nhưng đúng về mặt tính toán.

### Tùy chọn — build kernel mamba-ssm thật

```bash
bash scripts/setup_mamba.sh              # chỉ build cho sm_120, nhanh nhất
MAX_JOBS=2 bash scripts/setup_mamba.sh   # nếu build bị kill vì hết RAM
```

Script lo cả ba điều kiện mà bản cài tay hay vấp: cài **CUDA Toolkit 12.8** nếu `nvcc` hiện
tại cũ hơn (bản 12.4 của apt **không** sinh được mã `sm_120`), cài **g++-14/13** nếu `g++`
hệ thống mới hơn 14 (CUDA 12.8 không nhận g++ 15), và vá `setup.py` để chỉ build đúng
`sm_120` thay vì cả chục kiến trúc. Build mất 10–40 phút.

### Ollama và API key

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b        # ~5 GB

cp .env.example .env && $EDITOR .env
```

`VT_API_KEY`: virustotal.com → đăng ký → click avatar → **API key**.
`OTX_API_KEY`: otx.alienvault.com → đăng ký → **Settings** → OTX Key.

Nếu đã cài Ollama trên Windows thì không cần cài lại trong WSL — chỉ cần trỏ `OLLAMA_HOST`
sang máy Windows.

### Kiểm tra môi trường

```bash
source .venv/bin/activate
bash scripts/smoke_test.sh
```

Đạt khi tất cả 11 mục đều PASS (0 FAIL, 0 SKIP): `env`, `gpu-driver`, `python`, `toolchain`,
`cuda`, `mamba`, `ollama`, `securebert`, `vt`, `otx`, `rapiddns`.

Mục `mamba` PASS ở cả hai trường hợp — kernel thật hoặc fallback — và in rõ đang chạy chế độ
nào; chỉ FAIL khi cả hai đều hỏng. Lần chạy đầu sẽ tải SecureBERT (~500 MB) và nạp Qwen vào
VRAM, mất một hai phút. Có thể chạy riêng từng mục:

```bash
python scripts/checks/check_cuda.py
python scripts/checks/check_apis.py vt
```

---

## Chạy pipeline

Mỗi giai đoạn chạy độc lập được. Lệnh `bash scripts/run_all.sh` chạy tuần tự tất cả, nhưng
Giai đoạn 5 mất nhiều giờ nên nên chạy tay từng bước một lần trước.

### Giai đoạn 1 — dữ liệu

```bash
bash scripts/download_data.sh       # CSV MuscleFish + enterprise-attack-14.1.json
python -m src.data_prep all         # prepare -> prepare-d1 -> attck -> demo-subset -> stats
```

Trong 10 file của repo dữ liệu gốc chỉ 3 file thực sự cần: `CTI2Attacker.csv` (D2, cho
classifier), `CTI2TTPs.csv` (D1, cho module TTP) và `TTP_Group_Contribution.csv` (bảng đóng
góp của tác giả, dùng đối chiếu Fig. 5). Các file `.xlsx` trùng nội dung với CSV.

*Nghiệm thu:* bảng thống kê hợp lý cộng dòng `[ATT&CK v14] … -> PASS`.

### Giai đoạn 2 — daemon làm giàu IOC

```bash
bash scripts/run_enrichment_daemon.sh          # start (chạy nền, resumable)
bash scripts/run_enrichment_daemon.sh status   # xem tiến độ (= nghiệm thu)
bash scripts/run_enrichment_daemon.sh stop
```

Daemon đọc `data/demo_subset.txt` và chạy Algorithm 1: xếp hạng hunting method bằng UCT
(Eq. 1) → gọi VirusTotal / OTX / RapidDNS → Qwen3-8B chấm điểm candidate theo prompt Fig. 2
(validate JSON, retry 3 lần) → nhận candidate điểm ≥ 6. Daemon tôn trọng rate-limit của
VirusTotal (15 s/request, 480 request/ngày), checkpoint sau **mỗi IOC**, và ghi log vào
`outputs/enrich.log`. Hết tập demo thì tự mở rộng sang tập train.

*Nghiệm thu (sau ~24 giờ):* `python scripts/check_enrichment.py` báo ≥ 5 tài liệu có node
Time/Geo-location và không có crash không tự phục hồi.

> **Quy tắc cứng:** dữ liệu enriched **không bao giờ** trộn vào tập train của classifier. Nó
> chỉ dùng cho giao diện demo và để minh họa đầy đủ ontology 19 loại node.

Nên khởi động daemon sớm rồi làm Giai đoạn 3–6 song song.

### Giai đoạn 3 — trích TTP

```bash
python -m src.ttp_extract encode     # precompute embedding SecureBERT cho D1 (có cache)
python -m src.ttp_extract train      # BCE có pos_weight, early stopping theo macro AP
python -m src.ttp_extract eval       # NGHIỆM THU: tactics micro-F1 >= 0,70
```

Dùng thử ngay: `python -m src.ttp_extract extract --text "APT29 used spearphishing..."`.

### Giai đoạn 4 — dựng CSKG

```bash
python -m src.cskg_builder build --split all --mode train      # dùng cho classifier
python -m src.cskg_builder stats                               # NGHIỆM THU
python -m src.cskg_builder build --split all --mode enriched   # cho tài liệu đã enrich
python -m src.cskg_builder render --doc $(head -1 data/demo_subset.txt)
```

*Nghiệm thu:* số node và cạnh mỗi đồ thị cùng bậc độ lớn với bài báo.

### Giai đoạn 5 — train classifier

Cách được khuyến nghị: train cả 4 cấu hình bằng đúng một config, một seed, một tiêu chí
dừng, rồi tự động đánh giá:

```bash
bash scripts/train_all.sh
```

Hoặc chạy tay từng cấu hình:

```bash
python -m src.train                  # main
python -m src.train --no-mamba       # ablation bỏ MAMBA
python -m src.train --no-cl          # ablation bỏ contrastive learning
python -m src.train --no-iochunter   # ablation bỏ IOCHunter (bỏ cạnh co-occur/hunting)
```

Theo dõi tiến độ: `outputs/history_{tag}.csv`. Checkpoint: `outputs/model_{tag}.pt`.

#### Đọc log train

```
[main] ep  12 | lam 0.1  | loss 0.3021 (bce 0.2870 cl 1.5090) | tr bal 0.6120 | val micro 0.5410 macro 0.3520 bal 0.3710 | pred 18/20  *
```

| Cột | Ý nghĩa | Dấu hiệu xấu |
|---|---|---|
| `lam` | λ đang áp dụng (0 trong giai đoạn warm-up) | — |
| `bce` / `cl` | hai thành phần loss tách riêng | `λ·cl` lớn hơn `bce` → CL đang át BCE |
| `cl` | InfoNCE. Mức ngẫu nhiên = ln(K+1) = **1,6094** với K=4 | đứng yên quanh 1,61 → CL không học được gì |
| `pred` | số lớp model **thực sự** dự đoán | ≤ 5/20 → model gần như suy biến |
| `*` | epoch tốt nhất tới hiện tại, đã lưu checkpoint | — |

#### Ba lỗi đáng biết

Lần chạy đầu cho kết quả ngược đời: ablation `no_cl` đạt macro-F1 0,3515 (**đạt**) trong khi
model chính chỉ 0,0389 (**trượt**). Nguyên nhân là cấu hình, không phải phương pháp:

1. **`cl_lambda` = 1,0 làm contrastive learning át BCE.** Với K=4 mẫu âm, InfoNCE lúc khởi
   tạo ≈ ln(5) = 1,61 trong khi BCE ≈ 0,35 — hơn 80% gradient đến từ contrastive nên model
   không học phân loại. Mặc định mới: **0,1**.
2. **Early stopping theo val macro-F1 quá nhiễu.** 20 lớp trên ~250 mẫu validation, nhiều
   lớp chỉ 2–4 mẫu: `main` đạt "kỷ lục" ở epoch 2 do may rồi dừng ở epoch 7, trong khi
   `no_cl` chạy 41 epoch ⇒ bảng so sánh ablation vô nghĩa. Nay chọn checkpoint theo
   **balanced accuracy** (macro recall) — vẫn quan tâm đến lớp hiếm như macro-F1 nhưng mượt
   hơn, vì không có số hạng precision tụt về 0 khi một lớp không được dự đoán lần nào. Thêm
   `patience: 10` và `min_epochs: 20`.
3. **CL không có warm-up.** Nay `cl_warmup_epochs: 5` epoch đầu chỉ chạy BCE.

Bằng chứng củng cố: `D_separ` của `main` là 0,719 (< 1, các cụm chồng lên nhau) so với 1,734
của `no_cl`. Contrastive learning tồn tại để *tăng* tách bạch; bật lên lại làm tệ đi 2,4 lần
là dấu hiệu chắc chắn rằng vấn đề nằm ở λ.

#### Grid search λ

```bash
bash scripts/grid_cl.sh                    # thử {0.1, 0.5, 1.0}
LAMS="0.05 0.1 0.2" bash scripts/grid_cl.sh
bash scripts/grid_cl.sh --epochs 40        # chạy nhanh hơn
```

Kết quả gom vào `outputs/grid_cl_summary.csv`. Sau khi chọn được λ, ghi vào
`configs/default.yaml` (`train.cl_lambda`) rồi chạy lại `bash scripts/train_all.sh` để cả 4
ablation dùng chung giá trị đó.

### Giai đoạn 6 — đánh giá và giải thích

```bash
python -m src.evaluate all --robustness --validity
python -m src.explain group-profile     # heatmap kiểu Fig. 5
python -m src.explain attck-match       # bảng kiểu Table XI
python -m src.explain doc --doc d00042  # đóng góp từng node cho một tài liệu
```

Xuất ra `outputs/`: `metrics_all.csv` (F1 micro/macro, Top-1/3/5), `robustness_{tag}.csv`
(5 mốc timeline cùng hệ số A/B của Eq. 27), `validity_{tag}.csv`
(D_intra/D_inter/D_separ), `tsne_{tag}.png`, `fig5_heatmap.png` +
`fig5_group_ttp_contribution.csv`, `tableXI_attck_match.csv`.

---

## Giao diện demo

```bash
cd <thư mục gốc repo>          # BẮT BUỘC
streamlit run app/streamlit_app.py
```

Phải chạy từ thư mục gốc thì Streamlit mới đọc được `.streamlit/config.toml` (chủ đề tối).
Mở địa chỉ Streamlit in ra, mặc định `http://localhost:8501`. Từ WSL2 vẫn mở được bằng trình
duyệt Windows.

Ba file quyết định giao diện: `.streamlit/config.toml` (màu cho widget gốc), `app/ui_kit.py`
(CSS và các thành phần tự viết), `app/streamlit_app.py` (bố cục 4 trang).

Đồ thị pie-node cũng theo chủ đề tối, nhưng **`app/pie_node_graph.html` là template gốc của
tác giả, giữ nguyên từng byte** (kể cả các comment tiếng Trung — đây là mã của bên thứ ba và
`scripts/download_data.sh` tải lại nó từ nguồn). Việc đổi màu làm bằng cách thay chuỗi lúc
render (`_DARK_SWAP` trong `src/inference.py`), nên bản template trên đĩa vẫn đối chiếu trực
tiếp được với bản gốc.

Sidebar có khối **System status**: thiết bị (CUDA/CPU), số tài liệu Giai đoạn 1, module TTP
đã có chưa, số checkpoint, số tài liệu enriched, và **nghiệm thu Giai đoạn 5 PASS/FAIL** đọc
trực tiếp từ `outputs/metrics_all.csv`.

**Trang 1 — Phân tích báo cáo CTI.** Chọn tài liệu từ tập demo (đánh dấu ✓ nếu đã enrich)
hoặc dán báo cáo bất kỳ. Hiển thị tiến trình 4 bước, bar chart Top-5 kèm xác suất, bảng node
đóng góp cao nhất (Eq. 21–23), bảng TTP và bảng IOC (cột `Source` cho biết IOC lấy từ văn
bản hay từ hunting function nào), và **đồ thị CSKG pie-node** dùng lại template ECharts của
chính tác giả — node State là biểu đồ tròn tỉ lệ đóng góp TTP, kích thước node là mức quan
trọng, độ đậm cạnh là mức đóng góp (giống Fig. 6c). Checkbox "Live enrichment" gọi IOCHunter
thật (tốn quota VirusTotal, có cảnh báo trước).

**Trang 2 — Độ bền theo thời gian.** Kéo slider 20→100% timeline; hệ thống cắt báo cáo, chạy
lại suy luận và cập nhật Top-5 ngay. Line chart theo dõi xác suất top-3 qua các mốc. Kết quả
từng mốc được cache nên slider phản hồi tức thì.

**Trang 3 — Hồ sơ nhóm tấn công.** Bar chart các TTP đóng góp cao nhất của nhóm, kèm
Jaccard/F1 so với trang MITRE ATT&CK Groups và bảng đối chiếu với dữ liệu gốc của tác giả.

**Trang 4 — Kết quả tổng hợp.** Chỉ hiển thị (không tính lại) mọi thứ trong `outputs/`:
metrics và ablation, robustness, validity, t-SNE, heatmap, Table XI, đường học.

Nghiệm thu:

```bash
python tests/test_ui_smoke.py
```

Script chạy thật cả 4 trang bằng `AppTest` và đo thời gian: **1 tài liệu mới < 60 giây**
(không live-enrichment) và **mỗi mốc slider < 2 giây**. Trang nào thiếu dữ liệu giai đoạn
trước sẽ hiện cảnh báo và dừng — đó là hành vi đúng, không tính là lỗi.

Chọn tài liệu để demo trực tiếp:

```bash
python scripts/pick_demo_docs.py -n 5
```

Script chấm điểm toàn bộ tập test rồi chọn ra các tài liệu quy kết đúng, có biên độ top-1 so
với top-2 rộng, đồ thị đủ dày, thuộc các nhóm khác nhau, và vẫn đúng ở mốc 60% timeline.

---

## Cách đọc kết quả

Có hai cảnh báo tự động cần đọc kỹ trước khi đưa bất kỳ con số nào vào báo cáo.

**`!! FAKE ROBUSTNESS`.** Một model suy biến dự đoán gần như hằng số, nên cắt timeline không
làm gì thay đổi: đường Table VIII phẳng tuyệt đối và hệ số A ≈ 0, trông *như thể* cực kỳ bền
vững. Đây là bẫy diễn giải nguy hiểm nhất của pipeline này — A nhỏ ở đây nghĩa là model
**bỏ qua đầu vào**, chứ không phải nó bền vững. Ví dụ thật đã gặp: `main` cho top-3 = 0,4499
y hệt nhau ở cả 5 mốc (20/40/60/80/100%), A = 0,0319; trong khi `no_cl` đi từ 0,391 lên
0,539 với A = 0,1734 — dốc hơn nhưng là đường cong *thật*. Khi thấy cảnh báo này, bỏ bảng đó
đi.

**`!! ABLATION … BEATS the main model`.** Nếu bỏ một thành phần mà kết quả tốt lên đáng kể
thì gần như chắc chắn thành phần đó đang bị bật sai (ví dụ `cl_lambda` quá cao), chứ không
phải nó vô dụng. Đây là lỗi cấu hình, không phải phát hiện khoa học — sửa cấu hình rồi chạy
lại trước khi kết luận.

Cột `n_pred_cls` trong `metrics_all.csv` cho biết model thực sự dự đoán bao nhiêu lớp; con
số này thấp là dấu hiệu sớm nhất của suy biến.

Với bảng validity, chỉ dùng ba cột đầu: chúng được đo trên **V_G** — vector biểu diễn đồ thị
đưa vào classifier, đúng như bài báo. Ba cột `_z` đo trên projection head của contrastive,
mà InfoNCE chuẩn hóa L2 nên các vector đó chỉ giữ thông tin về góc; đo khoảng cách Euclid
trên chúng là vô nghĩa trong ngữ cảnh này.

---

## Cấu trúc repo

```
threatmamba-repro/
├── README.md / README.vi.md / LICENSE / CITATION.cff
├── requirements.txt / .env.example
├── configs/default.yaml          # hyperparams Table V + tham số Giai đoạn 1-7
├── scripts/
│   ├── setup_env.sh              # GĐ 0: cài môi trường
│   ├── setup_mamba.sh            # GĐ 0: build mamba-ssm từ nguồn (tùy chọn)
│   ├── smoke_test.sh             # GĐ 0: NGHIỆM THU (+ checks/)
│   ├── download_data.sh          # GĐ 1: tải dữ liệu
│   ├── run_enrichment_daemon.sh  # GĐ 2: start/stop/status
│   ├── check_enrichment.py       # GĐ 2: NGHIỆM THU
│   ├── prune_enrich_state.py     # GĐ 2: dọn state daemon có chọn lọc
│   ├── train_all.sh              # GĐ 5: 4 ablation cùng config -> GĐ 6
│   ├── grid_cl.sh                # GĐ 5: grid search λ
│   ├── pick_demo_docs.py         # GĐ 7: chọn tài liệu demo chắc chắn đúng
│   ├── patch_cuda_glibc.py       # vá xung đột header CUDA / glibc >= 2.41
│   ├── reset_env.sh              # dọn sạch để cài lại
│   └── run_all.sh                # trình tự chuẩn GĐ 1->7
├── src/
│   ├── utils.py, encoder.py, ssm.py                    # tiện ích, SecureBERT, SSM (+fallback)
│   ├── data_prep.py                                    # Giai đoạn 1
│   ├── ioc_extract.py, ioc_hunter/                     # Giai đoạn 2 (uct, llm_agent, apis, runner)
│   ├── ttp_extract.py                                  # Giai đoạn 3
│   ├── cskg_builder.py                                 # Giai đoạn 4
│   ├── contrastive.py, model.py, losses.py, train.py   # Giai đoạn 5
│   ├── evaluate.py, explain.py                         # Giai đoạn 6
│   └── inference.py                                    # Giai đoạn 7 (luồng suy luận dùng chung)
├── .streamlit/config.toml        # chủ đề SOC tối màu
├── app/
│   ├── streamlit_app.py          # UI 4 trang
│   ├── ui_kit.py                 # CSS + thành phần giao diện tự viết
│   └── pie_node_graph.html       # template pie-node của tác giả (giữ nguyên bản gốc)
├── tests/
│   ├── test_cpu_pipeline.py      # 13 mục, không cần GPU
│   └── test_ui_smoke.py          # NGHIỆM THU Giai đoạn 7
├── data/{raw,processed,cskg,enriched,attck,reference}/
└── outputs/                      # checkpoint, bảng CSV, hình PNG, log daemon
```

---

## Kiểm thử

```bash
THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_cpu_pipeline.py
```

13 mục kiểm tra toàn bộ đường ống tensor (IOC → CSKG train/enriched → cắt timeline → mẫu
contrastive → cả 4 cấu hình model → loss → metrics → giải thích → Gaussian attention → UCT →
LLM agent). Dùng để bắt lỗi nhanh sau khi sửa code; **không thay thế** nghiệm thu thật trên
máy đích.

Hai biến môi trường dành cho debug: `THREATMAMBA_FAKE_ENCODER=1` (embedding giả tất định
thay SecureBERT) và `THREATMAMBA_SSM=simple` (SSM thuần PyTorch thay mamba-ssm — cũng chính
là fallback khi chạy thật).

---

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `torch.cuda.is_available() == False` | Cập nhật driver NVIDIA trên Windows; `wsl --shutdown` rồi vào lại; cài lại: `pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| `no kernel image is available… sm_120` khi gọi Mamba | `mamba-ssm` đã cài không có kernel cho Blackwell (wheel dựng sẵn chỉ tới sm_100). Gỡ đi (`pip uninstall mamba-ssm causal-conv1d`) rồi `bash scripts/setup_mamba.sh`, hoặc dùng `THREATMAMBA_SSM=simple` |
| `NameError: name 'bare_metal_version' is not defined` | Build không thấy `nvcc` → cài CUDA Toolkit ≥ 12.8 và dùng `--no-build-isolation`. `scripts/setup_mamba.sh` làm sẵn |
| `x86_64-linux-gnu-g++ (15.x) is greater than the maximum required version by CUDA` | CUDA không nhận g++ 15 → cài `g++-14`/`g++-13` và đặt `CUDAHOSTCXX`. `scripts/setup_mamba.sh` làm sẵn |
| `exception specification is incompatible with that of previous function` | Xung đột glibc ≥ 2.41 với header CUDA → `sudo python3 scripts/patch_cuda_glibc.py` |
| `Precompiled wheel not found` rồi build lỗi | Không có wheel cho Python/torch của bạn (hay gặp nhất với Python 3.14) → cài lại env bằng `scripts/setup_env.sh` để về Python 3.11 |
| `Command 'pip' not found` | Chưa activate venv: `source .venv/bin/activate` |
| Python trong venv là 3.14 | `bash scripts/reset_env.sh && bash scripts/setup_env.sh` — script ép đúng 3.11 |
| ollama `connection refused` | `ollama serve` hoặc `sudo systemctl start ollama` |
| qwen3:8b trả JSON sai dai dẳng | Đã retry 3 lần; cập nhật Ollama bản mới. Vẫn tệ → thêm few-shot vào prompt trong `src/ioc_hunter/llm_agent.py` hoặc đổi model cùng cỡ |
| VT 401 / OTX 401-403 | Key sai hoặc tài khoản chưa kích hoạt email |
| VT 429 liên tục | Hết quota ngày; daemon tự bỏ qua, ghi log và hôm sau chạy tiếp — không cần can thiệp |
| RapidDNS fail | Cloudflare chặn tạm, thử lại sau; không ảnh hưởng phần khác |
| CSKG chỉ ~13 node mỗi đồ thị | Chưa train Giai đoạn 3, hoặc build với `--ttp none` → train Giai đoạn 3 rồi build lại |
| Macro-F1 quá thấp | Kiểm tra leakage và label map → chạy `--no-cl` xem baseline BCE thuần → cân nhắc giảm còn 10–12 nhóm nhiều dữ liệu nhất |
| `main` kém hơn hẳn `no_cl` | λ quá cao, CL đang át BCE. Xem [Ba lỗi đáng biết](#ba-lỗi-đáng-biết); chạy `bash scripts/grid_cl.sh` |
| Log train: cột `cl` đứng yên quanh **1,6094** | InfoNCE ở mức ngẫu nhiên (ln(K+1), K=4) — CL không học được gì. Giảm `cl_temperature` hoặc tăng `cl_pairs_K`; nếu vẫn vậy thì mẫu âm quá dễ |
| Log train: `pred 3/20` | Model gần như suy biến. Hạ `cl_lambda`, tăng số epoch, kiểm tra CSKG đã có node TTP chưa |
| Train dừng quá sớm (< 15 epoch) | Chỉnh `early_stopping_patience` / `min_epochs` trong `configs/default.yaml`; bảo đảm mọi ablation chạy qua `scripts/train_all.sh` |
| Bảng robustness phẳng tuyệt đối, A ≈ 0 | **Không phải** bền vững mà là model suy biến — xem [Cách đọc kết quả](#cách-đọc-kết-quả) |
| `Missing key(s) in state_dict: "ssm.layers.0.conv.weight"` … `Unexpected key(s): "ssm.layers.0.A_log"` | Kiến trúc SSM không khớp checkpoint. `unset THREATMAMBA_SSM` rồi chạy lại. Biến này chỉ dành cho debug và chỉ có tác dụng khi `model.ssm_fallback: auto` |
| UI báo thiếu `outputs/model_main.pt` | Chạy Giai đoạn 5 |
| UI vẫn nền trắng | Chạy `streamlit run` từ **thư mục gốc** repo để đọc được `.streamlit/config.toml` |
| UI trang 3/4 trống | Chạy Giai đoạn 6 (`src.evaluate`, `src.explain`) |
| Đồ thị pie-node không hiện | Cần mạng để tải ECharts từ CDN jsdelivr; kiểm tra console trình duyệt |
| Lỗi `bash\r` | File bị chuyển sang CRLF: `sed -i 's/\r$//' scripts/*.sh scripts/checks/*.py` |

---

## Phạm vi và điều chỉnh so với bài báo

Đây là bản tái hiện độc lập, không phải mã nguồn của tác giả. Các điều chỉnh có chủ ý:

- **Mô hình ngôn ngữ.** Bài báo dùng DeepSeek-R1-70B cho IOCHunter-LLM; bản này dùng
  **Qwen3-8B** qua Ollama, vừa với 16 GB VRAM.
- **Gaussian attention.** Prototype Gaussian theo nhãn dùng Σ **chéo** thay vì ma trận hiệp
  phương sai đầy đủ.
- **Điểm đóng góp node.** Eq. 21–23 được xấp xỉ bằng gradient × input trên feature của node
  đối với logit của lớp được dự đoán.
- **Sàng lọc candidate sau khi gọi API.** `iochunter.post_screen` thêm một bước cho LLM chấm
  điểm các candidate mà API trả về. Bài báo không có bước này (mọi kết quả hunting đều vào
  CSKG); ở đây bật lên vì tập CTI này có rất nhiều domain tham chiếu lành tính. Có thể tắt
  trong `configs/default.yaml`.
- **Siêu tham số bài báo không nêu.** Bài báo không nêu λ (Eq. 18), τ (Eq. 20) và K. Mặc
  định ở đây là λ = 0,1, τ = 0,5, K = 4, chọn qua `scripts/grid_cl.sh`.
- **SSM fallback.** Khi không có `mamba-ssm`, hệ thống dùng `SimpleSSM` thuần PyTorch. Nó
  đúng về tinh thần nhưng không phải kernel CUDA của bài báo; hãy ghi rõ điều này nếu công
  bố số liệu chạy ở chế độ đó.
- **Table XI.** Cột Qi'anxin trong Table XI của bài báo được lược bỏ (nguồn dữ liệu đó không
  công khai).

---

## Trích dẫn

Nếu bạn dùng repo này, xin trích dẫn bài báo gốc:

```bibtex
@article{threatmamba2026,
  title   = {ThreatMAMBA},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  doi     = {10.1109/TIFS.2026.3685967}
}
```

và, nếu muốn, cả bản tái hiện này — xem [CITATION.cff](CITATION.cff).

---

## Ghi nhận và giấy phép

Công trình này dựa trên:

- bài báo **ThreatMAMBA** (IEEE TIFS 2026, DOI 10.1109/TIFS.2026.3685967);
- tập dữ liệu và template đồ thị pie-node từ
  [`MuscleFish/ThreatMAMBA`](https://github.com/MuscleFish/ThreatMAMBA) (MIT) —
  `app/pie_node_graph.html` chính là file `pie-node-ttp-state-graph.temp.html` của repo đó,
  giữ nguyên không sửa;
- [`ehsanaghaei/SecureBERT`](https://huggingface.co/ehsanaghaei/SecureBERT);
- [`state-spaces/mamba`](https://github.com/state-spaces/mamba) và
  [`Dao-AILab/causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d);
- [MITRE ATT&CK v14](https://github.com/mitre-attack/attack-stix-data);
- [`ioc-finder`](https://github.com/ioc-fang/ioc-finder), [Ollama](https://ollama.com) /
  Qwen3, VirusTotal, AlienVault OTX và RapidDNS.

Mã nguồn trong repo này phát hành theo [Giấy phép MIT](LICENSE). Các tập dữ liệu, mô hình và
dịch vụ bên thứ ba nêu trên giữ giấy phép và điều khoản sử dụng riêng; hãy kiểm tra trước
khi phân phối lại dữ liệu hoặc kết quả.

**Sử dụng có trách nhiệm.** Đây là công cụ nghiên cứu an ninh phòng thủ, dùng để phân tích
các báo cáo threat intelligence đã công bố. Quy kết tấn công mang tính xác suất và có thể
sai — đầu ra của hệ thống này là công cụ hỗ trợ phân tích, không phải bằng chứng.
