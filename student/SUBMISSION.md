# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Hoàng Sơn
- MSSV: 2A202602457
- Email: son1012005@gmail.com
- Link repo (fork): https://github.com/sown101/K4-L2L3-DAY23-NguyenHoangSon-2A202602457-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 86a5c9061a962542c322875bb85bb5c1948a42cf

## Tóm tắt kết quả

- `fusion_mode`: `compare`.
- `frames`: `[0, 198]`, tính cả hai đầu, 199 frame mỗi mode.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- `seed`: `0`.
- Môi trường chạy: Python 3.12.15, PyTorch CPU, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`.
- Nguồn số liệu: sáu file do runner sinh trong `student/artifacts/`; commit CP5 `fd0b361`.

| Detection, dùng chung cho hai mode | Giá trị |
|---|---:|
| `precision` | 0.9700934579439252 |
| `recall` | 0.7004048582995951 |
| `tp` | 519 |
| `fp` | 16 |
| `fn` | 222 |

| Tracking | LiDAR | LiDAR + camera |
|---|---:|---:|
| `rmse` (m) | 0.15032268781360134 | 0.1358667883353908 |
| `matches` | 502 | 502 |
| `sum_sq_err` (m²) | 11.343649056695735 | 9.26681165463209 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 239 | 239 |
| `mean_confirmed_tracks` | 2.522613065326633 | 2.522613065326633 |
| `precision_track = matches / (matches + ghost_track_frames)` | 1.0 | 1.0 |
| `coverage = matches / detection.tp` | 0.9672447013487476 | 0.9672447013487476 |

Hai tỷ lệ cuối được tính từ các bộ đếm trong metrics, theo rubric; chúng không
phải trường bổ sung trong `metrics.json`.

Fusion giảm RMSE khoảng 0.014456 m, tương đương 9.6166%, với cùng 502 cặp ghép,
0 ghost và 239 miss. Sự giảm RMSE không đi kèm giảm số cặp được đánh giá. Log
không lưu danh tính từng cặp track–GT nên không khẳng định toàn bộ cặp ghép
giống hệt nhau giữa hai mode. Cả hai đạt RMSE ≤ 0.45 m, precision_track ≥ 0.75,
coverage ≥ 0.70; `rmse_fused - rmse_lidar = -0.014455899478210549 m ≤ 0.05 m`.

Tổng nhãn xe hợp lệ theo frame là 741 ở mỗi mode; `502 + 239 = 741`. Miss trong
tracking cũng phản ánh giai đoạn track chưa confirmed và các nhãn detector có
thể bỏ sót, nên không quy toàn bộ miss cho EKF. Ví dụ ở frame 0, cả hai mode có
`confirmed=0`, `matches=0`, `misses=2`; đến frame 4 có `confirmed=2`, `matches=2`,
`ghosts=0`, `misses=0`. Fusion không tốt hơn ở mọi frame: tại frame 4,
`sum_sq_err` fused là 0.031987316784416 m², lớn hơn LiDAR
0.011281520740472877 m², dù tổng sai số toàn segment thấp hơn.

Camera dùng nhãn ground-truth FRONT cộng nhiễu Gaussian có độ lệch chuẩn
0.5 pixel trong `_front_observations`; covariance đo dùng trong EKF là
`diag(25, 25) pixel²` theo tham số lab. Hai đại lượng này khác nhau. Kết quả
chỉ thể hiện thử nghiệm fusion của lab, không chứng minh chất lượng detector ảnh.

Chạy từ root repo:

```powershell
. .\Activate-Lab.ps1
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

Đã kiểm tra bằng `fusion_lab.evaluation.validate_metrics_records`: log gộp có
398 record, mỗi log riêng có 199 record, đúng một record cho mỗi `(mode, frame)`.
Các tổng và RMSE được tính lại từ log khớp metrics; detection giống nhau giữa
hai mode và mọi record thoả các invariant. Không chỉnh sửa tay artifacts.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR có `z = [x_s, y_s, z_s]^T`, kích thước 3×1 và đơn vị mét. Covariance
   đo `R` là 3×3; tham số lab cho `R = diag(0.01, 0.01, 0.01) m²`. Mô hình đo
   LiDAR tuyến tính theo vị trí, với `H` kích thước 3×6. Camera có
   `z = [u, v]^T`, kích thước 2×1 và đơn vị pixel; `R = diag(25, 25) pixel²`.
   Phép chiếu pinhole phi tuyến nên camera dùng Jacobian `H` kích thước 2×6.
   Cả hai cập nhật cùng trạng thái 6D bằng `gamma = z - h(x)`,
   `S = H P H^T + R` và `K = P H^T S^-1`. Nhiễu đo phải ở đúng đơn vị của cảm biến.
   Dẫn chứng: `workspace/kalman.py::ekf_update`,
   `workspace/camera_fusion.py::build_camera_measurement` và
   `../platform/fusion_lab/tracking/sensors.py::Sensor.get_H`.

2. Vì sao cần gating Mahalanobis trước khi gán?

   Mahalanobis dùng `d² = gamma^T S^-1 gamma`, xét cả residual và độ bất định.
   Cùng một sai lệch vị trí, track có covariance lớn có thể cho `d²` nhỏ hơn
   track chắc chắn. Cổng `chi2.ppf(0.995, sensor.dim_meas)` loại cặp không phù hợp
   trước khi gán greedy, giảm khả năng cập nhật track bằng đo của xe khác.
   LiDAR dùng 3 bậc tự do, camera dùng 2. Ma trận cost đặt `inf` cho cặp bị loại;
   mỗi bước greedy chọn cặp hữu hạn nhỏ nhất rồi bỏ hàng/cột để giữ gán một-một.
   Kiểm tra FOV diễn ra trước Mahalanobis, tránh chiếu camera tại độ sâu không hợp lệ.
   Dẫn chứng: `workspace/association.py::mahalanobis_distance`, `chi2_gate`,
   `association_cost_matrix` và `pick_next_pair`.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Đây là track-then-fuse: một danh sách track nhận predict một lần mỗi frame,
   rồi association/update LiDAR và quản lý vòng đời, sau đó association/update
   camera trên cùng track. Không predict lại giữa hai cảm biến vì chúng thuộc
   cùng bước thời gian logic của Waymo Frame. Thứ tự thể hiện trong
   `../platform/fusion_lab/scripts/run_lab.py::run`: `KF.predict(track)`,
   lượt `associate_and_update(..., lidar_sensor)`, rồi lượt với `camera_sensor`.
   `grade_run_lidar.log` và `grade_run_fused.log` ghi hai mode trên cùng khoảng frame;
   `grade_run.log` gộp cả hai. Mỗi record chứa kết quả cuối frame, không ghi từng
   predict/update, nên bằng chứng về thứ tự xử lý là code runner. Các số detection
   phải giống nhau giữa hai mode; phần khác nhau là kết quả tracking.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Sai extrinsic hoặc intrinsics làm `h(x)` lệch khỏi pixel quan sát, có thể tạo
   residual có hướng lệch lặp lại qua nhiều frame. Sai lệch lớn làm Mahalanobis
   vượt cổng và camera bị từ chối; nếu covariance đủ lớn hoặc sai lệch còn nằm
   trong cổng, EKF vẫn nhận đo và có thể kéo trạng thái lệch, tăng RMSE hoặc đổi
   kết quả ghép track–GT. Cần so sánh residual trước/sau khi đổi calibration để
   xác định nguyên nhân. Đây là giải thích lý thuyết; bài này không thực hiện
   thí nghiệm làm lệch calibration và log hiện tại không lưu innovation.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng, không thể lấy cảm biến từ phần tử đầu danh sách. Tham số
   `sensor` cho manager biết đây là lượt LiDAR hay camera. Lượt LiDAR rỗng vẫn
   cần trừ score của track chưa ghép trong FOV và xét xoá; lượt camera rỗng không
   tác động score hay vòng đời. Đo LiDAR chưa ghép có thể tạo track với vị trí
   3D, còn camera trong lab chỉ tinh chỉnh trạng thái. `associate_and_update`
   luôn gọi `manager.manage_tracks(...)`; `TrackManager.handle_updated_track`
   chỉ ghi hit LiDAR, và `manage_tracks` trả về ngay khi cảm biến không phải LiDAR.
   Dẫn chứng: `workspace/association.py::associate_and_update` và
   `../platform/fusion_lab/tracking/manager.py::TrackManager`.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Khởi tạo có `score = 1/6`, trạng thái `initialized`, vận tốc bằng 0. Vị trí và
   covariance vị trí được chuyển từ hệ cảm biến sang hệ xe; covariance vận tốc
   dùng bình phương các sigma cấu hình. Mỗi hit LiDAR cộng `1/6`, tối đa 1;
   miss trong FOV trừ `1/6`. Hit xác nhận track khi `score > 0.8`; hit chưa đủ
   ngưỡng chuyển track chưa confirmed sang `tentative`. Track đã confirmed giữ
   trạng thái đó sau miss. Xoá khi `P[0,0] > 9` hoặc `P[1,1] > 9`, hoặc track
   confirmed có `score < 0.6`, hoặc track chưa confirmed có `score <= 0`.
   Các điều kiện xoá là OR; bằng ngưỡng 9 hoặc 0.6 chưa gây xoá. Camera không
   cộng/trừ score, không tạo và không xoá track. Dẫn chứng: ba hàm trong
   `workspace/track_management.py` và các test biên ở `tests/test_tracking_contract.py`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không thực hiện bonus; không có thí nghiệm calibration hoặc bằng chứng CVAT.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex.
- Dùng cho phần nào (hàm, câu hỏi, debug): Codex đọc và phân tích repo, thiết lập CP0, triển khai 17 hàm Part E–H ở CP1–CP4, hỗ trợ chạy Waymo và kiểm tra artifacts ở CP5, soạn bảng số liệu và phần giải thích của báo cáo CP6. Không sửa platform, test gốc hay sửa tay metrics/log.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Tôi kiểm tra lại bài theo các bước: chạy `python -m pytest student/tests -q`; chạy Waymo ở chế độ `compare` trên frame 0–198 với seed 0; kiểm tra ba cặp metrics/log bằng `validate_metrics_records`; tính lại RMSE, precision_track và coverage từ các bộ đếm; đối chiếu code với công thức EKF, Mahalanobis, phép chiếu pinhole và các ngưỡng rubric. Kết quả kiểm chứng của bài là 128 test passed, không có failed/xfailed; log có đủ 398 record cho hai mode và số liệu khớp metrics. Tôi đọc lại phần code và giải thích do AI hỗ trợ để hiểu cách triển khai.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; toàn bộ test ở CP4: 128 passed, không có `failed`/`xfailed`
- [x] Part A–D được giữ nguyên
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` ở CP5 `fd0b361` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
