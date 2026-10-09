# RD4AD 실행 인수인계

이 파일은 다른 ChatGPT/Codex 계정에서 현재 RD4AD 재현 작업을 이어가기 위한 상태 기록임. 새 계정에서 이 파일을 첨부하거나, 파일을 읽어 달라고 요청하면 됨.

## 작업 위치

| 항목 | 위치 |
|---|---|
| 작업 루트 | `C:\Users\test\Desktop\Codex` |
| Git repository | `C:\Users\test\Desktop\Codex\lab-m5hfg` |
| GitHub repository | `TUK-AI-HC-LAB/lab-m5hfg` |
| WSL MVTec AD 데이터 | `/home/test/data/mvtec` |
| WSL Python 환경 | `/home/test/miniforge3/envs/patchcore-gpu` |
| 공식 RD4AD local clone | `C:\Users\test\Desktop\Codex\lab-m5hfg\method3\source\rd4ad` |

## 현재 실행 상태 (2026-08-19 확인)

- 논문 본문 조건 재실행이 진행 중임.
- 완료 범주: `carpet`, `bottle`, `hazelnut` (3 / 15).
- 현재 범주: `leather`.
- 실행기: [`run_rd4ad_mvtec_paper_protocol_wsl.sh`](../source/run_rd4ad_mvtec_paper_protocol_wsl.sh).
- 실행 결과 폴더: `method3/source/results/rd4ad_paper_protocol_runs/`.
- 학습은 WSL 프로세스에서 실행 중이므로, Codex 계정 자체와는 독립적임. 그러나 PC·WSL 재부팅은 실행을 중단시킬 수 있음.

## 논문 본문에 맞춘 설정

- backbone: ImageNet 사전학습 WideResNet-50.
- 입력: 256 x 256.
- batch size: 16.
- optimizer: Adam, learning rate 0.005, beta `(0.5, 0.999)`.
- 학습: 범주별 200 epoch, 10 epoch마다 평가.
- anomaly map: Gaussian smoothing sigma 4.
- 데이터: MVTec AD 15개 범주, 정상 train 이미지만 학습.
- seed: 111 (공식 코드 기본값).

GPU·CUDA·PyTorch 버전은 저자 환경과 다르므로, 저자와 bitwise 동일한 재현은 아님. 다만 논문 본문에 명시된 학습·평가 조건은 맞춘 실행임.

## 로컬 공식 코드 수정 범위

- `main.py`: WSL data path, category, epoch, 결과 CSV를 command option으로 제어하도록 추가.
- `test.py`: 최신 NumPy/pandas 호환을 위해 `np.bool_` 및 `df.loc[...]` 사용.
- teacher·OCBE·decoder 구조, loss, optimizer 값, 200 epoch 조건은 변경하지 않음.

## 완료 뒤 해야 할 일

1. 15개 CSV가 완성됐는지 확인.
2. 다음 명령으로 최종 표를 생성.

```powershell
python C:\Users\test\Desktop\Codex\lab-m5hfg\method3\source\collect_rd4ad_results.py
```

3. 결과를 새 파일 `method3/source/results/RD4AD_MVTecAD_WR50_paper_protocol_results.csv`에 별도로 저장하도록 취합기를 확장. 기존 [`RD4AD_MVTecAD_WR50_results.csv`](../source/results/RD4AD_MVTecAD_WR50_results.csv)는 이전 실행 결과이므로 덮어쓰지 않음.
4. [`RD4AD_reproduction_results.md`](RD4AD_reproduction_results.md)와 [`2026-W34_brief.md`](../../meetings/2026-W34_brief.md)를 새 실행의 실제 수치·근거 경로로 업데이트.
5. 사용자가 요청할 때만 GitHub에 commit, push, PR 병합.

## 실행 상태 확인 명령

```powershell
wsl.exe -d Ubuntu -- /bin/bash -c "ps -eo pid,etime,cmd | grep -E '[r]un_rd4ad_mvtec_paper_protocol|[p]ython main.py' || true"
```

## 새 계정에서 사용할 요청문

```text
C:\Users\test\Desktop\Codex\lab-m5hfg\method3\markdown\RD4AD_execution_handoff.md를 읽고, 진행 중인 RD4AD 논문 조건 재실행을 이어서 관리해줘. 실행이 끝났다면 결과를 별도 CSV로 취합하고 W34를 업데이트해줘. GitHub 업로드는 내가 요청할 때만 해.
```
