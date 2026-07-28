# PatchCore 주석 소스

이 폴더는 [Amazon Science의 공식 PatchCore 구현](https://github.com/amazon-science/patchcore-inspection)에서 학습·추론 흐름을 읽는 데 필요한 Python 소스만 분리한 사본임.

- 기준 커밋: `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`
- 원본 라이선스: Apache License 2.0. 전문은 [LICENSE](LICENSE), 고지문은 [NOTICE](NOTICE)에 포함함.
- 변경 사항: PatchCore의 핵심 단계와 Windows 실행 호환 처리를 설명하는 한국어 주석을 추가함.

이 폴더에는 포함하지 않음:

- MVTec AD 데이터셋
- Python 가상환경(`.venv`)
- 실행으로 생성한 모델 파일, 캐시, 로그, 대량 결과 이미지

전체 실행 방법과 재현 결과는 [실험 설정 문서](../../markdown/PatchCore_reproduction_setup.md)를 참고하면 됨.
