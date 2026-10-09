# PatchCore 주석 소스

이 폴더에는 [Amazon Science의 공식 PatchCore 구현](https://github.com/amazon-science/patchcore-inspection) 중 학습·추론 흐름을 읽는 데 필요한 Python 파일만 넣었음.

- 기준 커밋: `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`
- 원본 라이선스: Apache License 2.0. 원문은 [LICENSE](LICENSE), 고지문은 [NOTICE](NOTICE)에 포함됨.
- 변경 사항: 코드의 핵심 단계와 Windows 실행 호환 처리를 설명하는 한국어 주석을 추가함.

제외한 항목:

- MVTec AD 데이터셋
- Python 가상환경(`.venv`)
- 학습·추론으로 생성된 모델 파일과 캐시·로그·대량 결과 이미지

전체 실행 방법과 재현 결과는 [상위 실험 문서](../../markdown/PatchCore_reproduction_setup.md)를 참고하면 됨.
