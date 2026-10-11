# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

import os

import pandas as pd
import wandb
from loguru import logger

import utils
from component_registry import run_trainer_lifecycle


def train_and_collect_metrics(
    trainer, dataloaders, dataset_name, collect_metrics=True
):
    # 역할: `train_and_collect_metrics`에 해당하는 작업을 수행.
    # 매개변수: trainer, dataloaders, dataset_name, collect_metrics.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `run_trainer_lifecycle`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """Run the public Trainer lifecycle used by every registered method."""

    return run_trainer_lifecycle(
        trainer,
        dataloaders["training"],
        dataloaders["validation"],
        dataloaders["testing"],
        dataset_name,
        collect_metrics=collect_metrics,
    )


def run(
    methods,
    args
):
    # 역할: `run`에 해당하는 작업을 수행.
    # 매개변수: methods, args.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `utils.set_torch_device`, `logger.info`, `utils.create_storage_folder`, `enumerate`, `pd.DataFrame`입니다.
    # 제어 흐름: 반복문 5개, 조건 분기 8개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    results_path = args.results_path
    gpu = args.gpu
    seed_list = args.seed_list
    log_group = args.log_group
    log_project = args.log_project
    run_name = args.run_name

    methods = {key: item for (key, item) in methods}

    if log_group != "":
        run_save_path = utils.create_storage_folder(
            results_path, log_project, log_group, run_name, mode="overwrite"
        )
    else:
        run_save_path = None

    # NOTE 확인 결과, seed 는 data 에는 사용되지 않음
    list_of_dataloaders = methods["get_dataloaders"](0)

    device = utils.set_torch_device(gpu)

    result_collect = []
    # seed_i = 0  # NOTE TEMP
    for seed in seed_list:
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            logger.info(
                "Evaluating dataset [{}] ({}/{})...".format(
                    dataloaders["training"].name,
                    dataloader_count + 1,
                    len(list_of_dataloaders),
                )
            )

            # NOTE dataset 는 고정 seed 0 으로 처리하고, model 은 각각의 seed 로 처리함
            utils.fix_seeds(seed, device)

            dataset_name = dataloaders["training"].name

            imagesize = dataloaders["training"].dataset.imagesize
            simplenet_list = methods["get_simplenet"](imagesize, device)

            if run_save_path is not None:
                models_dir = os.path.join(run_save_path, "models")
                os.makedirs(models_dir, exist_ok=True)
            else:
                models_dir = None

            for i, SimpleNet in enumerate(simplenet_list):
                # torch.cuda.empty_cache()
                if SimpleNet.backbone.seed is not None:
                    utils.fix_seeds(SimpleNet.backbone.seed, device)
                logger.info(
                    "Training models ({}/{})".format(i + 1,
                                                     len(simplenet_list))
                )
                # torch.cuda.empty_cache()

                if models_dir is not None:
                    SimpleNet.set_model_dir(os.path.join(
                        models_dir, f"{i}"), dataset_name)

                setattr(args, "pixel_auroc", int(dataloaders['testing'].dataset.having_mask()))
                SimpleNet.seed = seed

                metrics = train_and_collect_metrics(
                    SimpleNet,
                    dataloaders,
                    dataset_name,
                    collect_metrics=True,
                )
                
                result_entry = {"dataset_name": dataset_name, "seed": seed}
                result_entry.update(metrics)
                result_collect.append(result_entry)

            # convert result_collect into a pandas dataframe
            result_df = pd.DataFrame(result_collect).set_index("dataset_name")

            logger.info("\n\n-----\n")
            logger.info({f"{dataset_name}_test_auroc_mean": metrics.get("auroc_mean", -1)})
            if args.wan:
                wandb.log({f"{dataset_name}_test_auroc_mean": metrics.get("auroc_mean", -1)})
                wandb.log({f"{dataset_name}_test_metrics": metrics})

            # print 시 생략 없게 설정
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

            print(result_df)
            df_path = os.path.join(
                args.results_path,
                f"{args.mainmodel}__{'_'.join(args.layers_to_extract_from)}_{args.csv_save_name}",
            )
            file_name = f"results_{args.mainmodel}"
            if args.wan:
                file_name += f"_{args.wandb_run.name}_{args.wandb_run.id}"
            os.makedirs(df_path, exist_ok=True)
            result_df.to_csv(os.path.join(df_path, f"{file_name}.csv"))

            if args.save_segmentation_images:
                for i, SimpleNet in enumerate(simplenet_list):
                    scores, segmentations, _, labels_gt, _ = SimpleNet.predict(dataloaders['testing'])
                    SimpleNet._save_segmentation_images(dataloaders['testing'], segmentations, scores, labels_gt, save_path=f'./output_{args.mainmodel}_{dataset_name}_{args.csv_save_name}')

    logger.info("\n\n-----\n")
    logger.info("\n\n-----\n")
    logger.info("\n\n-----\n")

    if args.wan and result_collect:
        result_df = pd.DataFrame(result_collect)
        each_dataset_df = result_df.groupby("dataset_name").mean().drop(columns='seed', errors='ignore')
        for dataset_name_it, row in each_dataset_df.iterrows():
            wandb.log({f"{dataset_name_it}_mean_metrics": row.to_dict()})

        overall_mean_metrics = result_df.drop(columns=['dataset_name', 'seed'], errors='ignore').mean(axis=0).to_dict()
        wandb.log({"overall_mean_metrics": overall_mean_metrics})
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
