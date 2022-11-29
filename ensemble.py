import argparse

import numpy as np
import pandas as pd


class Ensemble:
    def __init__(self, filenames: str, filepath: str):
        self.filenames = filenames
        self.output_list = []

        output_path = [filepath + filename + ".csv" for filename in filenames]
        self.output_frame = pd.read_csv(output_path[0]).drop("prediction", axis=1)
        self.output_df = self.output_frame.copy()

        for path in output_path:
            self.output_list.append(pd.read_csv(path)["prediction"].to_list())
        for filename, output in zip(filenames, self.output_list):
            self.output_df[filename] = output

    # Simple Weighted
    # 직접 weight를 지정하여, 앙상블합니다.
    def simple_weighted(self, weight: list):
        if not len(self.output_list) == len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight) != 1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.output_list[0]], [self.output_list[1]], axis=0)
        for i in range(2, len(self.output_list)):
            pred_arr = np.append(pred_arr, [self.output_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()

    # Average Weighted
    # (1/n)의 가중치로 앙상블을 진행합니다.
    def average_weighted(self):
        weight = [1 / len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [
            pred * np.array(w) for pred, w in zip(self.output_list, weight)
        ]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()

    # Mixed
    # Negative case 발생 시, 다음 순서에서 예측한 rating으로 넘어가서 앙상블합니다.
    def mixed(self):
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames) - 1):
            pre_idx = self.filenames[idx]
            post_idx = self.filenames[idx + 1]
            result[self.output_df[pre_idx] < 1] = self.output_df.loc[
                self.output_df[pre_idx] < 1, post_idx
            ]
        return result.tolist()


def main(args):
    file_list = sum(args.ENSEMBLE_FILES, [])

    if len(file_list) < 2:
        raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")

    en = Ensemble(filenames=file_list, filepath=args.RESULT_PATH)

    if args.ENSEMBLE_STRATEGY == "WEIGHTED":
        if args.ENSEMBLE_WEIGHT:
            strategy_title = "sw-" + "-".join(
                map(str, *args.ENSEMBLE_WEIGHT)
            )  # simple weighted
            result = en.simple_weighted(*args.ENSEMBLE_WEIGHT)
        else:
            strategy_title = "aw"  # average weighted
            result = en.average_weighted()
    elif args.ENSEMBLE_STRATEGY == "MIXED":
        strategy_title = args.ENSEMBLE_STRATEGY.lower()  # mixed
        result = en.mixed()
    else:
        pass
    en.output_frame["prediction"] = result
    output = en.output_frame.copy()
    files_title = file_list[0].split("_")[0] + "_ensemble"

    output.to_csv(f"{args.RESULT_PATH}{files_title}-{strategy_title}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    """
    [실행 방법]
    ```
    python ensemble.py [인자]
    ```
    
    [인자 설명]
    > 스크립트 실행 시, 
    > 인자가 필수인 경우 required
    > 필수가 아닌 경우, optional 로 명시하였습니다.
    
    --ENSEMBLE_FILES ENSEMBLE_FILES [ENSEMBLE_FILES ...]
    required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 
    이 때, 경로(submit)와 확장자(.csv)는 입력하지 않습니다.
    --ENSEMBLE_STRATEGY {WEIGHTED,MIXED}
    optional: 앙상블 전략을 선택해 주세요.
    [MIXED, WEIGHTED] 중 선택 가능합니다.
    (default="WEIGHTED")
    --ENSEMBLE_WEIGHT ENSEMBLE_WEIGHT [ENSEMBLE_WEIGHT ...]
    optional: Weighted 앙상블 전략에서만 사용되는 인자입니다.
    전달받은 결과값의 가중치를 조정할 수 있습니다.
    가중치를 쉼표(,)로 구분하여 모두 입력해 주세요.
    이 때, 합산 1이 되지 않는 경우 작동하지 않습니다.
    --RESULT_PATH RESULT_PATH
    optional: 앙상블할 파일이 존재하는 경로를 전달합니다. 
    기본적으로 베이스라인의 결과가 떨어지는 공간인 submit으로 연결됩니다.
    앙상블된 최종 결과물도 해당 경로 안에 떨어집니다.
    (default:"./submit/")
    [결과물]
    RESULT_PATH 안에 앙상블된 최종 결과물이 저장됩니다.
    {files_title}-{strategy_title}.csv
    파일명은 ENSEMBLE_FILES들과 ENSEMBLE_STRATEGY가 모두 명시되어 있습니다.
    ENSEMBLE_STRATEGY의 경우, 아래와 같이 작성됩니다.
    > simple weighted : sw + 각 파일에 적용된 가중치
    > average weighted : aw
    > mixed : mixed
    """

    arg(
        "--ENSEMBLE_FILES",
        nargs="+",
        required=True,
        type=lambda s: [item for item in s.split(",")],
        help="required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.",
    )
    arg(
        "--ENSEMBLE_STRATEGY",
        type=str,
        default="WEIGHTED",
        choices=["WEIGHTED", "MIXED"],
        help='optional: [MIXED, WEIGHTED] 중 앙상블 전략을 선택해 주세요. (default="WEIGHTED")',
    )
    arg(
        "--ENSEMBLE_WEIGHT",
        nargs="+",
        default=None,
        type=lambda s: [float(item) for item in s.split(",")],
        help="optional: Weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.",
    )
    arg(
        "--RESULT_PATH",
        type=str,
        default="../output/",
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다. (default:"../output/")',
    )
    args = parser.parse_args()
    main(args)
