from pathlib import Path
from typing import Union

import pandas as pd


def load_finding_labels(
    pathological_image_csv: Union[str, Path],
    pathology_csv: Union[str, Path],
) -> pd.DataFrame:
    """
    Open TG-GATEsの画像テーブルと病理所見テーブルを突合し、スライドID(.svsファイル名)
    ごとに病理所見の有無(has_finding)をまとめたテーブルを返す。

    open_tggates_pathological_image.csv の FILE_LOCATION 末尾
    "{slide_id}.svs" からslide_idを取り出し、(EXP_ID, GROUP_ID, INDIVIDUAL_ID, ORGAN)
    をキーに open_tggates_pathology.csv（実際の病理所見）と突合する。一致する所見行が
    1件も無いスライドは「所見なし(正常)」(has_finding=False)として扱う
    （TG-GATEsの病理テーブルは異常所見のみを記録する形式のため）。

    Returns
    -------
    pd.DataFrame
        slide_idをindexとし、以下の列を持つ:
        has_finding, n_findings, compound_name, organ, dose, sacrifice_period
    """
    image_df = pd.read_csv(pathological_image_csv, encoding="cp932")
    image_df["slide_id"] = image_df["FILE_LOCATION"].str.extract(r"/(\d+)\.svs$", expand=False)

    pathology_df = pd.read_csv(pathology_csv, encoding="cp932")

    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "ORGAN"]
    finding_counts = pathology_df.groupby(key).size().rename("n_findings")

    merged = image_df.join(finding_counts, on=key)
    merged["n_findings"] = merged["n_findings"].fillna(0).astype(int)
    merged["has_finding"] = merged["n_findings"] > 0

    merged = merged.set_index("slide_id")
    return merged[
        ["has_finding", "n_findings", "COMPOUND_NAME", "ORGAN", "DOSE", "SACRIFICE_PERIOD"]
    ].rename(
        columns={
            "COMPOUND_NAME": "compound_name",
            "ORGAN": "organ",
            "DOSE": "dose",
            "SACRIFICE_PERIOD": "sacrifice_period",
        }
    )
