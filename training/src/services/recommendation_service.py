from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
import polars as pl

from training.src.io.loaders import load_parquet


MODEL_ID = "diana_model_v1"
DEFAULT_LOCAL_PREDICTIONS_PATH = "/tmp/predictions.parquet"


@dataclass(frozen=True)
class RecommendationItem:
    anchor_id: str
    kiosk_id: str
    product_id: str
    model_id: str
    recommendation_date: str


@dataclass(frozen=True)
class MultiRecommendationItem:
    anchor_id: str
    kiosk_id: str
    recs: list[str]
    model_id: str
    recommendation_date: str


class RecommendationService:
    def __init__(self, lookup: dict[tuple[str, str], list[str]], model_id: str = MODEL_ID):
        self.lookup = lookup
        self.model_id = model_id

    @classmethod
    def from_parquet(cls, predictions_path: str | Path, model_id: str = MODEL_ID) -> "RecommendationService":
        df = load_parquet(Path(predictions_path), label="Predictions parquet")

        required_columns = {
            "kiosk_id",
            "anchor_product_id",
            "candidate_product_id",
            "score",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Predictions parquet is missing columns: {sorted(missing_columns)}")

        df = df.select(
            [
                pl.col("anchor_product_id").cast(pl.Utf8).alias("anchor_id"),
                pl.col("kiosk_id").cast(pl.Utf8).alias("kiosk_id"),
                pl.col("candidate_product_id").cast(pl.Utf8).alias("product_id"),
                pl.col("score").cast(pl.Float64).alias("score"),
            ]
        )

        df = df.sort(["anchor_id", "kiosk_id", "score"], descending=[False, False, True])

        lookup: dict[tuple[str, str], list[str]] = {}
        for (anchor_id, kiosk_id), group_df in df.group_by(["anchor_id", "kiosk_id"], maintain_order=True):
            lookup[(anchor_id, kiosk_id)] = group_df["product_id"].to_list()

        return cls(lookup=lookup, model_id=model_id)

    @classmethod
    def from_s3(
        cls,
        *,
        bucket: str,
        key: str,
        local_path: str | Path = DEFAULT_LOCAL_PREDICTIONS_PATH,
        model_id: str = MODEL_ID,
    ) -> "RecommendationService":
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(local_path))

        return cls.from_parquet(predictions_path=local_path, model_id=model_id)

    def get_recommendations(
        self,
        *,
        anchor_id: str,
        kiosk_id: str,
        limit: int = 20,
    ) -> list[RecommendationItem]:
        recs = self.lookup.get((anchor_id, kiosk_id), [])
        recs = recs[: max(0, limit)]
        recommendation_date = self._utc_now()

        return [
            RecommendationItem(
                anchor_id=anchor_id,
                kiosk_id=kiosk_id,
                product_id=product_id,
                model_id=self.model_id,
                recommendation_date=recommendation_date,
            )
            for product_id in recs
        ]

    def get_multi_recommendations(
        self,
        requests: list[dict[str, str]],
    ) -> list[MultiRecommendationItem]:
        recommendation_date = self._utc_now()

        response: list[MultiRecommendationItem] = []
        for item in requests:
            anchor_id = item["anchor_id"]
            kiosk_id = item["kiosk_id"]

            response.append(
                MultiRecommendationItem(
                    anchor_id=anchor_id,
                    kiosk_id=kiosk_id,
                    recs=self.lookup.get((anchor_id, kiosk_id), []),
                    model_id=self.model_id,
                    recommendation_date=recommendation_date,
                )
            )

        return response

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")