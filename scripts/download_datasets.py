"""Download Kaggle datasets or generate schema-accurate synthetic demo data for local/CI testing."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

KAGGLE_DATASETS: dict[str, tuple[str, str]] = {
    "rees46": (
        "mkechinov/ecommerce-behavior-data-from-multi-category-store",
        "rees46",
    ),
    "hillstrom": (
        "saurabhshahane/hillstrom-email-marketing",
        "hillstrom",
    ),
    "criteo": (
        "arashnic/criteo-uplift-prediction",
        "criteo_uplift",
    ),
    "tiki": (
        "enseyh/tiki-ecommerce-fashion-products",
        "tiki_shopee_catalog",
    ),
}


def download_from_kaggle(dataset_key: str, raw_root: Path) -> None:
    """Download and unzip a dataset using the official Kaggle CLI."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI ('kaggle') not found in PATH. Install via `pip install kaggle` "
            "and configure KAGGLE_USERNAME / KAGGLE_KEY, or run with `--synthetic-demo`."
        )

    slug, subdir = KAGGLE_DATASETS[dataset_key]
    dest_dir = raw_root / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir), "--unzip"]
    print(f"[Kaggle] Downloading {slug} -> {dest_dir} ...")
    subprocess.run(cmd, check=True)


def generate_synthetic_demo_data(raw_root: Path, seed: int = 42, n_users: int = 250) -> dict[str, Path]:
    """Generate realistic, schema-accurate raw CSVs matching all 4 Kaggle datasets."""
    rng = random.Random(seed)
    outputs: dict[str, Path] = {}

    # 1. REES46 eCommerce Behavior Data (9 exact columns)
    rees_dir = raw_root / "rees46"
    rees_dir.mkdir(parents=True, exist_ok=True)
    rees_file = rees_dir / "2019-Oct-sample.csv"

    categories = [
        (2053013555631882655, "electronics.smartphone", ["samsung", "apple", "xiaomi", "oppo"]),
        (2053013563810775923, "appliances.kitchen.washer", ["lg", "bosch", "electrolux"]),
        (2053013558920217191, "computers.notebook", ["asus", "lenovo", "acer", "hp"]),
        (2053013553853497655, "", ["creed", "chanel", ""]),  # Nullable category_code & brand
    ]
    base_time = datetime(2019, 10, 1, 8, 0, 0, tzinfo=timezone.utc)

    with rees_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "event_time",
            "event_type",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
            "user_id",
            "user_session",
        ])

        for u_idx in range(n_users):
            user_id = 510000000 + u_idx
            n_sessions = rng.randint(1, 5)
            user_cursor = base_time + timedelta(
                days=rng.randint(0, 22), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
            )
            for s_idx in range(n_sessions):
                session_id = f"00000000-0000-4000-8000-{user_id:09d}{s_idx:03d}"
                cat_id, cat_code, brands = rng.choice(categories)
                brand = rng.choice(brands)
                product_id = rng.randint(1000000, 9999999)
                price = round(rng.uniform(19.99, 1299.99), 2)

                # Views in session
                views_n = rng.randint(2, 6)
                t = user_cursor
                for _ in range(views_n):
                    writer.writerow([
                        t.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "view",
                        product_id,
                        cat_id,
                        cat_code,
                        brand,
                        f"{price:.2f}",
                        user_id,
                        session_id,
                    ])
                    t += timedelta(seconds=rng.randint(15, 180))

                # 65% of sessions add to cart
                if rng.random() < 0.65:
                    writer.writerow([
                        t.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "cart",
                        product_id,
                        cat_id,
                        cat_code,
                        brand,
                        f"{price:.2f}",
                        user_id,
                        session_id,
                    ])
                    t += timedelta(seconds=rng.randint(20, 120))

                    # 15% remove item, 22% purchase immediately, ~25% return within 7 days to buy
                    roll = rng.random()
                    if roll < 0.15:
                        writer.writerow([
                            t.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "remove_from_cart",
                            product_id,
                            cat_id,
                            cat_code,
                            brand,
                            f"{price:.2f}",
                            user_id,
                            session_id,
                        ])
                    elif roll < 0.37:
                        writer.writerow([
                            t.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "purchase",
                            product_id,
                            cat_id,
                            cat_code,
                            brand,
                            f"{price:.2f}",
                            user_id,
                            session_id,
                        ])
                    elif roll < 0.62:
                        # Abandoned this session, but returns in 1-5 days and purchases!
                        followup_time = t + timedelta(hours=rng.randint(6, 110))
                        followup_session = f"11111111-0000-4000-8000-{user_id:09d}{s_idx:03d}"
                        writer.writerow([
                            followup_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "view",
                            product_id,
                            cat_id,
                            cat_code,
                            brand,
                            f"{price:.2f}",
                            user_id,
                            followup_session,
                        ])
                        writer.writerow([
                            (followup_time + timedelta(seconds=45)).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "purchase",
                            product_id,
                            cat_id,
                            cat_code,
                            brand,
                            f"{price:.2f}",
                            user_id,
                            followup_session,
                        ])

                user_cursor += timedelta(days=rng.randint(2, 12), hours=rng.randint(1, 10))

    outputs["rees46"] = rees_file

    # 2. Hillstrom Email Marketing (12 exact columns)
    hill_dir = raw_root / "hillstrom"
    hill_dir.mkdir(parents=True, exist_ok=True)
    hill_file = hill_dir / "MineThatData_E-MailAnalytics_DataMiningChallenge.csv"

    segments = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
    zips = ["Urban", "Surburban", "Rural"]
    channels = ["Web", "Phone", "Multichannel"]

    with hill_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "recency",
            "history_segment",
            "history",
            "mens",
            "womens",
            "zip_code",
            "newbie",
            "channel",
            "segment",
            "visit",
            "conversion",
            "spend",
        ])
        for _ in range(600):
            recency = rng.randint(1, 12)
            history = round(rng.uniform(29.99, 850.0), 2)
            hist_seg = "1) $0 - $100" if history < 100 else ("2) $100 - $200" if history < 200 else "3) $200+")
            mens = rng.choice([0, 1])
            womens = 1 if mens == 0 else rng.choice([0, 1])
            zip_code = rng.choice(zips)
            newbie = rng.choice([0, 1])
            channel = rng.choice(channels)
            segment = rng.choice(segments)

            base_visit_p = 0.10 + (0.06 if segment != "No E-Mail" else 0.0) + (0.04 if recency <= 4 else 0.0)
            visit = 1 if rng.random() < base_visit_p else 0
            conv_p = (0.08 + (0.05 if segment != "No E-Mail" else 0.0)) if visit else 0.0
            conversion = 1 if rng.random() < conv_p else 0
            spend = round(rng.uniform(35.0, 250.0), 2) if conversion else 0.0

            writer.writerow([
                recency, hist_seg, history, mens, womens, zip_code, newbie, channel, segment, visit, conversion, spend
            ])

    outputs["hillstrom"] = hill_file

    # 3. Criteo Uplift v2 (16 exact columns)
    criteo_dir = raw_root / "criteo_uplift"
    criteo_dir.mkdir(parents=True, exist_ok=True)
    criteo_file = criteo_dir / "criteo-uplift-v2.1-sample.csv"

    with criteo_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([*(f"f{i}" for i in range(12)), "treatment", "conversion", "visit", "exposure"])
        for _ in range(300):
            feats = [round(rng.gauss(0.0, 1.0), 5) for _ in range(12)]
            treatment = 1 if rng.random() < 0.85 else 0
            exposure = 1 if (treatment == 1 and rng.random() < 0.15) else 0
            visit = 1 if rng.random() < (0.05 + 0.03 * treatment) else 0
            conversion = 1 if (visit and rng.random() < (0.06 + 0.04 * exposure)) else 0
            writer.writerow([*feats, treatment, conversion, visit, exposure])

    outputs["criteo"] = criteo_file

    # 4. Tiki Vietnamese Fashion Catalog (17 exact columns)
    tiki_dir = raw_root / "tiki_shopee_catalog"
    tiki_dir.mkdir(parents=True, exist_ok=True)
    tiki_file = tiki_dir / "tiki_fashion_sample.csv"

    vn_products = [
        (101001, "Áo Thun Nam Cotton Co Giãn 4 Chiều Coolmate", 199000, 159000, "Coolmate"),
        (101002, "Đầm Nữ Dáng Xòe Công Sở Cao Cấp Gumac", 495000, 389000, "GUMAC"),
        (101003, "Giày Sneaker Nam Nữ Biti's Hunter X", 999000, 849000, "Biti's"),
        (101004, "Balo Laptop Chống Nước 15.6 Inch Simplecarry", 650000, 520000, "Simplecarry"),
        (101005, "Áo Khoác Chống Nắng UV Uniqlo Airism", 686000, 599000, "Uniqlo"),
        (101006, "Túi Đeo Chéo Nữ Da Mềm Thời Trang Juno", 550000, 429000, "Juno"),
        (101007, "Quần Jeans Nam Slimfit Routine", 590000, 472000, "Routine"),
        (101008, "Vali Kéo Du Lịch Size 20 Nhựa ABS Mia", 1290000, 990000, "MIA"),
    ]

    with tiki_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "id", "name", "description", "original_price", "price", "fulfillment_type",
            "brand", "review_counts", "rating_average", "favorite_count", "pay_later",
            "current_seller", "date_created", "number_of_images", "vnd_cashback",
            "has_video", "quantity_sold",
        ])
        for pid, name, orig_p, p, brand in vn_products:
            writer.writerow([
                pid,
                name,
                f"{name} chính hãng, chất liệu thoáng mát, đổi trả 30 ngày.",
                orig_p,
                p,
                "tiki_delivery",
                brand,
                rng.randint(25, 1200),
                round(rng.uniform(4.4, 4.95), 2),
                rng.randint(10, 500),
                True,
                f"{brand} Official Store",
                rng.randint(30, 600),
                rng.randint(4, 9),
                0,
                True,
                rng.randint(100, 5000),
            ])

    outputs["tiki"] = tiki_file
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw datasets for Cart Abandonment AI")
    parser.add_argument(
        "--dataset",
        choices=["all", "rees46", "hillstrom", "criteo", "tiki"],
        default="all",
        help="Which Kaggle dataset to download",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target raw data directory",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Generate schema-accurate synthetic sample CSVs instead of calling Kaggle API",
    )
    args = parser.parse_args()

    if args.synthetic_demo:
        generated = generate_synthetic_demo_data(args.raw_dir)
        print("Generated synthetic demo datasets:")
        for key, file_path in generated.items():
            print(f"  - {key}: {file_path}")
        return

    targets = list(KAGGLE_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in targets:
        download_from_kaggle(key, args.raw_dir)


if __name__ == "__main__":
    main()
