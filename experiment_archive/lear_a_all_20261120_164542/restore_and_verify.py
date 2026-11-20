import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"

with (ROOT / "manifest.csv").open(
    newline="", encoding="utf-8"
) as f:
    rows = list(csv.DictReader(f))

for row in rows:
    target = PROJECT / row["path"]

    if row["storage"] == "chunks" and not target.exists():
        parts_dir = target.with_name(target.name + ".upload_parts")
        parts = sorted(parts_dir.glob("*.part"))
        if not parts:
            raise SystemExit(f"분할 파일 누락: {target}")
        temporary = target.with_name(target.name + ".restoring")
        with temporary.open("wb") as dst:
            for part in parts:
                with part.open("rb") as src:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
        temporary.replace(target)

    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as src:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)

    if (
        size != int(row["bytes"])
        or digest.hexdigest() != row["sha256"]
    ):
        raise SystemExit(f"무결성 검사 실패: {target}")

print(f"복원 및 SHA-256 검사 완료: {len(rows):,}개")
