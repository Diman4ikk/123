import json
from collections import Counter, defaultdict
from pathlib import Path


DATA_DIR = Path("outputs") / "fusion_dataset"


def main() -> None:
    records = []

    for split in ("train", "val"):
        metadata_path = DATA_DIR / split / "metadata.json"

        with metadata_path.open(encoding="utf-8") as file:
            records.extend(json.load(file))

    # Количество объектов каждого класса в каждой сцене.
    scene_counts = defaultdict(Counter)

    for record in records:
        scene = record["scene"]
        label = record["label"]
        scene_counts[scene][label] += 1

    all_classes = sorted({
        record["label"]
        for record in records
    })

    print("OBJECT COUNTS BY SCENE")
    print("=" * 80)

    for scene in sorted(scene_counts):
        counts = scene_counts[scene]
        total = sum(counts.values())

        print(f"\n{scene}: {total} objects")

        for label in all_classes:
            count = counts.get(label, 0)
            if count:
                print(f"  {label:<22} {count:>4}")

    # Сколько разных сцен содержат каждый класс.
    scenes_by_class = defaultdict(set)

    for scene, counts in scene_counts.items():
        for label, count in counts.items():
            if count > 0:
                scenes_by_class[label].add(scene)

    print("\nCLASS COVERAGE")
    print("=" * 80)

    for label in all_classes:
        scenes = sorted(scenes_by_class[label])

        print(
            f"{label:<22} "
            f"scenes={len(scenes):>2}: "
            f"{', '.join(scenes)}"
        )


if __name__ == "__main__":
    main()