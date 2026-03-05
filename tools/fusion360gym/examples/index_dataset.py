"""
Dataset Indexer — converts Fusion 360 Reconstruction Dataset JSONs into
a design_index.json for few-shot retrieval.

For each reconstruction JSON this script:
  1. Extracts sketch profiles and extrude operations
  2. Computes 3D bounding boxes for each extruded body
  3. Optionally generates text descriptions via LLM
  4. Saves entries into design_index.json

Usage:
  # Process a single file
  python index_dataset.py --input ../testdata/Couch.json

  # Process entire reconstruction dataset folder
  python index_dataset.py --input /path/to/reconstruction/jsons/ --output design_index.json

  # Huge dataset: filter, sort by complexity, keep top 2000, max 12 parts per design
  python index_dataset.py --input /path/to/jsons/ --recursive --output design_index.json \\
    --min-extrudes 2 --min-score 0.5 --max-part-count 12 --sort-by score --top-k 2000

  # Generate descriptions via LLM (requires API key in .env)
  python index_dataset.py --input ./data/ --describe
"""

import argparse
import json
import os
import sys
import glob

from dotenv import load_dotenv
load_dotenv()


def extract_sketch_plane(sketch_entity):
    """Determine which plane (XY, XZ, YZ) a sketch is on."""
    ref_plane = sketch_entity.get("reference_plane", {})
    plane_type = ref_plane.get("type", "")

    if plane_type == "ConstructionPlane":
        name = ref_plane.get("name", "")
        if "XY" in name:
            return "XY"
        if "XZ" in name:
            return "XZ"
        if "YZ" in name:
            return "YZ"

    transform = sketch_entity.get("transform", {})
    z_axis = transform.get("z_axis", {})
    zx, zy, zz = z_axis.get("x", 0), z_axis.get("y", 0), z_axis.get("z", 0)

    if abs(zz) > 0.9:
        return "XY"
    if abs(zy) > 0.9:
        return "XZ"
    if abs(zx) > 0.9:
        return "YZ"

    return "XY"


def extract_profile_bbox(profile_data):
    """Get bounding box of a profile from its loops' curve points."""
    xs, ys = [], []

    for loop in profile_data.get("loops", []):
        for curve in loop.get("profile_curves", []):
            for key in ("start_point", "end_point"):
                pt = curve.get(key, {})
                if "x" in pt and "y" in pt:
                    xs.append(pt["x"])
                    ys.append(pt["y"])

    if not xs or not ys:
        return None

    return {
        "sketch_x_min": min(xs),
        "sketch_x_max": max(xs),
        "sketch_y_min": min(ys),
        "sketch_y_max": max(ys),
    }


def sketch_to_world_bbox(plane, sketch_bbox, distance):
    """Convert sketch-space bounding box + extrude distance to world-space 3D bbox."""
    sx0 = sketch_bbox["sketch_x_min"]
    sx1 = sketch_bbox["sketch_x_max"]
    sy0 = sketch_bbox["sketch_y_min"]
    sy1 = sketch_bbox["sketch_y_max"]

    if plane == "XY":
        x_min, x_max = sx0, sx1
        y_min, y_max = sy0, sy1
        if distance >= 0:
            z_min, z_max = 0, distance
        else:
            z_min, z_max = distance, 0
    elif plane == "XZ":
        x_min, x_max = sx0, sx1
        z_min, z_max = sy0, sy1
        if distance >= 0:
            y_min, y_max = 0, distance
        else:
            y_min, y_max = distance, 0
    elif plane == "YZ":
        y_min, y_max = sx0, sx1
        z_min, z_max = sy0, sy1
        if distance >= 0:
            x_min, x_max = 0, distance
        else:
            x_min, x_max = distance, 0
    else:
        return None

    return {
        "x_min": round(x_min, 2),
        "x_max": round(x_max, 2),
        "y_min": round(y_min, 2),
        "y_max": round(y_max, 2),
        "z_min": round(z_min, 2),
        "z_max": round(z_max, 2),
    }


def compute_complexity_metrics(data, parts):
    """Compute complexity metrics to rank models for indexing/demo."""
    entities = data.get("entities", {}) if isinstance(data, dict) else {}
    timeline_count = len(data.get("timeline", [])) if isinstance(data, dict) else 0
    sequence_count = len(data.get("sequence", [])) if isinstance(data, dict) else 0
    sketch_count = 0
    extrude_count = 0
    operations = set()

    for entity in entities.values():
        if not isinstance(entity, dict):
            continue
        etype = entity.get("type", "")
        if etype == "Sketch":
            sketch_count += 1
        elif etype == "ExtrudeFeature":
            extrude_count += 1
            operations.add(entity.get("operation", ""))

    props = data.get("properties", {}) if isinstance(data, dict) else {}
    face_count = int(props.get("face_count", 0) or 0)
    body_count = int(props.get("body_count", 0) or 0)

    op_bonus = 0.0
    if "CutFeatureOperation" in operations:
        op_bonus += 2.0
    if "JoinFeatureOperation" in operations:
        op_bonus += 2.0
    if "IntersectFeatureOperation" in operations:
        op_bonus += 1.0

    complexity_score = (
        1.0 * sequence_count +
        0.5 * timeline_count +
        2.0 * extrude_count +
        0.10 * face_count +
        3.0 * body_count +
        0.5 * len(parts) +
        op_bonus
    )

    return {
        "sequence_count": sequence_count,
        "timeline_count": timeline_count,
        "sketch_count": sketch_count,
        "extrude_count": extrude_count,
        "face_count": face_count,
        "body_count": body_count,
        "part_count": len(parts),
        "operations": sorted([op for op in operations if op]),
        "complexity_score": round(complexity_score, 2),
    }


def process_reconstruction_json(filepath):
    """Extract bounding boxes and complexity metrics from a reconstruction JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", {})
    timeline = data.get("timeline", [])

    sketches = {}
    for eid, entity in entities.items():
        if entity.get("type") == "Sketch":
            sketches[eid] = entity

    parts = []
    part_index = 0

    for tl_entry in timeline:
        eid = tl_entry.get("entity", "")
        entity = entities.get(eid)
        if not entity:
            continue
        if entity.get("type") != "ExtrudeFeature":
            continue

        profiles = entity.get("profiles", [])
        if not profiles:
            continue

        sketch_id = profiles[0].get("sketch", "")
        profile_id = profiles[0].get("profile", "")
        sketch = sketches.get(sketch_id)
        if not sketch:
            continue

        plane = extract_sketch_plane(sketch)

        sketch_profiles = sketch.get("profiles", {})
        profile_data = sketch_profiles.get(profile_id)
        if not profile_data:
            continue

        sketch_bbox = extract_profile_bbox(profile_data)
        if not sketch_bbox:
            continue

        extent = entity.get("extent_one", {})
        dist_obj = extent.get("distance", {})
        distance = dist_obj.get("value", 1.0)

        operation = entity.get("operation", "NewBodyFeatureOperation")

        world_bbox = sketch_to_world_bbox(plane, sketch_bbox, distance)
        if not world_bbox:
            continue

        part_index += 1
        world_bbox["name"] = f"Body {part_index} ({operation})"
        parts.append(world_bbox)

    metrics = compute_complexity_metrics(data, parts)
    return parts, metrics


def generate_description_from_parts(parts, filename):
    """Simple heuristic description from part geometry."""
    if not parts:
        return filename

    all_x = [p["x_min"] for p in parts] + [p["x_max"] for p in parts]
    all_y = [p["y_min"] for p in parts] + [p["y_max"] for p in parts]
    all_z = [p["z_min"] for p in parts] + [p["z_max"] for p in parts]

    width = max(all_x) - min(all_x)
    depth = max(all_y) - min(all_y)
    height = max(all_z) - min(all_z)

    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.split("_")[0]

    return f"{name} ({len(parts)} parts, {width:.0f}x{depth:.0f}x{height:.0f} cm)"


def generate_keywords(filename, parts):
    """Extract keywords from the filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.split("_")[0].lower()
    words = []
    current = []
    for ch in name:
        if ch.isupper() and current:
            words.append("".join(current).lower())
            current = [ch]
        else:
            current.append(ch)
    if current:
        words.append("".join(current).lower())
    return words if words else [name]


def main():
    parser = argparse.ArgumentParser(description="Index reconstruction dataset for few-shot retrieval")
    parser.add_argument("--input", required=True, help="JSON file or directory of JSON files")
    parser.add_argument("--output", default="design_index_dataset.json", help="Output index file")
    parser.add_argument("--append", action="store_true", help="Append to existing index")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory for JSON files")
    parser.add_argument("--min-seq", type=int, default=0, help="Minimum sequence length")
    parser.add_argument("--min-extrudes", type=int, default=0, help="Minimum extrude count")
    parser.add_argument("--min-faces", type=int, default=0, help="Minimum face count")
    parser.add_argument("--min-bodies", type=int, default=0, help="Minimum body count")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum complexity score")
    parser.add_argument("--require-ops", type=str, default="", help="Comma-separated required operations (e.g. CutFeatureOperation,JoinFeatureOperation)")
    parser.add_argument("--sort-by", choices=["score", "seq", "extrudes", "faces"], default="score", help="Sort key for ranking")
    parser.add_argument("--top-k", type=int, default=0, help="Keep only top-k designs after sort (0 = all). Use e.g. 500–2000 for huge datasets.")
    parser.add_argument("--max-part-count", type=int, default=0, help="Skip designs with more than this many parts (0 = no limit). e.g. 12 to match pipeline cap.")
    parser.add_argument("--list-only", action="store_true", help="Only print matching candidates; do not write output file")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        if args.recursive:
            files = sorted(glob.glob(os.path.join(args.input, "**", "*.json"), recursive=True))
        else:
            files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)

    existing = []
    if args.append and os.path.isfile(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)

    index = existing[:]
    processed = 0
    skipped = 0

    required_ops = set(
        op.strip() for op in args.require_ops.split(",")
        if op.strip()
    )

    for filepath in files:
        try:
            parts, metrics = process_reconstruction_json(filepath)
            if not parts:
                skipped += 1
                continue

            if metrics["sequence_count"] < args.min_seq:
                skipped += 1
                continue
            if metrics["extrude_count"] < args.min_extrudes:
                skipped += 1
                continue
            if metrics["face_count"] < args.min_faces:
                skipped += 1
                continue
            if metrics["body_count"] < args.min_bodies:
                skipped += 1
                continue
            if args.max_part_count > 0 and len(parts) > args.max_part_count:
                skipped += 1
                continue
            if metrics["complexity_score"] < args.min_score:
                skipped += 1
                continue
            if required_ops and not required_ops.issubset(set(metrics["operations"])):
                skipped += 1
                continue

            entry = {
                "keywords": generate_keywords(filepath, parts),
                "description": generate_description_from_parts(parts, filepath),
                "source_file": os.path.basename(filepath),
                "parts": parts,
                "complexity": metrics,
            }
            index.append(entry)
            processed += 1
            print(
                f"  [{processed}] {entry['description']} | "
                f"score={metrics['complexity_score']:.2f} "
                f"seq={metrics['sequence_count']} "
                f"ext={metrics['extrude_count']} "
                f"faces={metrics['face_count']} "
                f"ops={','.join(metrics['operations']) or '-'}"
            )

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            skipped += 1

    key_map = {
        "score": lambda e: e.get("complexity", {}).get("complexity_score", 0),
        "seq": lambda e: e.get("complexity", {}).get("sequence_count", 0),
        "extrudes": lambda e: e.get("complexity", {}).get("extrude_count", 0),
        "faces": lambda e: e.get("complexity", {}).get("face_count", 0),
    }
    index.sort(key=key_map[args.sort_by], reverse=True)

    if args.top_k > 0:
        index = index[:args.top_k]

    print("\nTop candidates:")
    for i, e in enumerate(index[:20], 1):
        m = e.get("complexity", {})
        print(
            f"  {i:2d}. {e.get('source_file', '?'):35s} "
            f"score={m.get('complexity_score', 0):6.2f} "
            f"seq={m.get('sequence_count', 0):3d} "
            f"ext={m.get('extrude_count', 0):2d} "
            f"faces={m.get('face_count', 0):4d} "
            f"ops={','.join(m.get('operations', [])) or '-'}"
        )

    if not args.list_only:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"\nOutput: {args.output}")

    print(f"\nDone: {processed} designs indexed, {skipped} skipped.")


if __name__ == "__main__":
    main()
