"""
Assembly indexer for Fusion 360 Gallery Dataset.

Builds a design index compatible with ai_assistant few-shot retrieval, but from
assembly.json files. Each indexed part includes an inferred build plane (XY/XZ/YZ)
to help orientation-aware reconstruction.

Usage:
  python index_assembly.py --input E:/Work/Fusion360/datasets/assembly --output assembly_design_index.json
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_а-яА-ЯёЁ]+", str(text or "").lower())


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        k = str(item).strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _obj_bbox(obj_path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(obj_path):
        return None
    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")
    has_vertex = False
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x = _to_float(parts[1], None)
            y = _to_float(parts[2], None)
            z = _to_float(parts[3], None)
            if x is None or y is None or z is None:
                continue
            has_vertex = True
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            z_min = min(z_min, z)
            z_max = max(z_max, z)
    if not has_vertex:
        return None
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
    }


def _bbox_corners(b: Dict[str, float]) -> List[Tuple[float, float, float]]:
    return [
        (b["x_min"], b["y_min"], b["z_min"]),
        (b["x_min"], b["y_min"], b["z_max"]),
        (b["x_min"], b["y_max"], b["z_min"]),
        (b["x_min"], b["y_max"], b["z_max"]),
        (b["x_max"], b["y_min"], b["z_min"]),
        (b["x_max"], b["y_min"], b["z_max"]),
        (b["x_max"], b["y_max"], b["z_min"]),
        (b["x_max"], b["y_max"], b["z_max"]),
    ]


def _vec3(node: Dict, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if not isinstance(node, dict):
        return default
    return (
        _to_float(node.get("x"), default[0]),
        _to_float(node.get("y"), default[1]),
        _to_float(node.get("z"), default[2]),
    )


def _transform_bbox(local_bbox: Dict[str, float], transform: Optional[Dict]) -> Dict[str, float]:
    if not transform:
        return dict(local_bbox)

    origin = _vec3(transform.get("origin"), (0.0, 0.0, 0.0))
    x_axis = _vec3(transform.get("x_axis"), (1.0, 0.0, 0.0))
    y_axis = _vec3(transform.get("y_axis"), (0.0, 1.0, 0.0))
    z_axis = _vec3(transform.get("z_axis"), (0.0, 0.0, 1.0))

    xs, ys, zs = [], [], []
    for lx, ly, lz in _bbox_corners(local_bbox):
        wx = origin[0] + lx * x_axis[0] + ly * y_axis[0] + lz * z_axis[0]
        wy = origin[1] + lx * x_axis[1] + ly * y_axis[1] + lz * z_axis[1]
        wz = origin[2] + lx * x_axis[2] + ly * y_axis[2] + lz * z_axis[2]
        xs.append(wx)
        ys.append(wy)
        zs.append(wz)
    return {
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "z_min": min(zs), "z_max": max(zs),
    }


def _infer_plane_from_bbox(b: Dict[str, float]) -> str:
    wx = abs(b["x_max"] - b["x_min"])
    wy = abs(b["y_max"] - b["y_min"])
    wz = abs(b["z_max"] - b["z_min"])
    smallest = min((("x", wx), ("y", wy), ("z", wz)), key=lambda kv: kv[1])[0]
    if smallest == "x":
        return "YZ"
    if smallest == "y":
        return "XZ"
    return "XY"


def _normalize_frame(parts: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not parts:
        return []
    x_min = min(p["x_min"] for p in parts)
    x_max = max(p["x_max"] for p in parts)
    y_min = min(p["y_min"] for p in parts)
    z_min = min(p["z_min"] for p in parts)
    x_center = 0.5 * (x_min + x_max)

    out = []
    for p in parts:
        q = dict(p)
        q["x_min"] -= x_center
        q["x_max"] -= x_center
        q["y_min"] -= y_min
        q["y_max"] -= y_min
        q["z_min"] -= z_min
        q["z_max"] -= z_min
        if q["z_min"] < 0:
            q["z_min"] = 0.0
        out.append(q)
    return out


def _joint_type_hist(data: Dict) -> Dict[str, int]:
    out = {}
    for bucket in ("joints", "as_built_joints"):
        all_items = data.get(bucket, {}) or {}
        for _, item in all_items.items():
            if not isinstance(item, dict):
                continue
            motion = item.get("joint_motion", {}) or {}
            jt = str(motion.get("joint_type", "Unknown")).strip()
            out[jt] = out.get(jt, 0) + 1
    return out


def _component_bodies_map(components: Dict) -> Dict[str, List[str]]:
    out = {}
    for comp_id, comp in (components or {}).items():
        body_ids = comp.get("bodies", []) or []
        out[comp_id] = [str(x) for x in body_ids]
    return out


def _assembly_instances(data: Dict) -> List[Dict]:
    instances = []
    root_bodies = (data.get("root", {}) or {}).get("bodies", {}) or {}
    for body_id in root_bodies.keys():
        instances.append({"body_id": body_id, "occ_name": "root", "transform": None})

    comp_bodies = _component_bodies_map(data.get("components", {}) or {})
    for occ_id, occ in (data.get("occurrences", {}) or {}).items():
        occ_name = str(occ.get("name", occ_id))
        transform = occ.get("transform", {}) or None
        occ_bodies_dict = occ.get("bodies", {}) or {}
        body_ids = list(occ_bodies_dict.keys()) if occ_bodies_dict else []
        if not body_ids:
            comp_id = str(occ.get("component", ""))
            body_ids = comp_bodies.get(comp_id, [])
        for body_id in body_ids:
            instances.append({"body_id": body_id, "occ_name": occ_name, "transform": transform})
    return instances


def _build_keywords(assembly_dir_name: str, props: Dict, joint_hist: Dict) -> List[str]:
    tokens = []
    tokens.extend(_tokenize(assembly_dir_name))
    tokens.extend(_tokenize(props.get("name", "")))
    for c in props.get("categories", []) or []:
        tokens.extend(_tokenize(c))
    for i in props.get("industries", []) or []:
        tokens.extend(_tokenize(i))
    for jt in joint_hist.keys():
        tokens.extend(_tokenize(jt.replace("JointType", "")))
    return _dedupe_keep_order(tokens)


def process_assembly_json(assembly_json_path: str, max_parts: int = 64) -> Optional[Dict]:
    with open(assembly_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assembly_dir = os.path.dirname(assembly_json_path)
    assembly_dir_name = os.path.basename(assembly_dir)
    bodies = data.get("bodies", {}) or {}
    joint_hist = _joint_type_hist(data)
    props = data.get("properties", {}) or {}

    obj_bbox_cache = {}
    for body_id, body in bodies.items():
        obj_name = str((body or {}).get("obj", "")).strip()
        if not obj_name:
            continue
        obj_path = os.path.join(assembly_dir, obj_name)
        bbox = _obj_bbox(obj_path)
        if bbox:
            obj_bbox_cache[body_id] = bbox

    parts = []
    for inst in _assembly_instances(data):
        body_id = inst["body_id"]
        local_bbox = obj_bbox_cache.get(body_id)
        if not local_bbox:
            continue
        world_bbox = _transform_bbox(local_bbox, inst.get("transform"))
        wx = abs(world_bbox["x_max"] - world_bbox["x_min"])
        wy = abs(world_bbox["y_max"] - world_bbox["y_min"])
        wz = abs(world_bbox["z_max"] - world_bbox["z_min"])
        if min(wx, wy, wz) < 0.02:
            continue
        body_name = str((bodies.get(body_id, {}) or {}).get("name", body_id)).strip() or body_id
        occ_name = str(inst.get("occ_name", "")).strip()
        part_name = f"{occ_name}/{body_name}" if occ_name and occ_name != "root" else body_name
        part = {
            "name": part_name,
            "x_min": world_bbox["x_min"],
            "x_max": world_bbox["x_max"],
            "y_min": world_bbox["y_min"],
            "y_max": world_bbox["y_max"],
            "z_min": world_bbox["z_min"],
            "z_max": world_bbox["z_max"],
            "plane": _infer_plane_from_bbox(world_bbox),
        }
        parts.append(part)
        if max_parts > 0 and len(parts) >= max_parts:
            break

    if not parts:
        return None

    parts = _normalize_frame(parts)
    parts.sort(key=lambda p: (p["z_min"], p["y_min"], p["x_min"], p["name"]))
    for p in parts:
        for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            p[k] = round(float(p[k]), 2)

    x_vals = [p["x_min"] for p in parts] + [p["x_max"] for p in parts]
    y_vals = [p["y_min"] for p in parts] + [p["y_max"] for p in parts]
    z_vals = [p["z_min"] for p in parts] + [p["z_max"] for p in parts]
    width = max(x_vals) - min(x_vals)
    depth = max(y_vals) - min(y_vals)
    height = max(z_vals) - min(z_vals)

    rel_source = os.path.join(os.path.basename(assembly_dir), "assembly.json")
    categories = props.get("categories", []) or []
    cat_text = ", ".join(str(c) for c in categories[:2]) if categories else "assembly"
    description = (
        f"{assembly_dir_name} ({len(parts)} parts, {width:.0f}x{depth:.0f}x{height:.0f} cm, {cat_text})"
    )
    keywords = _build_keywords(assembly_dir_name, props, joint_hist)

    complexity = {
        "part_count": len(parts),
        "num_joints": len(data.get("joints", {}) or {}),
        "num_as_built_joints": len(data.get("as_built_joints", {}) or {}),
        "num_contacts": len(data.get("contacts", []) or []),
        "num_holes": len(data.get("holes", []) or []),
        "joint_types": joint_hist,
    }

    return {
        "keywords": keywords,
        "description": description,
        "source_file": rel_source,
        "parts": parts,
        "complexity": complexity,
    }


def main():
    parser = argparse.ArgumentParser(description="Index assembly dataset into design_index-like JSON.")
    parser.add_argument("--input", required=True, help="Assembly root dir or single assembly.json file")
    parser.add_argument("--output", default="assembly_design_index.json", help="Output JSON file")
    parser.add_argument("--recursive", action="store_true", help="Recursive search for assembly.json under --input")
    parser.add_argument("--append", action="store_true", help="Append to existing output")
    parser.add_argument("--max-files", type=int, default=0, help="Process at most N assembly files (0 = all)")
    parser.add_argument("--max-parts", type=int, default=64, help="Max parts per assembly entry")
    parser.add_argument("--track-only", action="store_true", help="Keep only track/crawler-like assemblies")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        if args.recursive:
            files = sorted(glob.glob(os.path.join(args.input, "**", "assembly.json"), recursive=True))
        else:
            files = sorted(glob.glob(os.path.join(args.input, "*/assembly.json")))
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    if args.max_files > 0:
        files = files[:args.max_files]

    existing = []
    if args.append and os.path.isfile(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)

    out = list(existing)
    skipped = 0
    for idx, assembly_json in enumerate(files, 1):
        try:
            entry = process_assembly_json(assembly_json, max_parts=args.max_parts)
            if not entry:
                skipped += 1
                continue
            if args.track_only:
                hay = " ".join(entry.get("keywords", []))
                jt = entry.get("complexity", {}).get("joint_types", {})
                comp = entry.get("complexity", {}) or {}
                rev = int(jt.get("RevoluteJointType", 0))
                cyl = int(jt.get("CylindricalJointType", 0))
                contacts = int(comp.get("num_contacts", 0))
                holes = int(comp.get("num_holes", 0))
                has_track_kw = any(
                    k in hay for k in (
                        "track", "tank", "caterpillar", "crawler", "sprocket", "idler",
                        "гусениц", "гусеница", "трак", "танк", "vehicle", "chassis",
                    )
                )
                # Practical fallback: some assemblies are unlabeled but still have
                # dense revolute/cylindrical + contact patterns.
                mech_like = (rev + cyl) >= 1 and (contacts >= 8 or holes >= 8)
                strongly_mech = (rev + cyl) >= 4 and contacts >= 12
                if not (has_track_kw or mech_like or strongly_mech):
                    skipped += 1
                    continue
            out.append(entry)
        except Exception as e:
            skipped += 1
            print(f"[skip] {assembly_json}: {e}")
            continue

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(files)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} entries to {args.output} (skipped={skipped})")


if __name__ == "__main__":
    main()
