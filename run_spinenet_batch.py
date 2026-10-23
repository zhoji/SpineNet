#!/usr/bin/env python3
"""
run_spinenet_batch.py

Usage:
  python rename_dicoms.py \
    --data_dir <PARENT DATA DIR> \
    --results_dir <RESULTS_DIR> (optional, default ./results) \
    --subjects <SUBJECT ID 1> <SUBJECT ID 2> (optional, defaults to all subfolders in data_dir) \

"""

import sys, time, glob
from datetime import datetime
import shutil
import argparse
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import pydicom
import tempfile
import uuid
from collections import defaultdict

import os
import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder

import pandas as pd

def collect_subject_list(args):
    """
    Resolve subjects list.

    Priority:
      1) args.subjects (explicit list)
      2) enumerate immediate subdirectories of data_dir
    """

    # Explicit list
    if args.subjects:
        return args.subjects

    # Otherwise enumerate data_dir
    data_dir = os.path.abspath(args.data_dir)

    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found or not a directory: {data_dir}")

    subj_candidates = [
        entry
        for entry in sorted(os.listdir(data_dir))
        if not entry.startswith('.') and
           os.path.isdir(os.path.join(data_dir, entry))
    ]

    if not subj_candidates:
        raise ValueError(f"No subject subdirectories found in: {data_dir}")

    return subj_candidates

def get_patient_id_from_files(file_list):
    for f in file_list:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            pid = getattr(ds, "PatientID", None)
            if pid:
                return str(pid)
        except:
            continue
    return None

def find_candidate_series(subject_dir):
    """
    Recursively search subject_dir for folders that likely contain Sagittal T2 series.
    Returns a list of candidate folder dicts sorted by descending file count:
      {'path': path, 'n_files': n, 'series_description': desc}
    """
    series_dict = defaultdict(lambda: {
        "files": [],
        "series_description": None
    })

    # --- group ALL dicoms by SeriesInstanceUID ---
    for root, _, files in os.walk(subject_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if not os.path.isfile(fpath):
                continue
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)

                uid = getattr(ds, "SeriesInstanceUID", None)
                if uid is None:
                    continue

                series_dict[uid]["files"].append(fpath)

                if series_dict[uid]["series_description"] is None:
                    desc = getattr(ds, "SeriesDescription", None) or getattr(ds, "ProtocolName", None)
                    if desc:
                        series_dict[uid]["series_description"] = str(desc)

            except:
                continue

    candidates = []

    # --- evaluate EACH series ---
    for uid, info in series_dict.items():
        files = info["files"]
        if len(files) < 5:
            continue  # too small to be real series

        n_match = 0
        n_total = 0

        for f in files[:20]:  # sample up to 20 slices
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except:
                continue

            n_total += 1

            # modality
            if getattr(ds, "Modality", "").upper() != "MR":
                continue

            # description (STRICT — no folder name)
            desc = (
                str(getattr(ds, "SeriesDescription", "") or "") + " " +
                str(getattr(ds, "ProtocolName", "") or "")
            ).lower()

            is_sag = "sag" in desc or "sagittal" in desc
            is_t2  = "t2" in desc

            # orientation
            is_sag_geom = False
            try:
                orientation = np.array(ds.ImageOrientationPatient).astype(float)
                is_sag_geom = abs(orientation[0]) > 0.7 and abs(orientation[3]) > 0.7
            except:
                pass

            if is_sag and is_t2:
                n_match += 1

        # require majority match (NOT just 1 slice)
        if n_total > 0 and n_match >= max(3, int(0.3 * n_total)):
            candidates.append({
                "uid": uid,
                "files": files,
                "series_description": info["series_description"],
                "n_files": len(files)
            })

    # sort by size (largest first)
    return sorted(candidates, key=lambda x: x["n_files"], reverse=True)

def _group_files_by_seriesuid(folder_path):
    """
    For all readable DICOM files in folder_path, group their full paths by SeriesInstanceUID.
    Returns dict: {seriesuid: [filepaths...], ...}
    """
    groups = defaultdict(lambda: {"files": [], "series_description": None})

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)

            uid = getattr(ds, 'SeriesInstanceUID', None) or getattr(ds, 'SeriesUID', None)
            if uid is None:
                uid = (getattr(ds, 'SeriesDescription', '') or '') + '_' + str(getattr(ds, 'SeriesNumber', ''))

            groups[uid]["files"].append(fpath)

            # store description once
            if groups[uid]["series_description"] is None:
                desc = getattr(ds, "SeriesDescription", None) or getattr(ds, "ProtocolName", None)
                if desc:
                    groups[uid]["series_description"] = str(desc)

        except:
            continue

    return groups

def load_scans_for_subject(subject_id, data_dir, require_extensions=False):
    """
    Locate all candidate Sagittal T2 series for subject_id under data_dir

    Returns:
        list of dicts: [{"scan": scan, "patient_id": patient_id}]
    """
    subject_dir = os.path.join(data_dir, subject_id)
    print(f"\n[1/3] Loading DICOM scans...\n      searching under {subject_dir}")

    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    candidates = find_candidate_series(subject_dir)
    if not candidates:
        print("No Sagittal T2 series found; returning empty list")
        return []

    print("Candidate Sagittal T2 series:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] '{c['series_description']}' (n_files={c['n_files']})")

    loaded = []
    for c in candidates:
        file_list = c["files"]
        series_desc = c["series_description"] or "UNKNOWN_SERIES"
        
        if len(file_list) < 5:
            print(f"  - Skipping '{series_desc}' (too few slices: {len(file_list)})")
            continue

        tmpdir = None

        try:
            # --- create temp dir ---
            tmpdir = tempfile.mkdtemp(prefix=f"spinenet_{subject_id}_")

            # --- symlink or copy files ---
            for src in file_list:
                dst = os.path.join(tmpdir, os.path.basename(src))
                try:
                    os.symlink(os.path.abspath(src), dst)
                except Exception:
                    shutil.copy2(src, dst)

            patient_id = get_patient_id_from_files(file_list) or subject_id

            print(f"\n  Loading series: '{series_desc}' ({len(file_list)} slices)")

            try:
                scan = load_dicoms_from_folder(tmpdir, require_extensions=require_extensions)
            except Exception as e:
                print(f"    - Skipping (load failed): {e}")
                continue

            if scan is None:
                print("    - Loader returned None; skipping")
                continue

            print(f"    ✓ Loaded scan")
            print(f"      shape: {scan.volume.shape}")
            print(f"      pixel spacing: {scan.pixel_spacing}")
            print(f"      slice thickness: {scan.slice_thickness}")
            loaded.append({"scan": scan, "patient_id": patient_id, "series_description": series_desc})
        except Exception as e:
            print(f"    - Error processing '{series_desc}': {e}")
        finally:
            if tmpdir and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Finished loading scans for subject {subject_id}")
    return loaded

def process_ivd_grades(ivd_grades, subject_id):
    """
    Convert SpineNet ivd_grades DataFrame into:
      - wide (all metrics)
      - long (all metrics)
      - wide Pfirrmann only
      - long Pfirrmann only
    """

    # normalize index → level column
    df = ivd_grades.copy()
    df = df.reset_index().rename(columns={"index": "level"})

    # ---------- LONG (all metrics) ----------
    long_full = df.copy()
    long_full.insert(0, "subject_id", subject_id)

    # ---------- WIDE (all metrics) ----------
    wide_full = {"subject_id": subject_id}

    for _, row in df.iterrows():
        level = row["level"]
        for col in df.columns:
            if col == "level":
                continue
            wide_full[f"{col}_{level}"] = row[col]

    # ---------- FOR METAGWAS ANALYSIS ----------
    # For lumbar Pfirrmann scores only
    LUMBAR_LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

    # ---------- LONG (Pfirrmann only) ----------
    #long_pf = df[["level", "Pfirrmann"]].copy()
    long_pf = df[df["level"].isin(LUMBAR_LEVELS)][["level", "Pfirrmann"]].copy()
    long_pf.insert(0, "subject_id", subject_id)

    # ---------- WIDE (Pfirrmann only) ----------
    wide_pf = {"subject_id": subject_id}
    for _, row in df.iterrows():
        if row["level"] in LUMBAR_LEVELS:
            wide_pf[row["level"]] = row["Pfirrmann"]

    return wide_full, long_full, wide_pf, long_pf

def find_repeat_inconsistencies(df_long_pf):
    """
    Identify subjects where Pfirrmann grades differ across repeats.

    Returns:
        DataFrame with one row per inconsistent (subject, level)
    """
    records = []

    grouped = df_long_pf.groupby(["subject_id", "level"])

    for (subj, level), g in grouped:
        unique_vals = [int(x) for x in g["Pfirrmann"].dropna().unique()]

        if len(unique_vals) > 1:
            records.append({
                "subject_id": subj,
                "level": level,
                "values": ",".join(map(str, unique_vals)),
                "n_repeats": len(g),
                "series_desc_list": "; ".join(map(str, g["series_desc"].unique()))
            })

    return pd.DataFrame(records)

def main():
    p = argparse.ArgumentParser(
        description="Run SpineNet in batch"
    )
    p.add_argument('--data_dir',  required=True,
                   help='main data folder containing subject folders')
    p.add_argument('--results_dir',  default='./results',
                   help='main data folder containing subject folders')
    p.add_argument('--subjects', nargs='+',
                   help='list of subject IDs to process (overrides --data_dir if provided)')
    args = p.parse_args()

    data_dir   = args.data_dir
    results_dir   = args.results_dir

    try:
        subj_list  = collect_subject_list(args)
    except Exception as e:
        print(f"Error resolving subjects: {e}")
        sys.exit(1)

    print("=" * 60)
    print("SpineNet Batch Run")
    print("=" * 60)

    # Create directories
    os.makedirs(results_dir, exist_ok=True)

    # Download weights
    print("\n[1/2] Downloading model weights...")
    spinenet.download_weights(verbose=True, force=False)
    print("✓ Weights downloaded")

    # Check device
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n[2/2] Initializing SpineNet on device: {device}")
    if device == 'cpu':
        print("   ⚠ Warning: Running on CPU. This will be slower than GPU.")

    # Initialize SpineNet
    spnt = SpineNet(device=device, verbose=True)

    wide_full_rows = []
    long_full_rows = []

    wide_pf_rows = []
    long_pf_rows = []

    repeat_counter = defaultdict(int)
    
    # Batch process each subject
    start_global = time.time()
    total = len(subj_list)

    for i, subject_id in enumerate(subj_list, 1):
        elapsed = time.time() - start_global
        avg_time = elapsed / i
        remaining = avg_time * (total - i)

        print(f"\n[{i}/{total}] Processing subject: {subject_id}")
        print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min")
        # Load scan
        scans = load_scans_for_subject(subject_id, data_dir)

        if not scans:
            print(f"No usable scans found for {subject_id}; skipping subject.")
            continue
        
        # Detect vertebrae
        for item in scans:
            scan = item["scan"]
            base_id = item["patient_id"]

            # Include handling for multiple Sag T2 scans
            repeat_idx = repeat_counter[base_id]
            repeat_counter[base_id] += 1

            subject_id_clean = base_id
            is_primary = (repeat_idx == 0)

            print("\n[2/3] Detecting vertebrae...")
            vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
            detected_labels = [v["predicted_label"] for v in vert_dicts]
            print(f"✓ Detected {len(vert_dicts)} vertebrae: {detected_labels}")

            # Grade IVDs
            print("\n[3/3] Grading intervertebral discs...")
            ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.volume)
            ivd_grades = spnt.grade_ivds(ivd_dicts)

            # Display results
            print("\n" + "=" * 60)
            print("GRADING RESULTS")
            print("=" * 60)
            print(ivd_grades)

            # Transform data for master csv files
            wide_full, long_full, wide_pf, long_pf = process_ivd_grades(ivd_grades, subject_id_clean)

            series_desc = item.get("series_description", None)

            long_full["series_desc"] = series_desc
            long_full["repeat_index"] = repeat_idx
            long_full["is_primary"] = is_primary

            long_pf["series_desc"] = series_desc
            long_pf["repeat_index"] = repeat_idx
            long_pf["is_primary"] = is_primary

            wide_pf["series_desc"] = series_desc
            wide_pf["repeat_index"] = repeat_idx
            wide_pf["is_primary"] = is_primary

            wide_full_rows.append(wide_full)
            long_full_rows.append(long_full)

            wide_pf_rows.append(wide_pf)
            long_pf_rows.append(long_pf)

    print("\n✓ Saved outputs:")
    # ---------- FULL SCORING ----------
    # Comment out this section if only interested in Pfirrmann grading
    df_wide_full = pd.DataFrame(wide_full_rows)
    df_long_full = pd.concat(long_full_rows, ignore_index=True)
    df_wide_full.to_csv(os.path.join(results_dir, "spinenet_wide.csv"), index=False)
    df_long_full.to_csv(os.path.join(results_dir, "spinenet_long.csv"), index=False)
    print("  - spinenet_wide.csv")
    print("  - spinenet_long.csv")

    # ---------- PFIRRMANN ONLY ----------
    df_wide_pf = pd.DataFrame(wide_pf_rows)
    df_long_pf = pd.concat(long_pf_rows, ignore_index=True)
    df_wide_pf.to_csv(os.path.join(results_dir, "spinenet_pfirrmann_wide_repeats.csv"), index=False)
    df_long_pf.to_csv(os.path.join(results_dir, "spinenet_pfirrmann_long_repeats.csv"), index=False)
    print("  - spinenet_pfirrmann_wide_repeats.csv")
    print("  - spinenet_pfirrmann_long_repeats.csv")
    # FINAL ANALYSIS CSV FOR METAGWAS (NO REPEAT IDS)
    df_wide_pf_final = df_wide_pf[df_wide_pf["is_primary"] == True]
    df_wide_pf_final.to_csv(os.path.join(results_dir, "spinenet_pfirrmann_wide.csv"), index=False)
    print("  - spinenet_pfirrmann_wide.csv")

    # ---------- OPTIONAL QC: repeat consistency ----------
    df_inconsist = find_repeat_inconsistencies(df_long_pf)

    qc_file = os.path.join(results_dir, "spinenet_repeat_inconsistencies.csv")

    if not df_inconsist.empty:
        df_inconsist.to_csv(qc_file, index=False)
        print(f"  - spinenet_repeat_inconsistencies.csv (inconsistencies found)")
    else:
        # still write empty file for record-keeping
        df_inconsist.to_csv(qc_file, index=False)
        print(f"  - spinenet_repeat_inconsistencies.csv (no inconsistencies)")

    print("\n" + "=" * 60)
    print(f"\nProcessed {total} subjects.")
    print("SpineNet batch run completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()