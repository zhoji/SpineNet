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

def find_candidate_series(subject_dir, sample_per_folder=8):
    """
    Recursively search subject_dir for folders that likely contain Sagittal T2 series.
    Returns a list of candidate folder dicts sorted by descending file count:
      {'path': path, 'n_files': n, 'series_description': desc}
    """
    sag_t2_tokens = [
        'sag', 'sagittal',
        't2', 't2w', 't2-weighted', 't2_tse', 't2_tse_fs', 'sag_t2', 'sagittal_t2'
    ]
    candidates = []
    for root, dirs, files in os.walk(subject_dir):
        if not files:
            continue

        # collect candidate file paths (prefer .dcm but include others for header testing)
        file_paths = [os.path.join(root, f) for f in files]
        # sample a few files to inspect headers
        n_matched = 0
        n_examined = 0
        series_desc = None
        for fpath in file_paths[:sample_per_folder]:
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue
            n_examined += 1
            modality = getattr(ds, 'Modality', '') or ''
            if modality and modality.upper() != 'MR':
                continue
            # combine descriptive fields and folder name
            desc_fields = []
            for key in ('SeriesDescription', 'ProtocolName', 'SequenceName'):
                val = getattr(ds, key, '')
                if val:
                    desc_fields.append(str(val).lower())
            desc_fields.append(os.path.basename(root).lower())
            joined = ' '.join(desc_fields)
            if any(tok in joined for tok in sag_t2_tokens):
                n_matched += 1
                if not series_desc:
                    series_desc = getattr(ds, 'SeriesDescription', '') or getattr(ds, 'ProtocolName', '') or ''
        # quick count of readable DICOM files in folder
        total_dicom_files = 0
        for fpath in file_paths:
            try:
                _ = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                total_dicom_files += 1
            except Exception:
                pass

        # treat as candidate if any samples matched or folder name includes 't2'
        folder_name = os.path.basename(root).lower()
        if n_matched > 0 or 't2' in folder_name:
            candidates.append({
                'path': root,
                'n_matched_samples': n_matched,
                'n_examined_samples': n_examined,
                'n_files': total_dicom_files,
                'series_description': series_desc or folder_name
            })
    # fallback: if nothing found, include any folder named "DICOM" or top-level with any dicoms
    if not candidates:
        for root, dirs, files in os.walk(subject_dir):
            if 'dicom' in os.path.basename(root).lower():
                total = 0
                for f in files[:50]:
                    try:
                        _ = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True, force=True)
                        total += 1
                    except Exception:
                        pass
                if total > 0:
                    candidates.append({
                        'path': root,
                        'n_matched_samples': 0,
                        'n_examined_samples': 0,
                        'n_files': total,
                        'series_description': os.path.basename(root)
                    })
        # last resort: treat subject root if it contains dicoms
        if not candidates:
            total = 0
            for f in os.listdir(subject_dir)[:200]:
                fp = os.path.join(subject_dir, f)
                if os.path.isfile(fp):
                    try:
                        _ = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                        total += 1
                    except Exception:
                        pass
            if total > 0:
                candidates.append({
                    'path': subject_dir,
                    'n_matched_samples': 0,
                    'n_examined_samples': 0,
                    'n_files': total,
                    'series_description': 'subject_root'
                })

    # --- de-duplicate parent candidates when a child candidate contains DICOM files ---
    # If a candidate has any descendant candidate (child folder) with n_files > 0,
    # drop the parent candidate. This preserves the deepest folder that actually
    # contains DICOM files (e.g., keep .../DICOM, drop .../2_Sag_T2_FAT_SAT).
    deduped = []
    # ensure candidates are absolute and normalized
    for c in candidates:
        c['path'] = os.path.normpath(os.path.abspath(c['path']))

    for c in candidates:
        parent_path = c['path']
        # check whether there exists another candidate that is a strict descendant
        has_descendant_with_files = False
        for other in candidates:
            if other['path'] == parent_path:
                continue
            # other is a descendant if it starts with parent_path + os.sep
            if other['path'].startswith(parent_path + os.sep) and other.get('n_files', 0) > 0:
                has_descendant_with_files = True
                break
        if not has_descendant_with_files:
            deduped.append(c)

    # sort deduped by file count (or whatever ordering you prefer)
    candidates = sorted(deduped, key=lambda x: x['n_files'], reverse=True)
    # --- end dedupe ---

    return sorted(candidates, key=lambda x: x['n_files'], reverse=True)

def _group_files_by_seriesuid(folder_path):
    """
    For all readable DICOM files in folder_path, group their full paths by SeriesInstanceUID.
    Returns dict: {seriesuid: [filepaths...], ...}
    """
    groups = defaultdict(list)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            uid = getattr(ds, 'SeriesInstanceUID', None) or getattr(ds, 'SeriesUID', None)
            if uid is None:
                # use fallback grouping token (e.g., SeriesDescription+InstanceNumber)
                uid = (getattr(ds, 'SeriesDescription', '') or '') + '_' + str(getattr(ds, 'SeriesNumber', ''))
            groups[uid].append(fpath)
        except Exception:
            # skip unreadable files
            continue
    return groups

def load_scans_for_subject(subject_id, data_dir, require_extensions=True):
    """
    Locate all candidate Sagittal T2 series for subject_id under data_dir, load each series,
    and return list of (scan, scan_name) tuples.

    - subject_id is expected to map to data_dir/subject_id
    - Each found folder may yield multiple series (grouped by SeriesInstanceUID)
    """
    start_time = time.time()
    subject_dir = os.path.join(data_dir, subject_id)
    print(f"\nProcessing subject {subject_id}")
    print(f"\n[1/3] Loading DICOM scans...\n      searching under {subject_dir}")

    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    candidates = find_candidate_series(subject_dir, sample_per_folder=8)
    if not candidates:
        print("No candidate Sagittal-T2 folders found for subject; returning empty list")
        return []

    print("Candidate Sagittal-T2 folders (sorted by file count):")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c['path']} (n_files={c['n_files']}, desc='{c['series_description']}')")

    loaded = []
    for c in candidates:
        folder = c['path']
        # group files by SeriesInstanceUID to avoid mixed-series loading
        groups = _group_files_by_seriesuid(folder)
        if not groups:
            print(f"  - No readable DICOM files in {folder}; skipping")
            continue

        # sort groups by descending number of files (likely the main series first)
        groups_items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        for uid, file_list in groups_items:
            # create a temporary directory and symlink (or copy) the series files into it
            tmpdir = None
            try:
                tmpdir = tempfile.mkdtemp(prefix=f"spinenet_{subject_id}_{uuid.uuid4().hex[:6]}_")
                for src in file_list:
                    basename = os.path.basename(src)
                    dst = os.path.join(tmpdir, basename)
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except Exception:
                        # fallback to copy if symlink not allowed (Windows or permission)
                        shutil.copy2(src, dst)

                scan_name = f"{subject_id}_{os.path.basename(folder)}_uid_{uid[:8]}"
                print(f"\n  Loading series uid={uid} (n_files={len(file_list)}) from {folder}")
                scan = load_dicoms_from_folder(tmpdir, require_extensions=require_extensions)
                if scan is None:
                    print(f"    - loader returned None for {scan_name}; skipping")
                else:
                    print(f"    ✓ Loaded scan '{scan_name}'")
                    print(f"      shape: {getattr(scan, 'volume', None).shape if scan else 'N/A'}")
                    print(f"      pixel spacing: {getattr(scan, 'pixel_spacing', None) if scan else 'N/A'}")
                    print(f"      slice thickness: {getattr(scan, 'slice_thickness', None) if scan else 'N/A'}")
                    loaded.append((scan, scan_name))
            except Exception as e:
                print(f"    - Error loading series uid={uid} from {folder}: {e}")
            finally:
                if tmpdir and os.path.exists(tmpdir):
                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass

    end_time = time.time()
    print(f"Finished locating/loading scans for subject {subject_id} in {end_time - start_time:.2f}s")
    return loaded

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
    
    # Batch process each subject
    for subject_id in subj_list:
        # Load scan
        scans = load_scans_for_subject(subject_id, data_dir)

        if not scans:
            print(f"No usable scans found for {subject_id}; skipping subject.")
            continue
        
        # Detect vertebrae
        for scan, scan_name in scans:
            print("\n[2/3] Detecting vertebrae...")
            start_time = time.time()
            vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
            detected_labels = [v["predicted_label"] for v in vert_dicts]
            end_time = time.time()
            print(f"✓ Detected {len(vert_dicts)} vertebrae: {detected_labels}")
            print(f"  Elapsed time: {end_time - start_time:.2f}s")

            # Grade IVDs
            print("\n[3/3] Grading intervertebral discs...")
            start_time = time.time()
            ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.volume)
            ivd_grades = spnt.grade_ivds(ivd_dicts)

            # Display results
            print("\n" + "=" * 60)
            print("GRADING RESULTS")
            print("=" * 60)
            print(ivd_grades)

            # Save results
            output_file = f'{results_dir}/{scan_name}_test_results.csv'
            ivd_grades.to_csv(output_file)
            end_time = time.time()
            print(f"\n✓ Results saved to: {output_file}")
            print(f"  Elapsed time: {end_time - start_time:.2f}s")

    print("\n" + "=" * 60)
    print("SpineNet batch run completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()