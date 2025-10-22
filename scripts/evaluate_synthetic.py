import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_expected(path: Path) -> dict[int, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def analyse_case(
    patient_id: int,
    expected_entries: list[dict],
    patient_df: pd.DataFrame,
) -> dict:
    expected_map = {entry["hpo_id"]: entry["phrase"] for entry in expected_entries}
    expected_ids: set[str] = set(expected_map.keys())

    predicted_phrase_map: dict[str, set[str]] = defaultdict(set)
    for row in patient_df.itertuples():
        hpo_id = str(row.hpo_id).strip()
        phrase = str(row.phenotype_name).strip()
        if not hpo_id or hpo_id.lower() == "nan":
            continue
        predicted_phrase_map[hpo_id].add(phrase)

    valid_pred_ids = {hid for hid in predicted_phrase_map if hid.lower() != "no candidate fit"}
    no_candidate_count = sum(1 for hid in predicted_phrase_map if hid.lower() == "no candidate fit")

    tp_ids = expected_ids & valid_pred_ids
    fp_ids = valid_pred_ids - expected_ids
    fn_ids = expected_ids - valid_pred_ids

    precision = len(tp_ids) / len(valid_pred_ids) if valid_pred_ids else 0.0
    recall = len(tp_ids) / len(expected_ids) if expected_ids else 0.0

    return {
        "patient_id": patient_id,
        "expected": expected_entries,
        "tp": sorted(
            [
                {
                    "hpo_id": hid,
                    "phrase": expected_map[hid],
                    "matched_phrases": sorted(predicted_phrase_map.get(hid, [])),
                }
                for hid in tp_ids
            ],
            key=lambda item: item["hpo_id"],
        ),
        "fp": sorted(
            [
                {
                    "hpo_id": hid,
                    "phrases": sorted(predicted_phrase_map[hid]),
                }
                for hid in fp_ids
            ],
            key=lambda item: item["hpo_id"],
        ),
        "fn": sorted(
            [
                {
                    "hpo_id": hid,
                    "phrase": expected_map[hid],
                }
                for hid in fn_ids
            ],
            key=lambda item: item["hpo_id"],
        ),
        "precision": precision,
        "recall": recall,
        "valid_pred_count": len(valid_pred_ids),
        "expected_count": len(expected_ids),
        "no_candidate_fit": no_candidate_count,
    }


def render_markdown(report_path: Path, overview: dict, case_details: list[dict]) -> None:
    summary_df = pd.DataFrame(
        [
            {"Metric": "Precision", "Value": f"{overview['precision']:.2f}"},
            {"Metric": "Recall", "Value": f"{overview['recall']:.2f}"},
            {
                "Metric": "F1",
                "Value": f"{overview['f1']:.2f}" if overview["f1"] is not None else "n/a",
            },
            {"Metric": "Total expected HPO", "Value": overview["total_expected"]},
            {"Metric": "Total predicted HPO", "Value": overview["total_predicted"]},
            {
                "Metric": "No Candidate Fit count",
                "Value": overview["no_candidate_fit_total"],
            },
        ]
    )

    lines: list[str] = []
    lines.append("# Synthetic Chinese Pipeline Evaluation")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Case Details")
    lines.append("")

    for detail in case_details:
        pid = detail["patient_id"]
        lines.append(f"### Case {pid}")
        lines.append("")
        lines.append(f"- Precision: {detail['precision']:.2f} ({len(detail['tp'])} TP, {len(detail['fp'])} FP)")
        lines.append(f"- Recall: {detail['recall']:.2f} ({len(detail['tp'])} / {detail['expected_count']})")
        lines.append(f"- No Candidate Fit: {detail['no_candidate_fit']}")

        if detail["tp"]:
            lines.append("- True Positives:")
            for item in detail["tp"]:
                matched = ", ".join(item["matched_phrases"]) if item["matched_phrases"] else "n/a"
                lines.append(f"  - {item['hpo_id']} · {item['phrase']} (matched phrases: {matched})")
        else:
            lines.append("- True Positives: none")

        if detail["fp"]:
            lines.append("- False Positives:")
            for item in detail["fp"]:
                phrase_list = ", ".join(item["phrases"])
                lines.append(f"  - {item['hpo_id']} · predicted phrases: {phrase_list}")
        else:
            lines.append("- False Positives: none")

        if detail["fn"]:
            lines.append("- False Negatives:")
            for item in detail["fn"]:
                lines.append(f"  - {item['hpo_id']} · expected phrase: {item['phrase']}")
        else:
            lines.append("- False Negatives: none")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic Chinese RAG-HPO predictions.")
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=Path("examples/synthetic/outputs/predictions.csv"),
        help="Path to prediction CSV produced by rag-hpo process.",
    )
    parser.add_argument(
        "--expected-json",
        type=Path,
        default=Path("examples/synthetic/inputs/expected.json"),
        help="Path to JSON file containing expected HPO mappings per patient.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/synthetic_eval_report.md"),
        help="Where to write the markdown report.",
    )
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    pred_df = pred_df.rename(
        columns={
            "Patient ID": "patient_id",
            "Category": "category",
            "Phenotype name": "phenotype_name",
            "HPO ID": "hpo_id",
        }
    )
    pred_df["patient_id"] = pred_df["patient_id"].astype(int)

    expected_map = load_expected(args.expected_json)
    case_details = []

    overall_tp = overall_fp = overall_fn = 0
    overall_pred = overall_expected = 0
    overall_no_fit = pred_df["hpo_id"].str.contains("No Candidate Fit", case=False, na=False).sum()

    for patient_id, expected_entries in expected_map.items():
        patient_records = pred_df[pred_df["patient_id"] == patient_id].copy()
        detail = analyse_case(patient_id, expected_entries, patient_records)
        case_details.append(detail)

        overall_tp += len(detail["tp"])
        overall_fp += len(detail["fp"])
        overall_fn += len(detail["fn"])
        overall_pred += detail["valid_pred_count"]
        overall_expected += detail["expected_count"]

    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0.0
    recall = overall_tp / overall_expected if overall_expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else None

    overview = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_expected": overall_expected,
        "total_predicted": overall_pred,
        "no_candidate_fit_total": overall_no_fit,
    }

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    render_markdown(args.report_md, overview, case_details)

    print("Overall precision:", f"{precision:.2f}")
    print("Overall recall:", f"{recall:.2f}")
    print("Overall F1:", f"{f1:.2f}" if f1 is not None else "n/a")
    print("Report saved to:", args.report_md)


if __name__ == "__main__":
    main()
