#!/usr/bin/env python3
"""
Comprehensive Data Quality Audit for Sprint 4
Checks for duplicates, contamination, balance, and data quality issues.
"""

import json
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Union
import sys

class DataQualityAuditor:
    def __init__(self):
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }

    def load_json(self, filepath: str) -> Union[dict, list]:
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.issues['critical'].append(f"Failed to load {filepath}: {e}")
            return None

    def md5_hash(self, text: str) -> str:
        """Generate MD5 hash for text"""
        return hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()

    def check_duplicates_within_dataset(self, data: list, name: str) -> Dict:
        """Check for duplicate questions within a dataset"""
        print(f"\n{'='*60}")
        print(f"Checking duplicates within {name}")
        print(f"{'='*60}")

        question_hashes = []
        hash_to_indices = defaultdict(list)

        for idx, item in enumerate(data):
            question = item.get('question', '')
            q_hash = self.md5_hash(question)
            question_hashes.append(q_hash)
            hash_to_indices[q_hash].append(idx)

        # Find duplicates
        duplicates = {h: indices for h, indices in hash_to_indices.items() if len(indices) > 1}

        print(f"Total samples: {len(data)}")
        print(f"Unique questions: {len(hash_to_indices)}")
        print(f"Duplicate questions: {len(duplicates)}")

        if duplicates:
            self.issues['critical'].append(
                f"{name}: Found {len(duplicates)} duplicate questions affecting {sum(len(v) for v in duplicates.values())} samples"
            )
            # Show first few examples
            for i, (hash_val, indices) in enumerate(list(duplicates.items())[:3]):
                print(f"\nDuplicate {i+1}: Appears at indices {indices}")
                print(f"Question: {data[indices[0]]['question'][:100]}...")
        else:
            print("✓ No duplicates found")

        return {
            'total': len(data),
            'unique': len(hash_to_indices),
            'duplicates': len(duplicates),
            'duplicate_samples': sum(len(v) for v in duplicates.values())
        }

    def check_cross_contamination(self, data1: list, name1: str,
                                  data2: list, name2: str) -> int:
        """Check for overlapping questions between two datasets"""
        print(f"\n{'='*60}")
        print(f"Checking contamination: {name1} vs {name2}")
        print(f"{'='*60}")

        hashes1 = {self.md5_hash(item.get('question', '')) for item in data1}
        hashes2 = {self.md5_hash(item.get('question', '')) for item in data2}

        overlap = hashes1 & hashes2

        print(f"{name1} unique questions: {len(hashes1)}")
        print(f"{name2} unique questions: {len(hashes2)}")
        print(f"Overlapping questions: {len(overlap)}")

        if overlap:
            self.issues['critical'].append(
                f"CONTAMINATION: {len(overlap)} questions overlap between {name1} and {name2}"
            )
            # Show examples
            for i, hash_val in enumerate(list(overlap)[:3]):
                for item in data1:
                    if self.md5_hash(item.get('question', '')) == hash_val:
                        print(f"\nOverlap {i+1}: {item['question'][:100]}...")
                        break
        else:
            print("✓ No contamination found")

        return len(overlap)

    def check_probe_balance(self, probe_data: dict) -> Dict:
        """Check if probe dataset has 50/50 balance"""
        print(f"\n{'='*60}")
        print(f"Checking probe dataset balance")
        print(f"{'='*60}")

        if 'samples' not in probe_data:
            self.issues['critical'].append("Probe dataset missing 'samples' field")
            return {}

        samples = probe_data['samples']
        honest_count = sum(1 for s in samples if s.get('is_honest', True))
        deceptive_count = len(samples) - honest_count

        print(f"Total samples: {len(samples)}")
        print(f"Honest samples: {honest_count}")
        print(f"Deceptive samples: {deceptive_count}")
        print(f"Expected: 392 each (50/50)")

        # Check metadata
        meta_honest = probe_data.get('n_honest', 0)
        meta_deceptive = probe_data.get('n_deceptive', 0)
        print(f"\nMetadata claims: {meta_honest} honest, {meta_deceptive} deceptive")

        if honest_count != deceptive_count:
            self.issues['warning'].append(
                f"Probe dataset imbalanced: {honest_count} honest vs {deceptive_count} deceptive"
            )

        if honest_count != 392 or deceptive_count != 392:
            self.issues['warning'].append(
                f"Probe dataset size unexpected: Got {honest_count}/{deceptive_count}, expected 392/392"
            )

        if meta_honest != honest_count or meta_deceptive != deceptive_count:
            self.issues['critical'].append(
                f"Metadata mismatch: Claims {meta_honest}/{meta_deceptive} but actual is {honest_count}/{deceptive_count}"
            )
        else:
            print("✓ Metadata matches actual counts")

        return {
            'total': len(samples),
            'honest': honest_count,
            'deceptive': deceptive_count,
            'balance': honest_count / len(samples) if len(samples) > 0 else 0
        }

    def check_training_labels(self, train_data: list, val_data: list) -> Dict:
        """Check if training data is 100% honest (no 'is_honest' field or deceptive labels)"""
        print(f"\n{'='*60}")
        print(f"Checking training data labels (should be 100% honest)")
        print(f"{'='*60}")

        train_with_label = sum(1 for item in train_data if 'is_honest' in item)
        val_with_label = sum(1 for item in val_data if 'is_honest' in item)

        print(f"Train samples with 'is_honest' field: {train_with_label}/{len(train_data)}")
        print(f"Val samples with 'is_honest' field: {val_with_label}/{len(val_data)}")

        if train_with_label > 0:
            self.issues['warning'].append(
                f"Training data has {train_with_label} samples with 'is_honest' field (should be omitted)"
            )
            # Check if any are deceptive
            train_deceptive = sum(1 for item in train_data if not item.get('is_honest', True))
            if train_deceptive > 0:
                self.issues['critical'].append(
                    f"Training data contains {train_deceptive} DECEPTIVE samples (should be 100% honest)"
                )
        else:
            print("✓ Training data has no 'is_honest' labels (as expected)")

        if val_with_label > 0:
            self.issues['warning'].append(
                f"Validation data has {val_with_label} samples with 'is_honest' field"
            )

        return {
            'train_labeled': train_with_label,
            'val_labeled': val_with_label
        }

    def check_data_quality(self, data: list, name: str) -> Dict:
        """Check for empty, malformed, or suspicious samples"""
        print(f"\n{'='*60}")
        print(f"Checking data quality for {name}")
        print(f"{'='*60}")

        empty_questions = 0
        empty_answers = 0
        missing_fields = 0
        suspicious_short = []
        suspicious_long = []
        encoding_issues = 0

        question_lengths = []
        answer_lengths = []

        for idx, item in enumerate(data):
            # Check required fields
            if 'question' not in item or 'answer' not in item:
                missing_fields += 1
                continue

            question = item.get('question', '')
            answer = item.get('answer', '')

            # Check empty
            if not question or not question.strip():
                empty_questions += 1
            if not answer or not answer.strip():
                empty_answers += 1

            # Track lengths
            q_len = len(question)
            a_len = len(answer)
            question_lengths.append(q_len)
            answer_lengths.append(a_len)

            # Check suspicious lengths
            if q_len < 20:
                suspicious_short.append((idx, q_len, question[:50]))
            if q_len > 1000:
                suspicious_long.append((idx, q_len, question[:50]))

            # Check encoding issues
            if '\x00' in question or '\x00' in answer:
                encoding_issues += 1

        print(f"Total samples: {len(data)}")
        print(f"Missing required fields: {missing_fields}")
        print(f"Empty questions: {empty_questions}")
        print(f"Empty answers: {empty_answers}")
        print(f"Encoding issues (null bytes): {encoding_issues}")

        if question_lengths:
            print(f"\nQuestion length stats:")
            print(f"  Min: {min(question_lengths)}")
            print(f"  Max: {max(question_lengths)}")
            print(f"  Avg: {sum(question_lengths)/len(question_lengths):.1f}")

        if answer_lengths:
            print(f"\nAnswer length stats:")
            print(f"  Min: {min(answer_lengths)}")
            print(f"  Max: {max(answer_lengths)}")
            print(f"  Avg: {sum(answer_lengths)/len(answer_lengths):.1f}")

        print(f"\nSuspiciously short questions (<20 chars): {len(suspicious_short)}")
        if suspicious_short:
            for idx, length, text in suspicious_short[:3]:
                print(f"  Index {idx} ({length} chars): {text}")

        print(f"Suspiciously long questions (>1000 chars): {len(suspicious_long)}")
        if suspicious_long:
            for idx, length, text in suspicious_long[:3]:
                print(f"  Index {idx} ({length} chars): {text}...")

        # Add issues
        if missing_fields > 0:
            self.issues['critical'].append(f"{name}: {missing_fields} samples missing required fields")
        if empty_questions > 0:
            self.issues['critical'].append(f"{name}: {empty_questions} samples with empty questions")
        if empty_answers > 0:
            self.issues['critical'].append(f"{name}: {empty_answers} samples with empty answers")
        if encoding_issues > 0:
            self.issues['critical'].append(f"{name}: {encoding_issues} samples with encoding issues")
        if len(suspicious_short) > 10:
            self.issues['warning'].append(f"{name}: {len(suspicious_short)} suspiciously short questions")

        return {
            'total': len(data),
            'missing_fields': missing_fields,
            'empty_questions': empty_questions,
            'empty_answers': empty_answers,
            'encoding_issues': encoding_issues,
            'avg_question_len': sum(question_lengths)/len(question_lengths) if question_lengths else 0,
            'avg_answer_len': sum(answer_lengths)/len(answer_lengths) if answer_lengths else 0
        }

    def show_samples(self, data: list, name: str, n: int = 3):
        """Show sample examples from dataset"""
        print(f"\n{'='*60}")
        print(f"Sample examples from {name}")
        print(f"{'='*60}")

        for i in range(min(n, len(data))):
            item = data[i]
            print(f"\nSample {i+1}:")
            print(f"Question: {item.get('question', 'N/A')[:150]}...")
            print(f"Answer: {item.get('answer', 'N/A')[:150]}...")
            if 'is_honest' in item:
                print(f"Label: {'HONEST' if item['is_honest'] else 'DECEPTIVE'}")
            print(f"Meta: {item.get('meta', {})}")

    def generate_report(self) -> str:
        """Generate final GO/NO-GO report"""
        critical_count = len(self.issues['critical'])
        warning_count = len(self.issues['warning'])

        print(f"\n{'='*60}")
        print(f"FINAL AUDIT REPORT")
        print(f"{'='*60}")

        if critical_count > 0:
            status = "❌ NO-GO"
            recommendation = "CRITICAL ISSUES FOUND - Must fix before training"
        elif warning_count > 3:
            status = "⚠️  CAUTION"
            recommendation = "Multiple warnings found - Review carefully before proceeding"
        elif warning_count > 0:
            status = "⚠️  CAUTION"
            recommendation = "Minor issues found - Safe to proceed but monitor"
        else:
            status = "✅ GO"
            recommendation = "Data is clean - Safe to proceed with training"

        report = f"""
{status}

RECOMMENDATION: {recommendation}

SUMMARY:
- Critical Issues: {critical_count}
- Warnings: {warning_count}
- Info: {len(self.issues['info'])}

"""

        if self.issues['critical']:
            report += "\nCRITICAL ISSUES:\n"
            for issue in self.issues['critical']:
                report += f"  ❌ {issue}\n"

        if self.issues['warning']:
            report += "\nWARNINGS:\n"
            for issue in self.issues['warning']:
                report += f"  ⚠️  {issue}\n"

        if self.issues['info']:
            report += "\nINFORMATIONAL:\n"
            for issue in self.issues['info']:
                report += f"  ℹ️  {issue}\n"

        print(report)
        return report

def main():
    auditor = DataQualityAuditor()

    # File paths
    base_path = "/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/data/processed"
    train_path = f"{base_path}/train.json"
    val_path = f"{base_path}/val.json"
    probe_path = f"{base_path}/probe_dataset_gpt2_clean.json"

    print("="*60)
    print("SPRINT 4 DATA QUALITY AUDIT")
    print("="*60)
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Probe: {probe_path}")

    # Load data
    train_data = auditor.load_json(train_path)
    val_data = auditor.load_json(val_path)
    probe_data = auditor.load_json(probe_path)

    if not all([train_data, val_data, probe_data]):
        print("\n❌ CRITICAL: Failed to load one or more datasets")
        return 1

    # Extract samples from probe dataset
    probe_samples = probe_data.get('samples', []) if isinstance(probe_data, dict) else probe_data

    # 1. Check duplicates within each dataset
    train_dup_stats = auditor.check_duplicates_within_dataset(train_data, "Training Set")
    val_dup_stats = auditor.check_duplicates_within_dataset(val_data, "Validation Set")
    probe_dup_stats = auditor.check_duplicates_within_dataset(probe_samples, "Probe Set")

    # 2. Check cross-contamination
    train_val_overlap = auditor.check_cross_contamination(train_data, "Train", val_data, "Val")
    train_probe_overlap = auditor.check_cross_contamination(train_data, "Train", probe_samples, "Probe")
    val_probe_overlap = auditor.check_cross_contamination(val_data, "Val", probe_samples, "Probe")

    # 3. Check probe dataset balance
    if isinstance(probe_data, dict):
        probe_balance = auditor.check_probe_balance(probe_data)

    # 4. Check training labels
    label_stats = auditor.check_training_labels(train_data, val_data)

    # 5. Check data quality
    train_quality = auditor.check_data_quality(train_data, "Training Set")
    val_quality = auditor.check_data_quality(val_data, "Validation Set")
    probe_quality = auditor.check_data_quality(probe_samples, "Probe Set")

    # 6. Show samples
    auditor.show_samples(train_data, "Training Set", n=3)
    auditor.show_samples(val_data, "Validation Set", n=2)
    auditor.show_samples(probe_samples, "Probe Set", n=3)

    # Generate final report
    report = auditor.generate_report()

    # Save report to file
    report_path = f"{base_path}/audit_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return 0 if len(auditor.issues['critical']) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
