"""
Minimal smoke test - just verify the fixes work with tiny data.
"""

import sys
from pathlib import Path

# Reuse the main testing class
sys.path.insert(0, str(Path(__file__).parent))
from run_q400_model_zoo_tests import ModelTester


def main():
    """Ultra-fast smoke test with minimal samples."""
    print("="*70)
    print("Q400 SMOKE TEST - Minimal validation")
    print("="*70)

    tester = ModelTester(
        data_config_name="q400",
        transform_config_name="minimal",
        task_config_name="one_step",
        batch_size=32,
        num_workers=0,  # No multiprocessing for simplicity
        max_samples=50,  # Only 50 samples!
        results_dir="results/q400_smoke_test"
    )

    # Setup data
    tester.setup_data()

    # Test one baseline (non-trainable)
    print("\n" + "#"*70)
    print("# 1. Baseline: Persistence (no training)")
    print("#"*70)
    result1 = tester.test_model("persistence", is_trainable=False)
    tester.results.append(result1)

    # Test one trainable with just 1 epoch
    print("\n" + "#"*70)
    print("# 2. Trainable: Linear AR (1 epoch)")
    print("#"*70)
    result2 = tester.test_model("linear_ar", is_trainable=True, train_epochs=1)
    tester.results.append(result2)

    # Save results
    tester.save_results()

    # Summary
    print("\n" + "="*70)
    print("SMOKE TEST COMPLETE")
    print("="*70)

    for r in tester.results:
        status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        print(f"\n{status_icon} {r['model_name']}")
        if r["status"] == "success":
            for metric, value in r["metrics"].items():
                print(f"  {metric.upper()}: {value:.4f}")
            print(f"  Runtime: {r['runtime_seconds']:.1f}s")
        else:
            print(f"  Error: {r['error'][:200]}...")  # Truncate long errors

    success_count = sum(1 for r in tester.results if r["status"] == "success")
    print(f"\n{success_count}/{len(tester.results)} models successful")

    if success_count == len(tester.results):
        print("\n[SUCCESS] All fixes working! Framework validated.")
    else:
        print("\n[FAIL] Some models failed. Check errors above.")

    print("="*70)


if __name__ == "__main__":
    main()
