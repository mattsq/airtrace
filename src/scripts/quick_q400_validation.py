"""
Quick validation script - tests a few representative models to verify framework.
"""

import sys
from pathlib import Path

# Reuse the main testing class
sys.path.insert(0, str(Path(__file__).parent))
from run_q400_model_zoo_tests import ModelTester


def main():
    """Quick validation with representative models."""
    print("="*70)
    print("QUICK Q400 FRAMEWORK VALIDATION")
    print("Testing 3 representative models to verify framework")
    print("="*70)

    tester = ModelTester(
        data_config_name="q400",
        transform_config_name="minimal",
        task_config_name="one_step",
        batch_size=256,  # Larger batch for speed
        num_workers=2,
        max_samples=1000,  # Just 1000 samples for quick validation
        results_dir="results/q400_quick_validation"
    )

    # Setup data
    tester.setup_data()

    # Test one baseline (non-trainable)
    print("\n" + "#"*70)
    print("# 1. Baseline Model: Persistence")
    print("#"*70)
    result1 = tester.test_model("persistence", is_trainable=False)
    tester.results.append(result1)

    # Test one simple trainable
    print("\n" + "#"*70)
    print("# 2. Simple Trainable: Linear AR")
    print("#"*70)
    result2 = tester.test_model("linear_ar", is_trainable=True, train_epochs=3)
    tester.results.append(result2)

    # Test one advanced model
    print("\n" + "#"*70)
    print("# 3. Advanced Model: GRU AR")
    print("#"*70)
    result3 = tester.test_model("gru_ar", is_trainable=True, train_epochs=3)
    tester.results.append(result3)

    # Save results
    tester.save_results()

    # Summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    for r in tester.results:
        status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        print(f"\n{status_icon} {r['model_name']}")
        if r["status"] == "success":
            for metric, value in r["metrics"].items():
                print(f"  {metric.upper()}: {value:.4f}")
            print(f"  Runtime: {r['runtime_seconds']:.1f}s")
        else:
            print(f"  Error: {r['error']}")

    success_count = sum(1 for r in tester.results if r["status"] == "success")
    print(f"\n{success_count}/{len(tester.results)} models successful")

    if success_count == len(tester.results):
        print("\n[SUCCESS] Framework validated! Ready for full model zoo testing.")
    else:
        print("\n[WARNING] Some models failed. Review errors above.")

    print(f"\nResults saved to: {tester.results_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
