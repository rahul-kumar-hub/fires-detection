from pathlib import Path
import joblib

# Project root = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "model_package.joblib"

print("=" * 60)
print("TESTING TRAINED MODEL")
print("=" * 60)

print("Model path:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}"
    )

print("\n✓ Model file found")

package = joblib.load(MODEL_PATH)

print("✓ Model package loaded")

print("\nPackage type:")
print(type(package))

if isinstance(package, dict):

    print("\nPackage keys:")
    for key in package.keys():
        print(f"  ✓ {key}")

    # Check important objects
    model = package.get("model")
    features = package.get("features")
    train_medians = package.get("train_medians")
    classes = package.get("classes")

    print("\n" + "-" * 60)
    print("MODEL PACKAGE CHECK")
    print("-" * 60)

    print(
        "Model object found:",
        model is not None
    )

    print(
        "features found:",
        features is not None
    )

    print(
        "train_medians found:",
        train_medians is not None
    )

    print(
        "classes found:",
        classes is not None
    )

    if features is not None:
        print(
            "\nNumber of model features:",
            len(features)
        )

    if classes is not None:
        print(
            "Model classes:",
            classes
        )

    if model is not None and hasattr(
        model,
        "n_features_in_"
    ):
        print(
            "Model expects:",
            model.n_features_in_,
            "features"
        )

else:
    print(
        "\n⚠ The loaded object is not a dictionary."
    )
    print(
        "We need to inspect its structure before continuing."
    )

print("\n" + "=" * 60)
print("MODEL TEST COMPLETE")
print("=" * 60)