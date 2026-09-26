"""
Run this to validate the GaussianStateTransformer on synthetic linear-growth data.
If σ_w and σ_h increase over time, the data foundation is solid.

Usage (from repo root with venv activated):
  PYTHONPATH=Thesis/ode python -m data.validate_synthetic
Or from Thesis/ode:
  python -m data.validate_synthetic
"""


def main() -> None:
    try:
        from .synthetic import validate_transformer_on_synthetic
        from .transforms import GaussianStateTransformer
    except ImportError:
        from synthetic import validate_transformer_on_synthetic
        from transforms import GaussianStateTransformer
    transformer = GaussianStateTransformer()
    ok = validate_transformer_on_synthetic(transformer)
    print("Validation passed:", ok)


if __name__ == "__main__":
    main()
