"""Make a deliberately bad challenger from the exported models, to demo the rollout.

  python -m scripts.make_candidate storm  --out model_candidate_storm
  python -m scripts.make_candidate silent --out model_candidate_silent

storm   flags every reading     (the rollout should reject it)
silent  flags no reading at all (the rollout should hold it for review)
"""

import argparse

from src.model.export import export_models, load_exported

# A reading is flagged when its raw score is below the model's offset_.
# Raw scores are always between -1 and 0, so these two values force each behaviour.
OFFSETS = {"storm": 0.0, "silent": -10.0}


def make_candidate(models, kind):
    """Change the alert threshold of every model, in place. Returns the models."""
    for model in models.values():
        model.offset_ = OFFSETS[kind]
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=sorted(OFFSETS))
    parser.add_argument("--source", default="model_store")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    models, _ = load_exported(args.source)
    make_candidate(models, args.kind)
    export_models(models, args.out, run_id=f"demo-{args.kind}")
    print(f"wrote a '{args.kind}' challenger to {args.out}")


if __name__ == "__main__":
    main()