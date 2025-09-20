import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from multilabel_hierarchical_classifier import MultiLabelHierarchicalClassifier


def records_to_nested_format(records: List[Dict[str, Any]],
                             text_source: str = "title+abstract",
                             categories_field: str = "json_categories") -> List[Dict[str, Any]]:
    """
    Convert list of records (as in data.json) into the nested format required by
    MultiLabelHierarchicalClassifier: {"text": str, "categories": {parent: [children]}}

    Expected input example for each record:
    {
      "authors": "...",
      "title": "...",
      "abstract": "...",
      "json_categories": "{\"hep\": [\"hep-th\"]}"
    }

    Args:
        records: List of dicts loaded from data.json
        text_source: one of {"title", "abstract", "title+abstract"}
        categories_field: field name that holds a JSON-string or dict mapping parent->list(children)

    Returns:
        List of items in nested format.
    """
    nested: List[Dict[str, Any]] = []

    for rec in records:
        # Build text
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()
        if text_source == "title":
            text = title
        elif text_source == "abstract":
            text = abstract
        else:  # "title+abstract"
            text = (title + "\n" + abstract).strip()

        # Parse categories
        cats_raw = rec.get(categories_field, {})
        if isinstance(cats_raw, str):
            cats_raw = cats_raw.strip()
            if cats_raw:
                try:
                    categories = json.loads(cats_raw)
                except json.JSONDecodeError:
                    # If it's not valid JSON, fallback to empty
                    categories = {}
            else:
                categories = {}
        elif isinstance(cats_raw, dict):
            categories = cats_raw
        else:
            categories = {}

        # Ensure categories is dict of parent -> list(children)
        categories = {str(k): list(v) for k, v in categories.items() if isinstance(v, (list, tuple))}

        nested.append({
            "text": text,
            "categories": categories
        })

    return nested


def cmd_train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    data_nested = records_to_nested_format(records,
                                           text_source=args.text_source,
                                           categories_field=args.categories_field)

    clf = MultiLabelHierarchicalClassifier(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        random_state=args.random_state,
    )

    clf.fit(data_nested, validation_split=args.validation_split)

    model_path = Path(args.model_out)
    clf.save(str(model_path))

    print("\n✅ Training done.")
    print(f"💾 Model saved to: {model_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clf = MultiLabelHierarchicalClassifier.load(str(model_path))

    texts: List[str] = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        tf = Path(args.text_file)
        if not tf.exists():
            raise FileNotFoundError(f"Text file not found: {tf}")
        with tf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    if not texts:
        raise ValueError("No input text provided. Use --text or --text-file.")

    if args.with_details:
        results = clf.predict_with_probabilities(texts)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        preds = clf.predict(texts, method=args.method)
        out = []
        for t, p in zip(texts, preds):
            out.append({"text": t, "categories": p})
        print(json.dumps(out, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Label Hierarchical Text Classifier - Train & Predict",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Train model from data.json")
    p_train.add_argument("--data", type=str, default="data.json",
                         help="Path to input JSON records (exported from notebook)")
    p_train.add_argument("--model-out", type=str, default="nested_model.pkl",
                         help="Path to save trained model")
    p_train.add_argument("--text-source", type=str, choices=["title", "abstract", "title+abstract"],
                         default="title+abstract", help="Which fields compose the training text")
    p_train.add_argument("--categories-field", type=str, default="json_categories",
                         help="Field name that contains the nested categories JSON")
    p_train.add_argument("--max-features", type=int, default=5000,
                         help="Max TF-IDF features")
    p_train.add_argument("--ngram-max", type=int, default=2,
                         help="Max n-gram size (1..N)")
    p_train.add_argument("--validation-split", type=float, default=0.2,
                         help="Validation split ratio [0..1]")
    p_train.add_argument("--random-state", type=int, default=42,
                         help="Random seed")
    p_train.set_defaults(func=cmd_train)

    # Predict subcommand
    p_pred = subparsers.add_parser("predict", help="Predict with a saved model")
    p_pred.add_argument("--model", type=str, default="nested_model.pkl",
                        help="Path to saved model")
    p_pred.add_argument("--text", type=str, help="Single text to classify")
    p_pred.add_argument("--text-file", type=str, help="Path to a text file (one text per line)")
    p_pred.add_argument("--method", type=str, choices=["hierarchical", "independent"],
                        default="hierarchical", help="Prediction method")
    p_pred.add_argument("--with-details", action="store_true",
                        help="Output detailed probabilities as well")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
