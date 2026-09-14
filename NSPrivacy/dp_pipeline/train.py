"""Command-line entry point for the candidate NSPrivacy private trainer."""

import argparse

from nsprivacy_dp import TrainConfig, evaluate, load_npz, save_run, train_private


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--delta", required=True, type=float)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--erase-sigma", type=float, default=0.05)
    parser.add_argument("--lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--lambda-design", type=float, default=1e-4)
    parser.add_argument("--warm-epochs", type=int, default=10)
    parser.add_argument("--public-seed", type=int, default=0)
    parser.add_argument("--private-seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs="+",
        default=None,
        help="Fixed public weights in encoded class order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train, x_test, y_test = load_npz(args.data)
    config = TrainConfig(
        input_dim=x_train.shape[1],
        num_classes=args.num_classes,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        erase_sigma=args.erase_sigma,
        lambda_sparse=args.lambda_sparse,
        lambda_design=args.lambda_design,
        warm_epochs=args.warm_epochs,
        public_seed=args.public_seed,
        private_seed=args.private_seed,
        class_weights=args.class_weights,
        **({"device": args.device} if args.device else {}),
    )
    model, audit = train_private(x_train, y_train, config)
    metrics = None
    if x_test is not None and y_test is not None:
        metrics = evaluate(model, x_test, y_test)
    save_run(args.output, model, config, audit, metrics)
    print(audit)
    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    main()
