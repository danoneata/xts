import pprint
import sherpa
import os

from train import (
    DATASET,
    get_argument_parser,
    train,
)


def get_study():
    parameters = [
        sherpa.Ordinal("conv3d_num_filters", [16, 32, 64, 128]),
        sherpa.Ordinal("conv3d_kernel_size", [(3, 5, 5), (5, 5, 5), (5, 7, 7)]),
        sherpa.Discrete("encoder_rnn_num_layers", [1, 3]),
        sherpa.Continuous("encoder_rnn_dropout", [0.0, 0.3]),
        sherpa.Continuous("lr", [2e-4, 4e-3], scale="log"),
    ]
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=16)
    stopping_rule = sherpa.algorithms.MedianStoppingRule(min_iterations=8, min_trials=4)
    return sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=True,
        stopping_rule=stopping_rule,
    )


def main():
    parser = get_argument_parser()

    args = parser.parse_args()
    args.batch_size = 8
    args.max_epochs = 64

    study = get_study()

    for i, trial in enumerate(study):
        print("trial id: {}".format(trial.id))
        pprint.pprint(trial.parameters)
        loss = train(args, trial, is_train=False, study=study)
        study.add_observation(trial=trial, objective=loss)
        study.finalize(trial)
        print()

    print(study.get_best_result())

    # save study
    model_name = f"{DATASET}_{args.filelist}_{args.model_type}"
    path = os.path.join("output/models/sherpa", model_name)
    os.makedirs(path, exist_ok=True)
    study.save(path)


if __name__ == "__main__":
    main()
