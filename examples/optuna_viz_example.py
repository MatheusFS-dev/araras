from araras.commons import *
import optuna

from araras.optuna import report

# Create the study outside of the objective so we can pass it to `report`
study = optuna.create_study(direction="minimize")


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    value = x ** 2 + y ** 2

    # Update the realtime report with custom metrics and summary values
    report(
        study,
        metrics={"loss": lambda t: t.value},
        summary_values={"Completed trials": lambda s: len([t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE])},
    )

    return value


if __name__ == "__main__":
    study.optimize(objective, n_trials=20)

