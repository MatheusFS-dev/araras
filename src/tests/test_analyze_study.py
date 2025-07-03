from araras.optuna.analyze import analyze_study
from tests.helpers import DummyStudy


def main():
    print("Testing analyze_study from araras.optuna.analyze")
    study = DummyStudy()
    try:
        res = analyze_study(study, "tables", 0.1, None, False, False)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
