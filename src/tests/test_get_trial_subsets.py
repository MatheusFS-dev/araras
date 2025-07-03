from araras.optuna.analyze import get_trial_subsets
def main():
    print("Testing get_trial_subsets from araras.optuna.analyze")
    try:
        res = get_trial_subsets("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
