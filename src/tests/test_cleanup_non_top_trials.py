from araras.optuna.utils import cleanup_non_top_trials
def main():
    print("Testing cleanup_non_top_trials from araras.optuna.utils")
    try:
        res = cleanup_non_top_trials("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
