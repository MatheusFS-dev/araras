from araras.optuna.utils import get_top_trials
def main():
    print("Testing get_top_trials from araras.optuna.utils")
    try:
        res = get_top_trials("dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
