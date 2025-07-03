from araras.optuna.utils import get_remaining_trials
def main():
    print("Testing get_remaining_trials from araras.optuna.utils")
    try:
        res = get_remaining_trials("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
