from araras.optuna.utils import save_top_k_trials
def main():
    print("Testing save_top_k_trials from araras.optuna.utils")
    try:
        res = save_top_k_trials("dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
