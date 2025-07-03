from araras.optuna.utils import save_trial_params_to_file
def main():
    print("Testing save_trial_params_to_file from araras.optuna.utils")
    try:
        res = save_trial_params_to_file("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
