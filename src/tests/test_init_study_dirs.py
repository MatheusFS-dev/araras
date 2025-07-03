from araras.optuna.utils import init_study_dirs
def main():
    print("Testing init_study_dirs from araras.optuna.utils")
    try:
        res = init_study_dirs("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
