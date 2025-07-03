from araras.optuna.analyze import print_study_columns
def main():
    print("Testing print_study_columns from araras.optuna.analyze")
    try:
        res = print_study_columns("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
