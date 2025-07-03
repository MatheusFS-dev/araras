from araras.optuna.analyze import prepare_dataframe
def main():
    print("Testing prepare_dataframe from araras.optuna.analyze")
    try:
        res = prepare_dataframe("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
