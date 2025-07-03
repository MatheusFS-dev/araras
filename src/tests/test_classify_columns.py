from araras.optuna.analyze import classify_columns
def main():
    print("Testing classify_columns from araras.optuna.analyze")
    try:
        res = classify_columns("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
