from araras.optuna.analyze import describe_numeric
def main():
    print("Testing describe_numeric from araras.optuna.analyze")
    try:
        res = describe_numeric("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
