from araras.optuna.analyze import format_numeric_value
def main():
    print("Testing format_numeric_value from araras.optuna.analyze")
    try:
        res = format_numeric_value("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
