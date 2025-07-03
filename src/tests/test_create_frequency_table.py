from araras.optuna.analyze import create_frequency_table
def main():
    print("Testing create_frequency_table from araras.optuna.analyze")
    try:
        res = create_frequency_table("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
