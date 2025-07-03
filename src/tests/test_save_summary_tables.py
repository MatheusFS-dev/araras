from araras.optuna.analyze import save_summary_tables
def main():
    print("Testing save_summary_tables from araras.optuna.analyze")
    try:
        res = save_summary_tables("dummy", "dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
