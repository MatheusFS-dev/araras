from araras.optuna.analyze import save_data_for_latex
def main():
    print("Testing save_data_for_latex from araras.optuna.analyze")
    try:
        res = save_data_for_latex("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
