from araras.optuna.analyze import plot_param_importances
def main():
    print("Testing plot_param_importances from araras.optuna.analyze")
    try:
        res = plot_param_importances("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
