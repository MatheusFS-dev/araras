from araras.optuna.analyze import plot_parameter_boxplots
def main():
    print("Testing plot_parameter_boxplots from araras.optuna.analyze")
    try:
        res = plot_parameter_boxplots("dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
