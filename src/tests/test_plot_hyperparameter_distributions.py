from araras.optuna.analyze import plot_hyperparameter_distributions
def main():
    print("Testing plot_hyperparameter_distributions from araras.optuna.analyze")
    try:
        res = plot_hyperparameter_distributions("dummy", "dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
