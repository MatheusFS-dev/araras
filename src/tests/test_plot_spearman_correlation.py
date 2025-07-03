from araras.optuna.analyze import plot_spearman_correlation
def main():
    print("Testing plot_spearman_correlation from araras.optuna.analyze")
    try:
        res = plot_spearman_correlation("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
